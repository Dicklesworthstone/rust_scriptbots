# Plan to Rearchitect and Revive Rust ScriptBots

**Status:** [Currently In Progress — Codex, 2026-07-11]

**Evidence date:** 2026-07-11

**Primary outcome:** turn the repository into a deterministic, genuinely evolving artificial-life laboratory with one correct simulation runtime, a polished FrankenTUI interface, a real GPU desktop interface, trustworthy replay/experiments, and tests that exercise the actual shipped paths.

**Relationship to older plans:** this document supersedes the execution order and completion claims in the older port, rendering, Bevy, and WASM plans. Those documents remain valuable historical context. A checked box in an older plan is not evidence that a feature works. Current source, executable tests, and the acceptance gates below are authoritative.

**Persistence decision (2026-07-11):** FrankenSQLite is the only embedded database backend. The former DuckDB architecture is retired in full; no dual-backend abstraction, compatibility shim, or legacy file migration is retained. The migration is complete only when source, manifests, lockfile, tests, CLI/help, CI, and active documentation all use the exact-revision `fsqlite` public facade and a repository-wide search finds no live DuckDB integration.

---

## 1. Why This Rebuild Exists

The repository contains a remarkable amount of ambitious code, but it does not yet form one coherent application. The core simulator, GPUI renderer, Bevy renderer, TUI, server, replay layer, storage layer, and web target were developed as parallel feature islands. They compile against some combinations of features, but their ownership models and behavioral contracts disagree.

That is why adding visual polish did not make the application feel alive. The most important failures happen below the pixels:

- the renderers independently own simulation time;
- GPUI can stop the simulation when repainting stops;
- two GPUI windows can both advance the same world;
- a TUI single-step can advance twice;
- headless TUI commands are never drained;
- ordinary offspring lose their brains;
- the default SIMD eyesight calculation suppresses most batched vision;
- combat reads the wrong neural output as boost;
- replay events are defined but never emitted;
- lifecycle records are discarded between persistence intervals;
- the macOS GUI launcher selects the TUI;
- the visual tests render substitute CPU images rather than the live GUI pipelines;
- all PNG goldens are globally ignored and absent;
- mutable Git dependencies and an ignored lockfile make builds change over time.

This plan fixes those foundations first. The visual redesign follows immediately after the runtime has a single source of truth, because then both interfaces can be cool without becoming new clocks, new simulators, or new sources of nondeterminism.

---

## 2. Product Thesis: ScriptBots Evolution Lab

The revived product is not merely “the old C++ simulation in Rust.” It is an **Evolution Lab**: a tool for watching, perturbing, comparing, and explaining emergent populations.

The satisfying loop is:

1. Choose a curated ecological scenario or define an experiment.
2. Start from a visible seed and a reproducible run manifest.
3. Watch agents sense, move, compete, cooperate, reproduce, mutate, and die.
4. Select an organism and inspect its brain, senses, lineage, energy budget, and recent decisions.
5. Intervene through explicit commands whose application tick is acknowledged.
6. Compare populations, brain families, or parameter variants.
7. Rewind or replay a run and obtain the same state hashes.
8. Export a scientifically useful artifact: metrics, lineage graph, event log, screenshot, or run bundle.

### 2.1 What “useful and interesting” means

The application is useful when it can answer questions such as:

- Did a mutation actually improve survival, or did food availability change?
- Which lineage dominates, and where did it diverge?
- Do MLP, DWRAON, and Assembly brains discover different strategies?
- Which sensors drove the selected agent’s last action?
- When did a population collapse begin?
- Can the same scenario reproduce the collapse from the same seed?
- What changes when one parameter is varied across otherwise identical runs?
- Can an external agent inspect and intervene through a truthful control API?

The application is interesting when the default launch already tells a story. A blank 6000x3000 world with sixteen identical agents clustered in one corner, no initial food, and most ecological systems disabled does not meet that bar.

### 2.2 Default first-run experience

The default `meadow` scenario should:

- seed agents across the full world rather than in a 4x4 corner grid;
- use only real, evolvable brain families;
- start with visible food, varied terrain, and meaningful ecological pressure;
- use a fixed disclosed seed unless the user asks for a random one;
- begin at tick zero without a hidden 120-tick bootstrap;
- show a short onboarding overlay and the command palette;
- make births, deaths, predation, and behavioral variation observable within a few minutes;
- expose the run seed, config hash, code revision, and deterministic status in the UI.

### 2.3 Curated scenario catalog

The initial catalog should contain:

| Scenario | Scientific story | Required visible behavior |
|---|---|---|
| `meadow` | Balanced introductory ecosystem | exploration, grazing, reproduction, occasional combat |
| `predator_prey` | Diet specialization under resource pressure | recognizable predator/prey population cycles |
| `islands` | Spatial isolation and divergence | separate lineages adapt on disconnected fertile regions |
| `boom_bust` | Resource overshoot | population expansion followed by measurable scarcity |
| `cold_front` | Temperature preference and migration | movement and selection along a temperature gradient |
| `brain_arena` | Brain-family comparison | equal seeded cohorts and comparable survival metrics |
| `closed_ecosystem` | No external population injection | extinction or equilibrium is allowed and explained |
| `hydrology` | Terrain, moisture, and food coupling | water/terrain changes visibly affect resources and paths |

Every scenario is data, not a partial mutation of defaults hidden in code. Each scenario includes a schema version, seed policy, full config, initial population recipe, terrain recipe, brain roster, expected invariants, and human description.

---

## 3. Evidence-Based Current-State Audit

The following findings are anchored in the current source tree. Line numbers are evidence pointers for the 2026-07-11 snapshot and may move during implementation.

### 3.1 Runtime ownership is split across renderers

`RendererContext` gives each renderer the raw shared world, storage, command drain, and command submitter (`crates/scriptbots-app/src/lib.rs:16-35`). That makes the presentation layer responsible for domain progress.

The three implementations disagree:

- GPUI calls `WorldState::step` from `SimulationView::pump_simulation`, reached through render-time snapshot creation (`crates/scriptbots-render/src/lib.rs:1367-1442`, `1793-1795`, `7972-7977`).
- TUI drains and steps from its event loop and also directly steps in `step_once` (`crates/scriptbots-app/src/terminal/mod.rs:301-366`).
- Bevy starts its own simulation driver and sends snapshots over its own channel (`crates/scriptbots-bevy/src/lib.rs:69-100`).

There is no renderer-independent authority for tick cadence, pause, speed, single-step, command order, or shutdown.

### 3.2 GPUI can both stall and overrun the simulation

The GPUI simulation advances only while a view is being rendered. No independent timer or host requests simulation ticks. If the window is occluded or nothing invalidates it, simulation progress can stop.

`run_demo` creates two separate `SimulationView` instances over the same shared world (`crates/scriptbots-render/src/lib.rs:1197-1257`). Each view owns its own accumulator and controls (`1268-1345`) and calls the same world step method. The number and repaint cadence of windows therefore affect simulated time.

The minimal world window does not share the HUD window’s keyboard handler. Pausing one view does not prove that the other view stopped stepping.

### 3.3 GPUI control commands are not authoritative

GPUI actions update local view controls and enqueue `SimulationCommand` values (`crates/scriptbots-render/src/lib.rs:3746-3758`, `3804-3816`). Core only stores those requests until a driver calls `drain_simulation_commands` (`crates/scriptbots-core/src/lib.rs:221-235`, `8116-8135`). GPUI never performs that drain. REST or MCP pause/speed requests therefore cannot reliably control the GPUI clock.

### 3.4 TUI command semantics are broken

`TerminalApp::step_once` submits a single-step command and directly calls `world.step` (`crates/scriptbots-app/src/terminal/mod.rs:356-366`). In the live loop, the queued command is drained later and forces another tick. One `s` keypress can become two ticks.

Headless mode repeatedly calls the same direct-step method but never executes the live command drain (`crates/scriptbots-app/src/terminal/mod.rs:147-169`, `301-354`). External config/control commands are not applied, while the bounded command queue can fill after 32 entries (`crates/scriptbots-app/src/command.rs:39-53`, `servers.rs:215-224`).

The 96-frame headless integration test prequeues an update and expects it to be applied (`crates/scriptbots-app/tests/terminal_end_to_end.rs:322-335`, `424-455`, `537-557`). Under the current control flow that expectation is impossible.

### 3.5 macOS GUI mode is misdetected

`run_macos_version_with_gui.sh` passes `--mode gui` but does not set the force-GUI environment override (`run_macos_version_with_gui.sh:21-35`). `should_use_terminal_mode` treats every Unix system without `DISPLAY` or `WAYLAND_DISPLAY` as headless (`crates/scriptbots-app/src/main.rs:944-967`). Native macOS does not normally use those variables, so the GUI launcher falls back to the TUI.

### 3.6 default feature and mode contracts disagree

`scriptbots-app` default features omit `gui` (`crates/scriptbots-app/Cargo.toml:53-59`). Auto mode can still select `GuiRenderer` on a desktop and then fail because the feature is unavailable (`crates/scriptbots-app/src/main.rs:880-919`).

`run_demo` returns `()` and logs window-open failures internally (`crates/scriptbots-render/src/lib.rs:1169-1175`, `1213-1225`, `1242-1257`). The app then returns success, so the promised fallback cannot observe a launch failure.

### 3.7 current GPUI/WGPU composition is not a viable live path

`GpuiImage` is a raw RGBA vector (`crates/scriptbots-render/src/lib.rs:46-53`). Presenting it scans pixels and emits one GPUI quad per equal-color run (`86-141`). Detailed frames can approach one quad per pixel.

The WGPU readback path waits indefinitely for device completion (`crates/scriptbots-world-gfx/src/lib.rs:437-485`), turning a supposed asynchronous ring into a UI-thread stall. Both differently sized GPUI windows share one static compositor (`crates/scriptbots-render/src/lib.rs:943-949`), causing resize/recreation churn.

The WGPU path is disabled unless `SB_RENDERER=wgpu` (`crates/scriptbots-render/src/lib.rs:924-935`), but the platform GUI scripts do not set that variable. Their WGPU tuning is therefore applied while the CPU canvas runs.

### 3.8 Bevy currently cannot scale to its stated target

Bevy uses an unbounded snapshot channel and repeatedly clones terrain arrays and all agents (`crates/scriptbots-bevy/src/lib.rs:69-100`, `953-999`, `1087-1265`).

Each simulated agent expands into roughly 23 ECS entities and 22 separately allocated materials (`crates/scriptbots-bevy/src/lib.rs:237-254`, `320-370`, `3211-3464`). Ten thousand agents imply roughly 230,000 entities and 220,000 materials, before effects or UI. The target must move to instanced/batched GPU data.

Additional concrete defects include:

- health is normalized against 100 rather than the simulation’s 0-2 range (`crates/scriptbots-bevy/src/lib.rs:3988-4000`);
- terrain change detection ignores kinds, fertility, temperature, and spatial rearrangement while still rebuilding meshes (`1061-1084`, `2552-2610`);
- reflection-probe handles are empty but installed on chunks (`1325-1328`, `2720-2760`);
- Escape is documented and implemented as clear-selection in one system, then exits the app in another (`1519-1532`, `1865-1890`, `4016-4019`).

### 3.9 the core simulation is not currently a faithful evolutionary system

The default SIMD eye calculation accepts `dot >= cos(fov)` and then computes a factor proportional to `cos(fov) - dot`, which is non-positive for accepted vectors and clamps to zero (`crates/scriptbots-core/src/lib.rs:4924-4962`, `5173-5186`). Batched neighbors are effectively invisible. The scalar fallback also adds heading twice to a precomputed view direction (`5217-5227`). The original C++ computes a positive angular falloff (`original_scriptbots_code_for_reference/World.cpp:241-259`).

Ordinary offspring lose their neural state:

- `BrainBinding::clone` retains metadata but drops the runner (`crates/scriptbots-core/src/lib.rs:556-563`);
- `build_child_runtime` resets the child brain to default/unbound (`7173-7195`);
- `stage_spawn_commit` installs that unbound runtime and never rebinds it (`7083-7120`).

The original C++ copies, mutates, and optionally crosses the actual brain (`original_scriptbots_code_for_reference/Agent.cpp:142-177`). In the Rust port, brain mutation/crossover exist on `scriptbots_brain::Brain`, but the object-safe `scriptbots_core::BrainRunner` exposes only `kind`, `tick`, and activation snapshots (`crates/scriptbots-core/src/lib.rs:657-670`). The adapter discards the evolution operations.

Combat decodes boost from neural output 6 during actuation but reads output 3 while calculating spike damage (`crates/scriptbots-core/src/lib.rs:5377-5397`, `6410-6423`). Output 3 is a color channel, so changing color can change combat damage.

### 3.10 advertised brain families are not actually installed

`install_brains` registers MLP, the optional ML wrapper, and optional NeuroFlow (`crates/scriptbots-app/src/main.rs:1776-1815`). It does not register DWRAON or Assembly even though they are compiled by default in `scriptbots-brain` and documented as active competitors.

The default `ml` feature registers `scriptbots-brain-ml`, whose implementation labels itself `ml.placeholder`, copies sensors to outputs, and implements no mutation (`crates/scriptbots-brain-ml/src/lib.rs:1-78`). A placeholder is therefore part of the default evolving population.

### 3.11 replay and persistence are largely ceremonial

`ReplayEvent` types and storage codecs exist, but no core code pushes into `WorldState::replay_events`. Replay verification can report that zero recorded events match zero simulated events.

`stage_persistence` clears births, deaths, and combat counters on every tick that is not a persistence interval (`crates/scriptbots-core/src/lib.rs:7511-7523`). Events between intervals are discarded rather than accumulated. History is appended only when persistence runs (`8063-8066`), so UI headline metrics can be stale for the default 60-tick interval.

### 3.12 control APIs acknowledge queueing as application

`ControlHandle::apply_patch` builds and returns a config snapshot at the current tick before the queued update is applied (`crates/scriptbots-app/src/control.rs:409-445`). A successful response means “accepted into a queue,” not “applied,” but the API presents it as current state.

The knobs cache uses `config_audit().len()` as a revision (`crates/scriptbots-app/src/control.rs:274-296`). The core audit is capped at 64 (`crates/scriptbots-core/src/lib.rs:8190-8198`). After saturation, cache revision can stop changing and knob values can remain stale forever.

The ASCII screenshot endpoint returns only `ScriptBots snapshot tN` (`crates/scriptbots-app/src/servers.rs:690-705`). It is not a screenshot.

### 3.13 current visual tests do not test the live visuals

Bevy’s `render_png_offscreen` is a standalone `image::ImageBuffer` rasterizer (`crates/scriptbots-bevy/src/lib.rs:4022-4088`). It does not run a Bevy app, scene, camera, PBR pipeline, HUD, or GPU renderer.

The GPUI golden test likewise calls a separate CPU rasterizer (`crates/scriptbots-render/src/lib.rs:7752`; `crates/scriptbots-render/tests/snapshot.rs:45-71`).

All PNGs are globally ignored (`.gitignore:18`). The checksum file references `golden/rust_default.png` and `golden/bevy_default.png`, but neither exists. CI invokes a nonexistent `scriptbots-bevy` feature named `bevy_render` (`.github/workflows/ci.yml:358-369`).

The TUI headless backend draws frames but tests only numeric reports and log strings. No test examines buffer cells, responsive layout, Unicode width, input transitions, or terminal restoration.

### 3.14 dependency and toolchain resolution are not reproducible

The workspace ignores `Cargo.lock` (`.gitignore:3`) even though the primary deliverable is an application. The toolchain is mutable `nightly`, GPUI tracks Zed’s mutable `main` branch, and dependency declarations are inconsistent between workspace and crate manifests.

The current host resolution pulled more than one thousand packages and multiple versions of WGPU, Crossterm, Unicode Width, TOML, and other libraries. A clean checkout can resolve differently months later.

### 3.15 monolith census

The largest files are both architectural and mechanical risk centers:

| File | Approximate lines | Mixed responsibilities |
|---|---:|---|
| `crates/scriptbots-render/src/lib.rs` | 12,724 | runtime clock, snapshots, compositor, canvas, HUD, input, capture, tests |
| `crates/scriptbots-core/src/lib.rs` | 10,650 | models, config, terrain, simulation stages, persistence, replay, tests |
| `crates/scriptbots-bevy/src/lib.rs` | 4,655 | bridge, scene, terrain, agents, camera, HUD, capture |
| `crates/scriptbots-app/src/terminal/mod.rs` | 3,297 | runtime, projection, widgets, reports, theme, input, tests |
| `crates/scriptbots-world-gfx/src/lib.rs` | 2,331 | device, pipelines, shaders, readback, tests |
| `crates/scriptbots-app/src/main.rs` | 2,237 | CLI, config, bootstrap, render selection, replay, profiling |

The repeated seam is more important than line count: GPUI, Bevy, and TUI each build their own world snapshot and implement their own tick, pause, speed, selection, health/color, terrain, and capture semantics.

---

## 4. Non-Negotiable Invariants

Every implementation bead must preserve or establish the following invariants.

### 4.1 Simulation invariants

1. Exactly one runtime owns and mutates `WorldState`.
2. Rendering, repaint count, window count, window occlusion, terminal refresh, and API polling cannot advance ticks.
3. A single-step command advances exactly one tick and leaves the world paused.
4. Commands are applied in a deterministic total order at tick boundaries.
5. Every admitted state-changing command reaches a queryable terminal `Applied`, `Durable`, `Rejected`, or `Failed` status; admission, scientific application, and crash durability are distinct.
6. Command backpressure is explicit. State-changing commands are never silently dropped.
7. Slow renderers may miss intermediate UI snapshots and event notifications, but scientific commands/events remain lossless in the sequenced journal and subscribers can detect lag/catch up.
8. A fixed run manifest and command trace produce the same periodic state hashes across null, TUI, GUI, server-only, and replay modes when run in the same bit-exact determinism lane.

### 4.2 Evolution invariants

1. Every live agent has either an intentional no-brain policy or a real brain binding; accidental unbound fallback is an error metric.
2. Offspring inherit, mutate, or cross parent genomes and receive dynamic evaluator state according to an explicit family policy.
3. Brain kind, genome/state, lineage, mutation parameters, and provenance are serializable.
4. Placeholder brain implementations cannot enter default or scientific scenarios.
5. Scalar and SIMD sensor paths agree within a documented tolerance.
6. Neural output channel meanings are defined once and shared by actuation, combat, replay, UI, and tests.

### 4.3 UI invariants

1. Frontends consume immutable snapshots and send typed intents.
2. Frontends do not hold a world mutex and do not query FrankenSQLite during paint.
3. UI state distinguishes `admitted`, `applied`, `durable`, `rejected`, `failed`, `disconnected`, and `stale`.
4. Every documented shortcut is covered by a state-transition test.
5. Terminal layout is valid at declared minimum sizes and width-correct under capability profiles.
6. GUI screenshots come from the actual live render path.

### 4.4 Evidence invariants

1. A passing test must exercise the implementation named by the test.
2. Golden assets are tracked, versioned, and never auto-blessed in CI.
3. Replay tests must contain nonzero commands/events and compare state hashes, not merely empty lists.
4. Performance claims include a command, dataset, machine/context, and saved result.
5. Documentation completion claims cite executable evidence.

### 4.5 Refactor invariants

1. Behavioral fixes and mechanical file moves are separate beads and separate commits.
2. Each extraction preserves public paths through a façade/re-export until the planned API change bead.
3. Feature/cfg matrices are captured before and after each extraction.
4. No file is deleted without the user’s explicit written permission.
5. De-monolithization follows the named skill’s Standard-mode characterization and isomorphism gates.

---

## 5. Target Architecture

```text
                         +----------------------+
scenario / experiment -->| RunManifest          |
                         | seed, config, brains |
                         +----------+-----------+
                                    |
                                    v
control clients ---> bounded CommandEnvelope queue
REST / MCP / TUI / GUI       id + expected revision + reply
                                    |
                                    v
                    +---------------+----------------+
                    | SimulationHost                  |
                    | sole WorldState owner           |
                    | fixed clock / pause / step      |
                    | deterministic command ordering  |
                    | domain events / state hashes    |
                    +-------+---------------+---------+
                            |               |
                 latest Arc |               | ordered batches
                            v               v
                     RenderSnapshot    StorageWorker
                     revisioned data   FrankenSQLite / run bundle
                       /    |    \
                      /     |     \
                     v      v      v
                FrankenTUI Bevy   server screenshots
                 frontend   GUI   / web projection
```

The central change is not a new framework. It is an ownership contract: the simulation exists independently of every presentation backend.

### 5.1 Proposed component boundaries

The exact crate boundary is proven through implementation, but the logical modules are:

- `simulation_host`: runtime lifecycle, clock, command ordering, receipts, shutdown;
- `run_manifest`: scenario, seed, version, dependency/code metadata, brain roster;
- `snapshot`: immutable renderer-neutral state and revisioned static layers;
- `projection`: bounded UI-specific maps, top-K tables, selected-agent detail, charts;
- `domain_events`: commands, births, deaths, combat, config, checkpoints, hashes;
- `brain_evolution`: versioned genome/evaluator-state families, mutation/crossover, checkpointing, and optional family-owned batch arenas;
- `storage_worker`: blocking FrankenSQLite boundary and run-bundle export;
- `frontends::terminal`: FrankenTUI model, view, actions, capability handling;
- `frontends::bevy`: instanced world renderer, camera, selection, actual capture;
- `frontends::gpui`: time-boxed dashboard/direct-texture feasibility adapter;
- `control_api`: request/receipt/read-model API and MCP tools;
- `experiments`: sweeps, matched seeds, comparisons, reports.

Creating a new crate is justified only when it enforces a dependency boundary used by multiple frontends. The likely justified addition is a small `scriptbots-runtime` crate that depends on core and exposes the host/control/snapshot protocol without depending on GPUI, Bevy, Ratatui, or Axum. This decision must be confirmed by the dependency-direction bead before creation.

### 5.2 `SimulationHost`

`SimulationHost` is the only owner of `WorldState`. It exposes handles rather than the world itself:

```rust,ignore
pub struct SimulationClient {
    commands: CommandSender,
    snapshots: SnapshotSubscription,
    events: EventJournalCursor,
    status: StatusReader,
}

pub struct CommandEnvelope {
    pub id: CommandId,
    pub expected: ExpectedRevisions,
    pub command: SimulationCommand,
    pub status: CommandStatusSender,
}

pub enum CommandStatus {
    Admitted { id: CommandId, admission_seq: u64 },
    Applied { id: CommandId, tick: Tick, scientific_revision: u64 },
    Durable { id: CommandId, event_seq: u64 },
    Rejected { id: CommandId, reason: CommandRejection },
    Failed { id: CommandId, reason: CommandFailure },
}
```

`CommandId`, host admission sequence, scientific revision, config revision, snapshot revision, and event sequence are distinct types. The host deduplicates retrying `CommandId`s, preserves queryable status after client disconnect, and defines whether an expected revision applies to scientific state, config, or both. A single-step command received while running atomically pauses, advances once, and remains paused. Shutdown uses a formally ordered admission rule rather than bypassing total order invisibly.

The concrete channel types may use Asupersync, but the protocol is not coupled to a runtime. A synchronous `HostCore`/`SimulationEngine` owns command/state transitions under injected time. Native Asupersync or a dedicated scheduler drives it; a browser adapter can drive the same state machine without requestAnimationFrame becoming scientific time. Async infrastructure handles lifecycle, control I/O, storage, and blocking isolation around the synchronous state machine.

This ownership claim has a strict prerequisite: core stepping must stop performing external side effects. Today `WorldState` owns `Box<dyn WorldPersistence>`, stores simulation-command queues, and calls persistence from inside `step`. Before the host is introduced:

- playback/control queue ownership moves out of core;
- `WorldState::step` stops calling storage or other external services;
- validated domain mutations are invoked explicitly at tick boundaries;
- a step returns a deterministic value such as `StepOutcome` containing its summary, domain events, lifecycle records, dirty/revision signals, and persistence projection inputs;
- storage sampling/batching happens in the host/runtime layer after the scientific state transition;
- an I/O failure cannot change the result of the completed scientific tick.

```rust,ignore
pub struct StepOutcome {
    pub tick: Tick,
    pub summary: TickSummary,
    pub events: Vec<DomainEvent>,
    pub births: Vec<BirthRecord>,
    pub deaths: Vec<DeathRecord>,
    pub revisions: WorldRevisions,
    pub digest: Option<WorldDigest>,
}
```

The exact payload is profiled so it does not clone a full render snapshot. This boundary is a prerequisite bead for `SimulationHost`, not an implementation detail hidden inside it.

### 5.3 tick-boundary algorithm

For each host iteration:

1. Observe cancellation and lifecycle state.
2. Drain up to the documented command budget in queue order.
3. Assign/observe admission sequences, deduplicate IDs, and validate the correct expected revision domains.
4. Apply config/control mutations at a tick boundary.
5. Complete rejected receipts immediately.
6. Determine the exact requested step count from run/pause/speed/single-step state.
7. Execute deterministic world steps without awaiting or performing I/O inside the step.
8. Consume the returned `StepOutcome`, emit its domain events, and compute/record periodic state hashes.
9. Build and hand a bounded persistence batch to the storage worker outside core.
10. Publish one latest-value snapshot when its cadence or revision requires it.
11. Publish applied status with the actual tick/scientific revision; publish durable status only after the corresponding journal record is acknowledged by storage.
12. Wait until the next fixed deadline or new command, without allowing missed deadlines to create unbounded catch-up.

Speed changes affect how many deterministic ticks are scheduled per wall-clock interval. They never change the internal calculation of one tick.

### 5.4 command classes

Commands are classified so overload behavior is truthful:

| Class | Examples | Backpressure policy |
|---|---|---|
| State-changing | config patch, apply scenario, spawn, kill, reset | bounded, acknowledged, never silently dropped |
| Playback | pause, resume, speed, step | bounded and coalesced only by an explicit semantic rule |
| Selection/UI | select agent, follow mode | latest-value per client, does not mutate simulation science |
| Query | status, config, snapshot | served from immutable read model |
| Shutdown | graceful stop | prioritized, acknowledged, idempotent |

### 5.5 immutable snapshot model

`RenderSnapshot` is renderer-neutral and revisioned:

- current tick and current tick summary, independent of persistence cadence;
- run state, speed, queue depth, last applied command, health flags;
- compact agent structure-of-arrays or a packed immutable buffer;
- client-neutral compact agent data plus separately requested/keyed detail payloads and brain activation snapshots;
- current food projection or revisioned tiles;
- terrain/hydrology layers behind `Arc` values that change only on revision;
- recent domain-event ring;
- chart/downsample series;
- run manifest identity and deterministic state hash.

Dynamic agent data may be rebuilt at UI cadence. Static terrain and configuration must be shared by revision instead of cloned every frame. A multi-subscriber latest-value hub gives each frontend an independent cursor over an atomic/shared `Arc`; a stalled consumer does not retain an unbounded queue. Frontends are allowed to drop stale snapshots and consume the newest one.

Scientific events use different semantics: they live in a sequenced journal. UI subscriptions receive bounded notifications/cursors, detect gaps, and catch up from the journal or explicitly report that their display window was truncated. Snapshot loss never implies scientific event loss.

### 5.6 projection model

Shared pure projection functions or a keyed projection broker perform work that is currently duplicated under world locks:

- spatial binning into viewport/map cells;
- top predators, oldest agents, and lineage leaders;
- selected-agent senses, outputs, brain layers, and ancestry;
- decimated charts and event timeline;
- terrain/food glyph or color classification;
- capability-neutral labels and semantic colors.

Viewport size, camera, selected agent, requested detail, and chart window are per-client inputs, not mutable global projection state and not scientific world state. The TUI, GUI HUD, server ASCII capture, and web snapshot reuse semantic projection functions and caches keyed by request/revision. They may render the result differently.

### 5.7 Asupersync adoption boundary

Asupersync is a strong fit for:

- bounded two-phase MPSC control and worker channels;
- structured lifecycle/cancellation of host, storage, API, and capture tasks;
- `Cx::spawn_blocking` around synchronous FrankenSQLite work;
- explicit shutdown obligations and leak detection;
- deterministic-mode tests of task orchestration.

It must not turn the deterministic world tick into a collection of independently scheduled async tasks. The world step remains synchronous and ordered. Tokio/Axum replacement is a later, evidence-driven migration; it is not bundled into the first host extraction.

### 5.8 renderer decision

The plan treats renderer selection as an evidence gate, not loyalty to the historical plan.

- **FrankenTUI** is the target terminal frontend.
- **Bevy** is the leading rich GUI candidate because it can present directly to the GPU and already has a 3D scene, but its per-agent ECS/material design must be replaced by instancing/batching.
- **GPUI** receives one time-boxed direct-texture feasibility spike. If a pinned GPUI API cannot present the world without GPU readback and massive quad reconstruction, GPUI becomes an optional control/dashboard frontend or is retired after explicit deletion permission.
- **Custom WGPU** remains valuable as a batched world renderer or Bevy render resource, but readback is reserved for screenshots and tests.

The primary GUI decision is made from live-path correctness, frame-time, memory, capture, input, and maintenance evidence.

---

## 6. Scientific Kernel Redesign

The simulator must become trustworthy before its output is treated as research data. This phase begins with oracle tests for known defects, then makes the smallest semantic fixes, then introduces the new brain/RNG/replay contracts.

### 6.1 legacy micro-oracle harness

The preserved C++ implementation is not copied wholesale, and every historical behavior is not automatically desirable. It is used as a local oracle for mechanics that the Rust port claims to preserve.

Create focused fixtures for:

- eye view direction, FOV falloff, density, and color channels;
- blood sensor half-FOV and wounded-target scaling;
- differential-drive heading and movement;
- neural output channel mapping;
- food intake, health/energy changes, and metabolic drains;
- spike reach, facing, cost, boost, and damage;
- reproduction gate and resource accounting;
- brain copy, mutation, and crossover behavior;
- toroidal distance and boundary wrapping.

The oracle harness records inputs and expected semantic outputs in small, reviewable fixtures. It does not require running the legacy OpenGL application in CI.

### 6.2 sensing correctness

The sensing bead must first add a red test that forces more than four neighbors through the SIMD chunk path. The test must fail under the current inverted factor and must not pass merely because a remainder neighbor was visible.

Required cases:

- one target centered in an eye;
- target exactly at FOV boundary;
- target just inside and outside boundary;
- target across a toroidal edge;
- at least eight visible targets so both full chunks and remainder execute;
- rotated subject with nonzero eye direction;
- scalar build without `simd_wide`;
- SIMD build with `simd_wide`;
- serial and Rayon collection paths;
- color accumulation and density accumulation;
- zero/near-zero distance exclusion;
- very wide FOV whose cosine is zero or negative.

Use one shared, numerically explicit eye-weight function or equivalent proven formulation. SIMD is an optimization of the scalar meaning, not a separate meaning. Compare both paths to the same oracle.

The blood sensor must receive a separate legacy-parity decision because Rust currently uses `3π/8` while the legacy implementation uses `3π/16`. If the new wider value is intentionally better, name it as a product change and update tests/docs; do not leave it accidental.

### 6.3 neural output contract

Define the nine output channels in one type instead of indexing raw arrays throughout the code:

```rust,ignore
pub struct BrainOutputs {
    pub left_wheel: f32,
    pub right_wheel: f32,
    pub color: [f32; 3],
    pub spike: f32,
    pub boost: f32,
    pub sound: f32,
    pub give: f32,
}
```

The exact storage may remain `[f32; OUTPUT_SIZE]` in hot paths, but accessors/constants establish one mapping. Actuation, combat, rendering, replay, persistence, and brain tests use the same mapping. The combat fix must include a regression in which only green changes and damage remains constant, then only boost changes and damage responds.

### 6.4 resource ledger and viable ecology

Current defaults mix health and energy drains in a way that makes the tiny seeded population fade. The redesign needs an explicit ledger:

- sources: ground food, carcass intake, population injection, scenario intervention;
- transfers: giving and reproduction allocation;
- sinks: basal metabolism, movement, boost, combat, aging, temperature stress;
- conversions: food to energy and, if chosen, health recovery;
- caps and rejected overflow.

For each tick, debug/test instrumentation can explain total resource delta within tolerance. Long-run tests define expected viability or extinction envelopes for each curated scenario.

Default ecology changes are evaluated as a coherent model, not by copying legacy constants blindly. The product must state whether health and energy are separate resources, how food affects each, and what selection pressure reproduction consumes.

### 6.5 brain genome/evaluator-state separation

The current `Brain`/`BrainRunner` bridge loses the operations required by evolution. Replace trait-object cloning with versioned heritable data **and** separately versioned dynamic evaluator state. Evaluator state is not universally ephemeral: recurrent MLP/DWRAON node state and Assembly working cells can affect future outputs and must survive checkpoints when the family contract says so.

```rust,ignore
pub struct BrainGenomeEnvelope {
    pub family_id: BrainFamilyId,
    pub schema_version: u32,
    pub payload: Vec<u8>,
    pub provenance: BrainProvenance,
}

pub struct BrainStateEnvelope {
    pub family_id: BrainFamilyId,
    pub schema_version: u32,
    pub payload: Vec<u8>,
}

pub trait BrainFamily: Send + Sync {
    fn id(&self) -> BrainFamilyId;
    fn random_genome(&self, rng: &mut dyn RandomStream) -> Result<BrainGenomeEnvelope>;
    fn validate(&self, genome: &BrainGenomeEnvelope) -> Result<()>;
    fn mutate(
        &self,
        genome: &BrainGenomeEnvelope,
        rates: MutationRates,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeEnvelope>;
    fn crossover(
        &self,
        left: &BrainGenomeEnvelope,
        right: &BrainGenomeEnvelope,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainGenomeEnvelope>;
    fn initial_state(
        &self,
        genome: &BrainGenomeEnvelope,
        rng: &mut dyn RandomStream,
    ) -> Result<BrainStateEnvelope>;
    fn evaluator(
        &self,
        genome: &BrainGenomeEnvelope,
        state: &BrainStateEnvelope,
    ) -> Result<Box<dyn BrainEvaluator>>;
    fn offspring_state(
        &self,
        child: &BrainGenomeEnvelope,
        parents: &[&BrainStateEnvelope],
        rng: &mut dyn RandomStream,
    ) -> Result<BrainStateEnvelope>;
}

pub trait BrainEvaluator: Send {
    fn evaluate(&mut self, sensors: &BrainInputs) -> BrainOutputs;
    fn inspect(&self, request: BrainInspection) -> Option<BrainActivations>;
    fn checkpoint_state(&self) -> Result<BrainStateEnvelope>;
}
```

Each agent stores the exact genome envelope, versioned evaluator state, and a live evaluator constructed from both. Reproduction operates on genomes. Each family explicitly declares whether child dynamic state is reset, inherited, or blended; there is no accidental trait-object clone policy. Checkpoints persist both envelopes, and `WorldDigest` includes all evaluator state that can affect future behavior.

This removes the need to clone an opaque runner and makes persistence, replay, lineage, stateful-brain behavior, and external inspection truthful.

### 6.6 brain-family acceptance

Before a family can enter a scenario it must prove:

- random genome generation from a supplied stream;
- deterministic evaluation;
- mutation that can change heritable data;
- crossover or an explicitly declared asexual policy;
- serialization and schema validation;
- deterministic initial evaluator-state construction;
- evaluator-state checkpoint and reconstruction;
- explicit child-state reset/inherit/blend policy;
- bounded output values or documented normalization;
- on-demand introspection without serializing every brain every tick;
- lineage provenance in child genomes.

MLP, DWRAON, and Assembly should be the first honest families. Their recurrent/working state receives family-specific checkpoint and offspring tests. NeuroFlow remains optional until its genome/evaluator-state reconstruction and introspection cost are acceptable. Candle/Tract/Tch adapters remain disabled from default scenarios until they load actual models and define reproduction semantics. `ml.placeholder` is removed from production registration.

### 6.7 brain-specific parity work

The brain-family beads must decide and test known divergences:

- DWRAON legacy input-node source behavior;
- DWRAON per-field crossover rather than whole-node selection;
- MLP damping range (`1.0` versus the legacy `1.1` cap);
- Assembly invalid-instruction and mutation bounds;
- NeuroFlow serialization/reconstruction determinism.

Not every legacy detail must be retained. Every divergence must be intentional, named, and observable.

### 6.8 stable logical agent identity

`slotmap::AgentId` remains an efficient in-memory handle, but it is not the stable scientific identity for persisted/replayed lineages. Add a monotonic or seed-derived `AgentUid` that:

- never reuses within a run;
- persists across snapshots and checkpoints;
- appears in command, event, lineage, and analysis records;
- does not depend on dense storage order;
- can derive deterministic per-agent RNG streams.

### 6.9 domain-separated random streams

A single world RNG makes unrelated code changes perturb all future outcomes. Introduce a `RandomStream` abstraction and named domains:

- scenario/environment generation;
- food dynamics;
- population injection;
- each lineage/agent;
- brain mutation;
- brain crossover;
- optional presentation-only effects, which must never consume scientific streams.

Dynamic agent streams derive from `(run seed, domain tag, stable lineage or agent UID, birth ordinal)`. The state required for checkpoints is explicit.

The leading reuse candidate is `franken_numpy`’s `fnp-random`, which provides `SeedSequence`, hierarchical child streams, PCG64DXSM state restoration, and jump-ahead. Adoption is gated by a small adapter/toolchain/build-size spike because the current crate is pre-0.2, pulls ndarray and Rayon, and does not implement `rand::RngCore`. Pin an exact commit; never use a mutable sibling path in released builds.

### 6.10 deterministic digest

Add a canonical `WorldDigest` with stable ordering and explicit float-bit encoding. It contains:

- digest schema/hash algorithm version;
- tick, epoch, closed/run state;
- full scientific config hash;
- terrain/food/hydrology hashes;
- agents sorted by `AgentUid`;
- each agent’s physical state, resource state, lineage, genome hash, future-affecting evaluator state, and RNG state;
- spawn ordinals, stage-local counters, and any buffer/cache state that can influence future transitions.

Also support per-stage hashes after:

1. sensing;
2. brain evaluation;
3. actuation;
4. ecology/food;
5. combat/death;
6. reproduction/population.

The first divergent stage makes replay and cross-feature failures diagnosable.

#### 6.10.1 determinism tiers

Do not conflate bit-exact replay with cross-feature numerical conformance:

- **Bit-exact replay lane:** fixed toolchain, target, dependency lock, math/SIMD path, thread count, iteration/reduction order, RNG algorithm, and hash version. Periodic/final bit digests must match exactly.
- **Same-build fast lane:** a documented subset of thread-count/parallel settings that is proven reproducible; settings outside it are marked non-bit-exact.
- **Cross-feature/cross-target oracle lane:** scalar/SIMD, native/WASM, and platform comparisons use field-specific tolerances and semantic invariants, not an expectation that a chaotic long run ends with identical float-bit hashes.

If exact cross-target behavior becomes a requirement, add an explicit deterministic-math/fixed-point or canonical-quantization architecture bead. Diagnostic per-stage hashing is sampled/test-only so it does not violate hot-path budgets.

### 6.11 metamorphic correctness matrix

In addition to golden traces, prove properties that should hold without a hand-authored oracle:

- toroidal translation invariance;
- rotation invariance where scenario fields rotate too;
- dense storage permutation invariance;
- adding a distant noninteracting agent does not affect existing agents;
- persistence cadence does not change scientific state;
- UI snapshot cadence does not change scientific state;
- scalar/SIMD agreement;
- serial/Rayon agreement under deterministic reductions;
- checkpoint encode/decode idempotence;
- tiny wrapped worlds do not deliver duplicate neighbors;
- all values remain finite under validated configs;
- every child has a bound, valid genome with provenance.

---

## 7. Persistence, Replay, and Experiment Data

### 7.1 event-sourced run manifest

Every run starts with a tracked manifest containing:

- schema version;
- run ID and optional experiment ID/variant ID;
- scenario name/version;
- full normalized config and config hash;
- root seed and RNG algorithm/version;
- brain-family versions and initial roster;
- code commit, source-tree digest, and dirty-state status;
- Rust toolchain and Cargo lock hash;
- enabled feature set and target triple;
- start time as metadata only, never a deterministic input;
- requested tick budget or live-run policy.

The manifest is written before tick zero and is queryable from every frontend.

A strict reproducible run must have a clean source tree or embed an exact reviewed patch/source bundle whose digest matches. A mere dirty flag cannot recreate source. Runs with unbundled dirty code or mutable sibling path dependencies are marked non-reproducible and strict bundle verification refuses them.

### 7.2 command log

Persist each state-changing command with:

- command ID;
- source/client ID;
- host admission sequence;
- expected scientific/config revisions;
- payload and schema version;
- admitted/applied/durable/rejected/failed status transitions;
- actual application tick/scientific revision and durable event sequence.

Replay consumes the manifest plus applied command log. It does not infer configuration from a partially populated database.

### 7.3 domain event log

Record scientifically meaningful events:

- birth with parents, genome hash, family, mutation/crossover provenance;
- death with cause and resource state;
- attack/hit with attacker/victim and damage;
- give/transfer;
- scenario intervention;
- config change;
- checkpoint;
- periodic state/stage digest;
- explicit warnings such as unbound brain or non-finite state.

Do not record every floating-point operation by default. Detailed trace mode is opt-in and bounded.

### 7.4 replay success criteria

A replay succeeds only if:

- at least one expected event or checkpoint is present for nonempty runs;
- manifest and schema are supported;
- commands apply at the recorded ticks;
- periodic digests match;
- final digest matches;
- any divergence reports the first tick, stage, agent UID, and field when possible.

Empty-versus-empty is reported as `NoEvidence`, never “matched.”

### 7.5 bounded storage worker

The current unbounded storage channel carrying cloned full batches can exhaust memory. The new worker uses:

- a bounded queue sized from measured write throughput;
- explicit lag and queue-depth metrics;
- backpressure or documented sampling only for lossy analytics, never lifecycle/command events;
- a high-water policy that pauses the host before the lossless journal queue overflows;
- a fail-closed timeout for strict experiment runs when storage cannot recover;
- an optional future on-disk WAL bead if longer disconnected operation is required;
- structured error propagation to host status and frontends;
- graceful flush with deadline on shutdown;
- `spawn_blocking` or a dedicated thread for the synchronous FrankenSQLite API;
- batch schemas that avoid cloning full render snapshots.

`Admitted` is not crash durability. `Applied` means the scientific transition occurred. `Durable` is emitted only after the command/event journal is acknowledged by storage. In strict experiment mode, new scientific ticks do not proceed past the lossless high-water mark. The first implementation creates one FrankenSQLite connection inside the storage worker thread, keeps it exclusively there, and commits each accepted batch transactionally. This is mandatory because the current `Connection` is deliberately `!Send + !Sync`; the existing pre-opened `Arc<Mutex<Storage>>` topology cannot be retained. `StoragePipeline` keeps only bounded channels, receipts, published analytics state, and the join handle. Analytics use a separate same-thread read connection or the published read model; a `Connection` is never moved to or concurrently shared across threads. Multi-connection write concurrency is enabled only after a focused MVCC contention/retry test proves the access pattern.

### 7.6 FrankenSQLite replacement and multi-run schema

DuckDB is not an accepted backend. Remove its manifest edges, native build graph, direct test imports, mode names, file extensions, documentation, and CI assumptions. The replacement is the public `fsqlite` facade from `~/projects/frankensqlite`, pinned to immutable commit `cd9990bb16291d8c7c247b75b47faae8d7701adb` (package `0.1.16`). Its `default-features = false, features = ["native"]` source check passed under this workspace's `nightly-2026-07-09` in 13m44s; the sibling workspace emitted two upstream unused-feature warnings for obsolete `core_intrinsics` gates, which downstream Git dependency resolution must recheck. JSON payloads remain ordinary `TEXT`, so ScriptBots does not explicitly request extension features; the current upstream `native` feature still activates its extension bundle and is measured separately.

FrankenSQLite uses `LicenseRef-MIT-OpenAI-Anthropic-Rider`, not plain MIT. Local integration is authorized by this project decision, but public distribution stays gated on an explicit license/notice decision and honest downstream package metadata. Do not silently describe the combined product as only `MIT OR Apache-2.0`.

The current facade is synchronous and value-oriented: `Connection::open`, `prepare`, `query[_with_params]`, `execute[_with_params]`, and explicit `begin_transaction`/`commit_transaction`/`rollback_transaction`. Migration code must follow this real surface rather than the outdated rusqlite-shaped example in the sibling README. `SqliteValue` conversion and row decoding live at the storage boundary and return contextual errors; no compatibility wrapper preserves DuckDB APIs.

| Current storage behavior | FrankenSQLite disposition |
|---|---|
| file path and `:memory:` open | accept through `Connection::open` |
| `SET threads`, progress-bar setting | remove; engine-specific and unsupported |
| `BIGINT`/`DOUBLE`/`BOOLEAN`/`JSON` declarations | adapt to `INTEGER`/`REAL`/checked integer/validated `TEXT` |
| positional `params!` binding | adapt to numbered `?1` parameters and `SqliteValue` slices |
| driver `Transaction` object | use the facade RAII `compat::TransactionExt` or an explicit rollback guard; never leave a failed transaction active |
| `INSERT OR REPLACE` batch writes | accept only at the pinned revision containing the current corruption fixes; prove all seven tables |
| cursor-style typed row extraction | adapt to `Vec<Row>`/streaming callback plus storage-owned typed decoders or `compat::RowExt` |
| `MAX`, `AVG`, `COUNT`, `GROUP BY`, ordering, parameterized limit | accept only after exact query conformance fixtures |
| `PRAGMA optimize` | remove; current engine silently treats it as an unknown no-op |
| `VACUUM`, WAL checkpoint, integrity check | accept after file-backed proof; do not put expensive maintenance in UI paths |
| cross-thread `Arc<Mutex<Connection>>` | reject; `Connection` is `!Send + !Sync` |
| optional `AsyncConnection` | hold; do not bridge Tokio by enabling an Asupersync runtime solely for storage |
| old database artifacts from the retired backend | reject without conversion; leave files untouched |

The first dependency uses `fsqlite` only in `scriptbots-storage`. The application and renderer crates consume storage ports and analytics snapshots, never the database crate. The current `native` feature also activates extension/platform families that ScriptBots does not need; correctness migration may use it initially, while a separate non-blocking bead qualifies a lean file-backed feature with before/after package and compile measurements.

Existing DuckDB artifacts are pre-release development data and are not migrated in place. Backwards compatibility is explicitly out of scope. Tests create new FrankenSQLite databases, and unsupported legacy files fail clearly without destructive conversion.

Every table is scoped by `run_id`. The minimum normalized schema includes:

- `runs`;
- `run_features`;
- `commands`;
- `checkpoints`;
- `tick_summaries`;
- `metrics`;
- `agents` or sampled agent state;
- `births`;
- `deaths`;
- `interactions`;
- `genomes` and lineage edges;
- `state_digests`;
- `artifacts`.

Schema migrations use FrankenSQLite's `MigrationRunner`/`PRAGMA user_version`, are versioned and transactionally tested, and never destructively rewrite a user database without explicit permission. One database may hold multiple matched-seed variants without primary-key collisions. The conformance gate covers file reopen, explicit close/checkpoint, rollback on a failed batch, prepared parameter binding, aggregate queries, replay ordering, two independent connections, shutdown flush, `PRAGMA integrity_check`, and a concurrent-writer retry scenario that preserves FrankenSQLite's default MVCC behavior. Retry the whole transaction only for `FrankenError::is_transient()`.

### 7.7 experiment runner

The headless experiment runner accepts a declarative experiment:

```toml
schema_version = 1
scenario = "predator_prey"
seeds = [101, 102, 103, 104]
ticks = 200000

[[variants]]
name = "mlp"
brain_families = ["mlp"]

[[variants]]
name = "dwraon"
brain_families = ["dwraon"]
```

It supports matched seeds, bounded parallelism, checkpoint/branch experiments, resumable runs, and machine-readable progress. It never launches a renderer unless asked.

### 7.8 offline analysis integrations

Sibling scientific projects are optional analysis adapters, not hot-loop dependencies:

- `frankenscipy`: bootstrap confidence intervals, permutation tests, nonparametric comparisons, ACF/FFT, phenotype clustering; gated because its stats dependency graph is large and some periodogram/Welch behavior is known incomplete;
- `franken_networkx`: lineage DAGs, interaction/cooperation/attack networks, communities and centrality; build graphs post-run because its concrete nodes/results allocate strings;
- `frankenpandas`: external reporting or Parquet/Arrow/CSV handoff only; FrankenSQLite remains the primary in-process query engine;
- `franken_numpy`: only the focused RNG adapter is a direct candidate.

Each optional adapter has a small conformance fixture and does not change core determinism when absent.

---

## 8. FrankenTUI Frontend

FrankenTUI is already current at local/upstream commit `fccff2a7e51d39a927bced882877a45aef5c8d39`. The repository has unrelated local changes, so no stash or pull is needed or appropriate. Adoption should pin an exact released version or exact commit after licensing/distribution review; it must not use a mutable sibling path in CI.

### 8.1 adoption boundary

The first TUI migration preserves the public `TerminalRenderer` façade while replacing internals with FrankenTUI’s Elm-style model:

- `Model::init` creates frontend-only state from the latest snapshot;
- `update` maps messages/events to frontend state and typed simulation intents;
- `view` renders a snapshot without mutating simulation state;
- `subscriptions` supplies UI ticks, input, and snapshot updates;
- `on_shutdown` and `on_error` guarantee terminal restoration.

The initial executor is Crossterm for cross-platform behavior. FrankenTUI’s optional Asupersync executor is a separate later bead after the frontend protocol is stable.

### 8.2 screen model

The primary screen is **World**:

- high-resolution Braille/block canvas for terrain, food, and agents;
- pan, zoom, follow, and nearest-agent selection;
- top status ribbon: seed, tick, TPS, population, births/deaths, run state, queue/storage/determinism health;
- responsive inspector: vitals, senses, outputs, brain layers, lineage, recent events;
- bottom timeline and compact population/resource charts;
- command palette for every action, scenario, and editable knob.

Additional screens:

- **Lineages**: founders, branches, reproductive success, genome diffs;
- **Brain Arena**: family cohorts and selected-brain activation/output views;
- **Experiments**: variants, seed cohorts, progress, confidence intervals;
- **Replay**: checkpoints, scrubber, first divergence;
- **Environment**: terrain, fertility, temperature, hydrology layers;
- **Diagnostics**: queues, storage lag, snapshot timings, feature/build provenance.

### 8.3 responsive layouts

Use FrankenTUI `ResponsiveLayout` with declared breakpoints:

- `40x12`: emergency compact status and tiny world;
- `60x20`: stacked world and tabbed inspector;
- `80x24`: standard narrow terminal;
- `120x40`: two-pane world/inspector plus charts;
- `200x60`: three-pane observatory.

No panel receives fixed heights that silently exceed the viewport. Every size has a golden and a “no content outside bounds” assertion.

### 8.4 world canvas

Use `ftui-extras` directly for:

- Braille 2x4 density rendering;
- block 2x2 and half-block 1x2 fallbacks;
- grow-only reusable painter buffers;
- capability-based glyph selection.

Emoji are sparse semantic accents for selection and legends, not one-character-per-cell terrain. The emoji/fallback API handles display width and continuation cells.

### 8.5 command palette

FrankenTUI’s `CommandPalette` becomes the discoverability surface. Entries include:

- pause/resume/step/speed;
- scenario and experiment selection;
- layers and overlays;
- follow/select/search agent;
- config knobs with validation and receipt status;
- checkpoint/export/screenshot;
- diagnostics and help.

The palette shows the command’s applied/rejected receipt rather than optimistically pretending the world changed.

### 8.6 deterministic frontend testing

Use `ProgramSimulator` to inject events/messages and capture arbitrary viewport buffers. Required artifacts:

- full-cell buffer hashes, not text-only snapshots;
- continuation-aware buffer text for debugging;
- golden frames for initial, ten ticks, paused, one-step, palette, select, resize, and error states;
- capability profiles: truecolor, 256 color, 16 color, monochrome, ASCII-only, wide-character;
- deterministic 1,000-event input storm;
- PTY test of the actual binary, including normal exit, signal exit, panic/error path, resize, and terminal restoration;
- shadow-run comparison between legacy Ratatui and FrankenTUI semantic actions during migration.

No test may declare the TUI rendered correctly without inspecting buffer cells or terminal output.

Normal/error cleanup uses RAII. Release builds use `panic = "abort"`, so unwinding is not a guarantee: install and test a minimal emergency panic hook and supported-signal restoration path, avoid allocations/locks where practical, and document best-effort limits. `SIGKILL`, power loss, and equivalent uncatchable termination cannot restore a terminal and are not falsely claimed.

### 8.7 control CLI convergence

The separate `control_cli watch` loop currently duplicates Ratatui/Crossterm behavior and defaults to a different server port. After the main frontend is stable, reuse the same model/widgets or replace watch with the shared TUI client. All control surfaces default to 8088 or one centrally defined value.

---

## 9. Native GUI Frontend

### 9.1 immediate startup truth fixes

Before visual redesign:

- explicit `--mode gui` must not be overridden by headless auto-detection;
- macOS uses native-window availability, not X11/Wayland variables;
- Auto selects only a renderer compiled into the binary;
- a launch failure returns a real error and can invoke a documented fallback;
- default feature/mode behavior in README and `--help` matches the binary;
- launch scripts set only variables that the renderer actually reads.

### 9.2 Bevy presentation-only adapter

Remove Bevy’s simulation worker. Bevy consumes `Arc<RenderSnapshot>` and emits typed intents. Its app lifecycle cannot control scientific time.

Replace per-agent child entities/materials with:

- one or a few instanced meshes or storage-buffer-driven draws;
- palette/health/selection/brain data packed per instance;
- batched eyes/spikes/effects where visually necessary;
- revisioned terrain chunks updated only when their content hash changes;
- cached materials shared by class rather than agent;
- level-of-detail and culling based on measured need.

### 9.3 Evolution Lab visual design

The GUI should look like a living research instrument:

- stylized terrain relief and water/moisture layers;
- legible agents at all useful zoom levels;
- subtle trails and interaction flashes;
- selection/follow camera with clear focus;
- dockable inspector for vitals, lineage, senses, outputs, and brain;
- population/resource charts and event timeline;
- scenario/experiment browser;
- replay/checkpoint controls;
- deterministic/provenance badge and warnings;
- accessible palettes and reduced-effects mode.

Visual richness is layered so low-power mode can disable expensive effects without changing simulation state.

### 9.4 actual Bevy capture tests

An accepted Bevy golden must run the actual app/render world, camera, ECS synchronization, material/shader path, and capture/readback. A CPU circle raster is useful only as a semantic reference and is named accordingly.

Tracked scenes:

- default meadow;
- dense 10k-agent stress;
- combat close-up;
- selected agent and inspector;
- hydrology/terrain;
- each camera preset;
- empty/extinct world;
- error/disconnected state.

Use deterministic assets and tolerance-aware image metrics with region-specific semantic checks. Save actual/diff artifacts on failure.

Every image artifact declares its provenance:

- `SemanticProjection`: renderer-neutral CPU/API visualization;
- `FrontendFramebuffer`: pixels captured from an interactive shipped frontend;
- `OffscreenLiveRenderer`: the real frontend render graph executed offscreen.

Metadata includes frontend/backend, adapter, scene, seed, tick, viewport, color space, code/lock/toolchain, and world digest. A semantic PNG is never described as a Bevy or GPUI framebuffer.

### 9.5 GPUI direct-texture spike

Time-box the GPUI spike to answer one question: can the pinned GPUI revision present a persistent image/texture or custom GPU surface without a GPU-to-CPU round trip and per-pixel quad reconstruction?

The spike must report:

- exact GPUI revision and supported API;
- copies/readbacks per frame;
- quads/draw calls;
- 1080p frame p50/p95/p99;
- resize behavior with two windows;
- memory and device-loss behavior;
- screenshot path;
- maintenance cost relative to Bevy.

If the answer is no, stop investing in GPUI as the world renderer. It may remain a lightweight dashboard during migration. Removal requires explicit user permission because file deletion is forbidden.

### 9.6 GUI acceptance matrix

- native macOS explicit GUI launch;
- Windows D3D12 launch;
- Linux Vulkan launch with clear headless failure/fallback;
- GPU adapter unavailable;
- window creation failure;
- device loss and resize;
- one and two frontend windows with zero effect on tick count;
- paused and single-step control from GUI, REST, and MCP;
- stalled renderer with bounded snapshot memory;
- screenshot from actual live frame;
- 1k and 10k agent frame/perf budgets;
- 30-minute soak without unbounded resources.

---

## 10. Dependency Modernization Program

The named `library-updater` workflow applies across the workspace. “Across the board” does not mean changing every version in one unreviewable commit. It means inventorying every direct dependency, choosing an explicit disposition, updating one dependency or tightly coupled family at a time, and proving the workspace after each change.

### 10.1 reproducibility before upgrades

Before the first version change:

1. Record the current resolved graph and duplicate packages.
2. Track the application workspace’s `Cargo.lock`.
3. Pin the Rust nightly to a dated toolchain that passes the baseline.
4. Replace mutable Git branches with exact revisions.
5. Capture workspace, feature, target, and smoke-test baselines.
6. Create `UPGRADE_LOG.md` with one entry per dependency change.

Without these steps, a “before” build can change while an upgrade is being evaluated.

### 10.2 dependency dispositions

Every direct dependency receives one of:

- **upgrade now**: supported stable release, small migration, high value;
- **upgrade after boundary**: major migration is safer after a module/runtime seam exists;
- **pin current revision**: mutable source must become reproducible before redesign;
- **replace**: dependency is the wrong architectural fit;
- **remove after proof**: unused/placeholder dependency, pending explicit file/code decisions;
- **optional analysis only**: excluded from normal runtime graph;
- **hold with reason**: upstream defect, platform blocker, or incompatible MSRV.

### 10.3 current risk inventory

| Domain | Current concern | Planned disposition |
|---|---|---|
| Toolchain | floating `nightly`; README says 1.85 while active code uses current nightly | pin date, document actual policy |
| Lockfile | ignored application `Cargo.lock` | track and verify |
| GPUI | Zed Git `branch = "main"` | exact rev immediately; later keep/retire decision |
| FrankenTUI | local 0.5 series, pre-1.0 and licensing rider | exact release/rev after license and published-crate audit |
| Asupersync | local 0.3.8, rapidly evolving | exact release/rev behind runtime boundary |
| Ratatui/Crossterm | alpha declaration resolves stable Ratatui; duplicate Crossterm versions | converge during FrankenTUI migration |
| Unicode width | direct 0.1 plus newer transitive API | converge and test continuation cells |
| Bevy | large default feature graph | update only with live renderer tests; minimize features |
| WGPU/Naga | Bevy and custom renderer use different major lines | isolate renderer graphs; upgrade as a coupled family |
| GPUI/WGPU bridge | depends on mutable GPUI and direct WGPU path | feasibility spike before more churn |
| FrankenSQLite | exact-revision pre-1.0 API; synchronous facade; MVCC retry semantics | replace DuckDB immediately, pin a minimal feature set, then harden behind the bounded storage seam |
| Axum/Tokio | currently functional transport boundary | patch updates first; runtime replacement later |
| MCP SDK | pre-1.0 protocol surface | pin, conformance-test, upgrade separately |
| Candle/Tract/Tch | old optional versions and placeholder implementation | disable from defaults; upgrade only with real backend tests |
| NeuroFlow | old/lightweight but introspection is expensive | hold until genome/evaluator contract |
| rand 0.8 + 0.9 | dual versions required by WFC boundary | isolate legacy adapter; remove when upstream allows |
| wfc/direction | legacy API pulls rand 0.8 | hold or replace after map-generation conformance |
| Image | duplicated through render stacks | converge where semver permits |
| TOML/RON | old direct versions plus newer transitive versions | upgrade after config golden tests |
| Windows/Linux target deps | unconditional or overly broad platform graph | make target-specific and feature-minimal |

The exact researched target versions and primary-source links are recorded in `UPGRADE_LOG.md` as changes begin. A version is not selected solely because `cargo outdated` reports it.

#### 10.3.1 complete direct-dependency ledger

Before updates become ready, generate a reviewable ledger covering all twelve manifests and every direct registry/Git/path declaration. Each row includes:

- package and declaring manifest;
- current constraint and resolved version;
- latest stable candidate;
- disposition and reason;
- upstream release/changelog/security links;
- enabled/default/target-specific features;
- coupled-family membership;
- supported platforms and required test lane;
- owner bead and status.

Bevy/`bevy_mesh`/`bevy_post_process` and WGPU/Naga are explicit coupled families. Asupersync, FrankenTUI, and `fnp-random` are new-dependency decision beads, not ordinary upgrade rows.

### 10.4 one-dependency protocol

For each upgrade:

1. Read upstream release notes, migration guide, repository, security notes, and feature changes.
2. Record the manifest constraint, old resolved version, target version, exact allowed coupled-family members, reason, and source links.
3. Update one dependency or one inseparable family such as WGPU/Naga.
4. Use a targeted lock operation such as `cargo update -p name@old --precise target` when appropriate; stop if unrelated lock entries move.
5. Review the exact manifest and lock delta before continuing.
6. Run format, targeted check **and tests**, and the relevant live smoke after every dependency.
7. Run workspace check before moving to the next dependency.
8. Record code changes, feature changes, binary/build-time impact, vulnerabilities, and evidence.
9. Stop and reverse only the dependency’s own reviewed patch if the upgrade fails; never use checkout, stash, reset, or another operation that could overwrite shared changes.

Research may run in parallel, but dependency mutations are serialized through one exclusive lane because every change shares `Cargo.lock`. Reserve the relevant manifests and lockfile for the active bead; `bv --robot-plan` does not authorize concurrent lock mutations.

### 10.5 circuit breakers

Pause an upgrade for explicit user review when:

- it requires semantic edits across more than ten files;
- it forces a renderer/runtime architecture decision scheduled for a later bead;
- it removes a supported backend;
- it changes persisted schemas or scientific results;
- upstream licensing is unclear;
- the new version requires a toolchain incompatible with the pinned workspace;
- compile time, binary size, or memory grows materially without measured benefit.

Operational stop rules:

- at most three repair attempts for one dependency, then reverse its reviewed patch and log the hold;
- at most three network retries, then skip and log;
- stop the update campaign after five consecutive dependency failures;
- stop when total newly observed test failures exceed ten;
- stop immediately if an audit reports a newly introduced vulnerability;
- finish the campaign with a security/advisory audit;
- the stricter user-review gate for changes spanning more than ten files always applies.

### 10.6 upgrade order

The safe order is:

1. lockfile, dated toolchain, mutable Git revision pins;
2. leaf developer/test dependencies;
3. serde/error/logging/data-format patch updates;
4. terminal stack convergence through FrankenTUI;
5. HTTP/control/MCP patch updates;
6. core numerical and indexing dependencies;
7. exact-revision FrankenSQLite replacement, then storage lifecycle hardening;
8. Bevy family after live render harness;
9. custom WGPU/Naga family after renderer ownership is fixed;
10. GPUI only if the direct-texture spike justifies it;
11. real ML backend families independently;
12. optional pinned sibling adapters.

### 10.7 feature graph cleanup

The default binary should not compile placeholder ML stacks or multiple rich renderers accidentally. Define supported products explicitly:

- `scriptbots` default: core + storage + FrankenTUI + server/control;
- `gui-bevy`: primary native GUI;
- `gui-gpui`: temporary/optional dashboard if retained;
- `audio`: optional;
- `brain-neuroflow`: optional;
- `brain-candle`, `brain-tract`, `brain-tch`: independently optional and real;
- `analysis-*`: never in normal application defaults;
- `web`: browser-compatible subset.

CI tests declared products, not the combinatorial powerset of nonsensical feature combinations.

### 10.8 dependency completion criteria

- every direct dependency has a disposition and owner bead;
- lockfile and toolchain are tracked/pinned;
- no mutable Git branch dependencies remain;
- duplicate major versions have a documented reason;
- default features contain no placeholders;
- each update has an `UPGRADE_LOG.md` entry and primary-source research;
- full workspace gates pass after the final update batch;
- README/toolchain/install instructions match reality.

### 10.9 registry-verified candidate target floors

These candidates were checked against live crates.io records on 2026-07-11. Registry metadata establishes a candidate version, not its migration risk. Release notes, changelog, repository, features, and security research are still required before the bead becomes actionable. They are manifest lower-bound candidates plus a committed lock, not universal exact `=version` requirements.

| Group | Researched stable targets |
|---|---|
| Core/data | `anyhow 1.0.103`, `rayon 1.12.0`, `serde 1.0.228`, `serde_json 1.0.150`, `thiserror 2.0.18`, `tracing 0.1.44`, `tracing-subscriber 0.3.23`, `slotmap 1.1.1`, `postcard 1.1.3`, `num_cpus 1.17.0`, `csv 1.4.0` |
| App/server | `axum 0.8.9`, `async-trait 0.1.89`, `clap 4.6.1`, `serde_path_to_error 0.1.20`, `smallvec 1.15.2`, `futures-util 0.3.32`, `tokio-stream 0.1.18`, `supports-color 3.0.2`, `tokio 1.52.3`, `utoipa 5.5.0`, `owo-colors 4.3.0` |
| Platform/dev | `libc 0.2.186`, `windows-sys 0.61.2` (platform-gated), `wayland-client 0.31.14`, `mimalloc 0.1.52`, `tempfile 3.27.0`, `serial_test 3.5.0` |
| Storage/render | target `fsqlite 0.1.16 @ cd9990bb` (minimal native storage gate), `crossbeam-channel 0.5.16`, `image 0.25.10`, `bytemuck 1.25.1`, `winit 0.30.13` |
| Web | `js-sys 0.3.103`, `wasm-bindgen 0.2.126`, `wasm-bindgen-test 0.3.76`, `serde-wasm-bindgen 0.6.5` (browser/WASM gate) |
| Already latest | `mcp-protocol-sdk 0.5.1`, `utoipa-swagger-ui 9.0.2`, `neuroflow 0.2.0` |

Each actual change still follows the one-dependency protocol. The registry source for a target is `https://crates.io/crates/<crate>/<version>`.

### 10.10 immediate dependency truth fixes

Before general bumps:

1. Replace GPUI’s mutable branch with exact revision `5f8a7413a31769e0882357f90dc424b3962ac72d`, the captured lockfile revision. Zed `main` has already moved beyond it (observed at `bc99075...` during this audit), which is the reproducibility defect. Pinning the captured revision makes the current red baseline reproducible; a separate GPUI API/renderer decision bead must make it green.
2. Remove the unused `num-bigint-dig` Git patch; Cargo already reports it as `[[patch.unused]]`.
3. Declare Ratatui’s actually resolved stable `0.30.2` only if it remains during the FrankenTUI transition.
4. Converge direct Crossterm to `0.29.0` only if the legacy adapter remains.
5. Remove wasm-bindgen’s deprecated `serde-serialize` feature because the code already uses `serde-wasm-bindgen`.
6. Correct `rust-version`: locked Bevy 0.17.3, WGPU 27.0.1, and Ratatui 0.30.2 require Rust 1.88, so the current 1.85 claim is false even before upgrades.

Primary references:

- [exact captured Zed/GPUI commit](https://github.com/zed-industries/zed/commit/5f8a7413a31769e0882357f90dc424b3962ac72d)
- [published GPUI 0.2.2](https://crates.io/crates/gpui/0.2.2)
- [num-bigint-dig 0.9.1](https://crates.io/crates/num-bigint-dig/0.9.1)
- [Ratatui 0.30.2](https://crates.io/crates/ratatui/0.30.2)
- [Crossterm 0.29.0](https://crates.io/crates/crossterm/0.29.0)
- [wasm-bindgen deprecated features](https://docs.rs/wasm-bindgen/latest/wasm_bindgen/#deprecated-features)

### 10.11 proven dead/redundant declarations

Static source search found no current use for the following direct declarations. Remove them one at a time with the owning-crate checks rather than carrying or upgrading them:

- workspace `derive_more` and `fastrand`;
- app `image` and `unicode-width` in the current implementation;
- Bevy `owo-colors`;
- brain `rayon` and `thiserror`;
- brain-ml `serde` and `thiserror`;
- core `ordered-float`, whose only source occurrence is an explicitly allowed
  unused import;
- render `serde`;
- storage `tracing`;
- world-gfx `futures-intrusive` and `glam`.

Workspace Candle/Tract/Tch declarations are redundant because the brain-ml crate redeclares them. More importantly, none of those backend crates is used by the placeholder’s evaluation code. Disable/remove their current default surface and reintroduce each only with a real implementation bead.

Optional `rstar` and `kiddo` features advertise implementations that do not exist. Implement and test them before upgrading, or remove the fictional features.

The same slice should relocate rather than delete declarations whose scope is
wrong: app `rand` and render's `scriptbots-brain` edge are integration-test
only; app `libc` and `windows-sys` belong under Unix and Windows target
tables; brain-neuro repeats Serde's already-inherited `derive` feature. These
changes reduce production closure without changing executable behavior.

### 10.12 intentional holds

- Keep `wfc 0.10.7`, `direction 0.18.1`, and the `rand 0.8` alias together. WFC depends on the older direction type and rand line; upgrading direct direction alone creates incompatible types.
- Keep `neuroflow 0.2.0` as an explicit maintenance-risk hold because no newer release exists.
- Remove unused `futures-intrusive 0.5.0` instead of “updating” it.
- Keep render's direct Naga declaration until a focused proof replaces its
  activation-only `termcolor` effect on the shared WGPU 27 graph. It has no
  API reference, but deleting feature-unification dependencies by grep is not
  safe.
- Keep the direct getrandom activation edges until their full target graphs are
  replaced: core's 0.3 declaration feature-unifies `wasm_js` for the
  rand_core 0.9 line, while web's wasm32-only 0.2 declaration activates `js`
  for the rand_core 0.6/WFC line. They have no API references but are not dead.
  Any migration must prove both wasm32 graphs instead of deleting them by grep.

References:

- [WFC 0.10.7](https://crates.io/crates/wfc/0.10.7)
- [Direction 0.19.1](https://crates.io/crates/direction/0.19.1)
- [NeuroFlow 0.2.0](https://crates.io/crates/neuroflow/0.2.0)
- [getrandom WebAssembly guidance](https://docs.rs/getrandom/latest/getrandom/#webassembly-support)

### 10.13 major migration gates

The following are architectural migrations, not routine bumps:

| Dependency | Target | Required gate |
|---|---|---|
| `rand` | `0.10.2` | choose named reproducible algorithm, digest tests, migrate public brain RNG contract |
| `getrandom` | `0.4.3` | migrate after rand, scope WASM backend to final application |
| `ordered-float` | `5.3.0` | spatial/property/replay tests for changed hash/operator behavior |
| `wide` | `1.5.0` | MSRV >=1.89, scalar/SIMD numerical and benchmark gates |
| `criterion` | `0.8.2` | isolated benchmark harness migration |
| `reqwest` | `0.13.4` | explicit rustls feature/default decision and control CLI E2E |
| `crossfire` | `3.1.19` | replace broken command driver first; prefer removal if Asupersync channel wins |
| `ron` | `0.12.2` | scenario/config corpus golden round trips |
| `toml` | `1.1.2+spec-1.1.0` | config corpus and formatted-output review |
| `pollster` | `1.0.1` | update after renderer/WGPU family |
| `kira` | `0.12.1` | cpal-only feature and audio smoke |
| `rstar` | `0.13.0` | real implementation/conformance first |
| `kiddo` | `5.3.2` | real implementation/conformance first |
| Candle | `0.11.0` | real model evaluator/genome semantics first |
| Tract | `0.23.4` | MSRV >=1.91, security/facade/ndarray migration, real ONNX tests |
| Tch | `0.24.0` | exact libtorch/PyTorch 2.11 provisioning and native CI |
| Bevy | `0.18.1`, then `0.19.0` | one release at a time, real render goldens, MSRV 1.95 |
| WGPU | `28`, `29.0.3`, optionally `30.0.0` | one major at a time after primary renderer decision; GPU backend matrix |

Primary migration references:

- [rand 0.10 update guide](https://rust-random.github.io/book/update-0.10.html)
- [ordered-float 5.0 release](https://github.com/reem/rust-ordered-float/releases/tag/v5.0.0)
- [wide 1.5 changelog](https://github.com/Lokathor/wide/blob/v1.5.0/changelog.md)
- [Reqwest TLS documentation](https://docs.rs/reqwest/latest/reqwest/tls/index.html)
- [Crossfire MPMC API](https://docs.rs/crossfire/latest/crossfire/mpmc/index.html)
- [RON 0.12 changelog](https://github.com/ron-rs/ron/blob/v0.12.2/CHANGELOG.md)
- [TOML 1.1 changelog](https://github.com/toml-rs/toml/blob/toml-v1.1.2/crates/toml/CHANGELOG.md)
- [Tract 0.23 changelog](https://github.com/sonos/tract/blob/v0.23.4/CHANGELOG.md)
- [Tch 0.24 requirements](https://docs.rs/crate/tch/0.24.0/source/README.md)
- [Bevy 0.17 to 0.18 migration](https://bevy.org/learn/migration-guides/0-17-to-0-18/)
- [Bevy 0.18 to 0.19 migration](https://bevy.org/learn/migration-guides/0-18-to-0-19/)
- [WGPU 28 changelog](https://github.com/gfx-rs/wgpu/blob/v28.0.0/CHANGELOG.md)
- [WGPU 29 changelog](https://github.com/gfx-rs/wgpu/blob/v29.0.0/CHANGELOG.md)
- [WGPU 30 changelog](https://github.com/gfx-rs/wgpu/blob/v30.0.0/CHANGELOG.md)

Renderer upgrades, deterministic RNG/public trait migration, real ML backends, and broad toolchain/CI changes cross the library-updater circuit breaker and require their own approved implementation beads.

---

## 11. Isomorphic De-Monolithization Program

The named `de-monolithize-your-codebase-isomorphically` skill is used in Standard mode. Its purpose is to reduce change risk, not to shuffle code for aesthetics.

### 11.1 execution gate

The dynamic skill run requires explicit confirmation of:

- target repository: `/Users/jemanuel/projects/rust_scriptbots`;
- mode: Standard;
- separate workspace: `/Users/jemanuel/projects/rust_scriptbots__demonolith_workspace`;
- inventory-only first pass;
- current RCH/local verification routing;
- confirmed extraction seams only;
- separate permission for any tool installation.

Until confirmed, this plan may use the skill’s taxonomy and gates, but it does not create the separate workspace or execute extraction scripts.

If a `phase0_run.json` already exists in that workspace, stop and ask whether to resume or start fresh. Never overwrite prior evidence.

### 11.2 characterization package

**Status:** [Currently In Progress — Phase 1 recon and enriched census, run `2026-07-12-rust_scriptbots-1`, Codex, 2026-07-12]

Before a mechanical split, capture:

- Git commit, branch, dirty paths, toolchain, target, and features;
- public API inventory for each affected crate;
- full current test/check/fmt/clippy outcome;
- known red tests separated from new regressions;
- deterministic world traces and digests;
- CLI help and exit-code snapshots;
- TUI headless output and buffer snapshot;
- GUI launch/capture status;
- build time, peak memory, artifact size;
- representative simulation and render benchmarks;
- cfg/feature matrix;
- monolith declaration/impl/test census;
- import/dependency graph and likely seams.

### 11.3 Standard-mode review rounds

Standard mode does not authorize a split from a line-count list. It requires at least ten complete discovery/experiment loops over phases 2-6, at least two viable candidate seams, and then two consecutive quiet rounds with zero unresolved findings before a candidate can receive `SEAM_CONFIRMED`.

Each loop produces evidence rather than a generic reread and revisits:

1. public API and re-export surface;
2. private data ownership and invariants;
3. call graph and state mutation;
4. cfg/feature/platform boundaries;
5. tests/fixtures and hidden coupling;
6. performance/resource coupling;
7. error/lifecycle coupling;
8. persistence/serialization coupling;
9. documentation/examples/CI coupling;
10. adversarial isomorphism review and next-experiment selection.

Every monolith receives at least two scored decompositions; the winner and runner-up are recorded. Every candidate gets an explicit verdict such as `SEAM_REJECTED`, `SEAM_NEEDS_EVIDENCE`, `DEFERRED` with rationale/review, or `SEAM_CONFIRMED`. Only confirmed seams enter an extraction bead. The inventory below is a set of hypotheses to test, not a pre-approved order.

### 11.4 core seam candidates

Candidate mechanical seams that may preserve `scriptbots_core::*` re-exports and behavior include:

1. pure identifiers, math constants, and value types;
2. `agent` data/storage/runtime types;
3. `brain_contract` registry/binding interfaces;
4. `config` and validation;
5. `environment::food`;
6. `environment::terrain`;
7. `environment::mapgen` and WFC;
8. `environment::hydrology`;
9. `replay` and persistence DTO contracts;
10. `analytics` types;
11. `world` state and lifecycle;
12. stage impls: sense, brain, actuation, ecology, combat, reproduction, population, persistence;
13. inline tests into corresponding modules according to the experiment-proven privacy boundary.

An experiment must prove whether keeping `WorldState` in `world/mod.rs` with descendant stage modules actually preserves privacy, API, and compile-resource behavior. Scratch buffers are not redesigned during a mechanical move. Grouping them into stage-specific scratch structs is a later semantic/performance bead.

### 11.5 app seam candidates

Candidate seams to test for binary-behavior preservation:

1. CLI declarations and parse validation;
2. configuration layering/env overrides;
3. run manifest/bootstrap/scenario loading;
4. renderer selection and launch policy;
5. replay/profile/experiment command dispatch;
6. control runtime/server startup;
7. binary `main` as thin composition root.

### 11.6 TUI seam candidates

Before FrankenTUI replacement, characterize these candidate seams:

1. snapshot/projection DTO;
2. action/key mapping;
3. runtime adapter;
4. layout/screen selection;
5. world map widget;
6. panels/charts;
7. theme/glyph capabilities;
8. headless report/harness;
9. inline tests.

Only after a seam is confirmed may the presentation module move behind the same snapshot/control ports.

The FrankenTUI replacement/retention decision precedes broad extraction of legacy-only presentation code. Characterize all code, but do not spend an extraction campaign on a surface scheduled for retirement.

### 11.7 Bevy seam candidates

1. renderer-neutral snapshot adapter;
2. bridge/inbox and lifecycle;
3. scene setup/resources;
4. terrain mesh/material pipeline;
5. agents/instancing;
6. camera and picking;
7. HUD and input actions;
8. capture/test harness;
9. app entry façade.

### 11.8 GPUI/render seam candidates

1. snapshot types/projection adapter;
2. CPU semantic raster reference;
3. canvas terrain;
4. canvas agents/effects;
5. HUD/panels;
6. input/control actions;
7. compositor/direct-texture experiment;
8. capture;
9. application/window shell;
10. tests.

The GPUI direct-texture/retention decision precedes broad GPUI extraction. Characterization is still required, but rejected renderer code is not decomposed for aesthetic completeness.

### 11.9 world-gfx seam candidates

1. public render data contract;
2. adapter/device selection;
3. renderer resources;
4. terrain pipeline;
5. agent pipeline;
6. post-processing;
7. readback/capture;
8. shader assets;
9. tests and GPU harness.

### 11.10 per-extraction gates

Each extraction runs the skill’s named `isomorphism-gate.sh` and must prove:

- equal test pass and skip counts plus unchanged goldens;
- an empty or additions-only public API diff; any approved breaking change belongs to a separate semantic bead;
- identical feature/cfg build matrix;
- identical deterministic traces/digests;
- identical CLI help and exit behavior;
- unchanged relevant golden artifacts;
- no new module/dependency cycles;
- no performance regression beyond noise budget;
- compile time and peak compiler RSS remain within the recorded gate;
- downstream compile probes and API inventory match;
- clean and incremental compile measurements on the same quiet machine;
- binary size/codegen checks where relevant;
- no new warnings;
- no widened dependency direction;
- no file deletion;
- one focused diff with no semantic cleanup mixed in.

### 11.11 skill loop and why splitting follows characterization

The monoliths contain duplicated behavior and hidden feature paths. Splitting before establishing oracles can preserve bugs invisibly or introduce new ones that current tests cannot see. Conversely, redesigning inside a 12,724-line renderer makes review and rollback difficult. Within an active de-monolith run, the correct sequence is:

1. characterize;
2. propose at least two candidates;
3. run bounded extraction/import/API/compile/perf experiments without landing them;
4. record candidate verdicts;
5. repeat the full phase 2-6 loop at least ten times;
6. achieve two quiet rounds and zero unresolved findings;
7. mark one seam `SEAM_CONFIRMED`;
8. isolate that existing seam mechanically;
9. prove API, behavior, compile-RSS, and performance isomorphism;
10. only in a separate later bead, add a red semantic test and fix behavior.

A correctness fix may land before a de-monolith run starts, but it becomes a new baseline and must be fully re-characterized. Once an extraction experiment is active on a surface, semantic edits do not interleave with it. Before landing from the sibling workspace, refresh against the exact current `main` commit non-destructively and re-run every gate; stale extraction work never overwrites newer semantic fixes.

Skill inventory/census/verification scripts may produce evidence outside project source. They may not rewrite source: AGENTS requires every extraction edit to be performed manually. Tests may move with a confirmed seam when Rust privacy demands it; do not widen visibility merely to force all test movement to the end.

---

## 12. Phased Execution Roadmap

Priorities are `P0` correctness/blocking, `P1` usable product, `P2` scientific expansion, and `P3` optional reach.

### Phase 0 — Establish Truth and Reproducibility (`P0`)

#### 0.1 clean-room baseline

- capture Git/toolchain/dependency/feature state;
- finish host-target workspace check;
- run targeted and workspace tests;
- run clippy and format;
- execute current TUI headless tests and actual binary;
- attempt explicit macOS GUI launch through a bounded probe;
- run render tests to demonstrate missing goldens/invalid CI contract;
- classify every failure as code, test, environment, or infrastructure.

**Exit:** a checked-in baseline section identifies exact green/red commands and expected failures.

#### 0.2 reproducible dependency graph

- track `Cargo.lock`;
- pin dated nightly;
- pin GPUI exact revision;
- add `UPGRADE_LOG.md`;
- capture duplicates and default feature graph;
- repair invalid CI feature names and action refs;
- make golden paths trackable without allowing runtime frame dumps everywhere.

The existing `.gitignore` contains unrelated user changes. This bead must reserve/coordinate that file, preserve those exact hunks, and apply only the minimal lock/golden exceptions. Reproducibility pins are a hard dependency of semantic implementation beads.

**Exit:** two clean resolutions at the same commit produce the same lock and sources.

#### 0.3 honest test assets

- distinguish semantic CPU references from live renderer captures;
- add tracked metadata-bearing golden directories;
- make missing goldens a clear failure with generation instructions;
- prohibit CI auto-bless;
- inspect TUI buffers;
- require nonempty replay evidence.

**Exit:** every advertised test exercises its named path or is renamed to describe its true scope.

#### 0.4 mode/startup contract

- explicit mode overrides auto-detection;
- fix macOS detection;
- select only compiled renderers;
- return launch failures;
- centralize port/defaults;
- remove hidden bootstrap from startup contract or expose it explicitly.

**Exit:** startup matrix passes for compiled/uncompiled GUI, terminal, headless, server, and launch failure.

#### 0.5 characterization manifest and digest v0 [Implemented: `bd-2z0.1.6`]

- define a minimal `RunManifestV0` for seed, normalized config, code/toolchain/lock/features, and scenario identity;
- define a characterization-only digest over the stable state available before `AgentUid`/brain-state redesign;
- capture fixed pre-fix traces for representative seeds;
- label v0 as a temporary oracle format, not the final replay schema;
- make de-monolith and semantic beads cite the exact baseline artifact.

**Exit:** pre-fix behavior has a reproducible before/after oracle; Phase 4 packages later artifacts rather than inventing manifest/digest concepts.

### Phase 1 — Restore Simulation Correctness (`P0`)

#### 1.1 sensing and spatial oracle

- red SIMD chunk vision test;
- scalar heading test;
- FOV/blood policy tests;
- toroidal index duplicate/distance tests;
- scalar/SIMD and serial/parallel comparisons;
- fix implementation only after oracle fails correctly.

**Exit:** sensor digests agree across supported feature paths.

#### 1.2 output/combat contract

- typed output mapping;
- green-versus-boost damage regression;
- replay/render use same accessors;
- finite/range validation.

**Exit:** no raw magic indexes outside the centralized conversion hot path.

#### 1.3 persistence cadence

- current per-tick summary independent of persistence;
- accumulate lifecycle/combat events until flushed;
- test multiple events between intervals;
- prove persistence/analytics cadence does not change scientific state.

**Exit:** cadence changes storage frequency but not world behavior or event totals.

Writer errors, queue bounds, flush, and cancellation belong to the runtime/storage worker phase; they do not mix with this core event-accounting fix.

#### 1.4 scenario/startup construction

- minimal scenario schema on `RunManifestV0`;
- full-world seeding;
- remove hidden 120 ticks;
- preserve current ecology parameters until honest brains/RNG/digests exist;
- add deterministic seed/placement fixtures.

**Exit:** startup is explicit and reproducible; no claim about a tuned ecosystem is made yet.

#### 1.5 stable IDs and minimal random-stream protocol

- add stable `AgentUid` without changing slot-map lookup behavior;
- define the minimal `RandomStream` interface required by core/brain families;
- adapt the current RNG behind that interface first;
- version algorithm identity/state in the manifest;
- add basic state round-trip and stream-identity tests.

**Exit:** downstream brain APIs depend on a real protocol, while the later `fnp-random`/domain-separation decision remains independent.

#### 1.6 brain genome/evaluator-state protocol

- versioned genome envelope;
- versioned evaluator-state envelope;
- family registry;
- explicit child-state reset/inherit/blend policy;
- genome/state codec fixtures;
- object-safe evaluator construction/checkpoint interface;
- batch/arena extension point so `Box<dyn BrainEvaluator>` is not locked in as the only 10k-agent representation.

This is a protocol bead. Separate child beads implement each family and integration so the change does not silently exceed the ten-file review gate.

**Exit:** the protocol and fixture family round-trip genome plus future-affecting evaluator state; full world checkpointing is not claimed yet.

#### 1.7 brain-family adapters and inheritance

- MLP adapter and recurrent-state policy;
- DWRAON adapter/parity and recurrent-state policy;
- Assembly program/genome versus working-state policy;
- offspring genome mutation/crossover and state initialization;
- bound-child/provenance tests;
- remove placeholder from default registration;
- register only honest scenario families;
- on-demand introspection.

**Exit:** a multi-generation fixture proves real brain genomes evolve and every child is bound.

#### 1.8 canonical digest and checkpoint skeleton

- canonical, versioned `WorldDigestV1` using `AgentUid` and stable ordering;
- include genome, evaluator state, RNG state, config, terrain/food, spawn ordinals, and all future-affecting counters;
- per-stage digest hooks enabled only in diagnostic/test mode;
- checkpoint envelope/codec skeleton;
- encode/decode idempotence fixtures;
- upgrade `RunManifestV0` to a versioned canonical manifest.

**Exit:** fixed runs have a canonical first-divergence oracle and checkpoint schema before runtime/replay work.

#### 1.9 domain-separated RNG and `fnp-random` decision

- benchmark/integrate or reject pinned `fnp-random` adapter;
- domain-separated streams;
- checkpoint states;
- permutation/noninteraction tests.

**Exit:** unrelated agent insertion or analytics cadence does not perturb existing agent streams.

#### 1.10 resource ledger and meadow tuning

- explicit resource source/transfer/sink ledger;
- tune meadow only with honest brain inheritance, stable streams, and canonical digests;
- run a seed cohort rather than one attractive seed;
- declare viability/extinction/equilibrium envelopes;
- validate the other curated scenario stories separately.

**Exit:** default run is alive, reproducible, and visibly interesting without undocumented overrides or broken offspring.

### Phase 2 — One Authoritative Runtime (`P0`)

#### 2.1 characterize command semantics

- pause/resume/speed/step truth table;
- choose one-step policy: a step request atomically pauses, advances exactly one tick, and remains paused even if received while running;
- config conflict/validation cases;
- queue overflow behavior;
- combined auto-pause conditions;
- shutdown/flush behavior;
- tests covering more than 32 commands.

**Exit:** current broken behavior is captured where intended to change, and target behavior is explicit.

#### 2.2 extract core side effects

- remove simulation-command queue ownership from `WorldState`;
- make core step return deterministic `StepOutcome`/domain deltas;
- remove synchronous `WorldPersistence::on_tick` I/O from the scientific transition;
- keep current summary/event accumulation scientifically correct;
- prove the same fixed trace before/after the boundary.

**Exit:** core performs one deterministic state transition with no command transport or storage I/O.

#### 2.3 runtime dependency and protocol decision

- confirm whether `scriptbots-runtime` crate is justified;
- define a synchronous `HostCore`/`SimulationEngine` state machine with injected/manual time;
- define host/client/snapshot/event ports and separate revision domains;
- define `CommandId`, admission sequence, scientific revision, config revision, snapshot revision, and event sequence;
- define admitted/applied/durable/rejected status, deduplication, retry, disconnect lookup, and exact tick-boundary meaning;
- run a bounded Asupersync MPSC/cancellation/lifecycle spike before choosing native scheduling types;
- prohibit frontend dependency on mutable world;
- add null frontend.

**Exit:** dependency graph enforces that renderers cannot call `WorldState::step`.

#### 2.4 pure host state machine

- sole world ownership;
- deterministic command drain;
- pause/speed/single-step;
- two-phase command status and status lookup;
- status and health.

**Exit:** host-core tests prove exactly-one-step, deduplication, revision conflicts, and command ordering under manual time without any platform runtime.

#### 2.5 native scheduler and lifecycle

- drive the same `HostCore` with a fixed-deadline native scheduler;
- Asupersync structured cancellation/obligations if the spike passes;
- prioritized but formally ordered graceful shutdown;
- bounded catch-up policy;
- browser/WASM adapter remains able to drive `HostCore` without requestAnimationFrame owning scientific time.

**Exit:** native lifecycle tests are cancel-clean and produce the same host-core traces as manual time.

#### 2.6 canonical snapshot hub and projections

- current summary;
- revisioned static layers;
- compact dynamic agents;
- multi-subscriber latest-value `Arc`/watch semantics with independent cursors;
- selected detail and viewport projection as per-client pure requests or keyed broker state, never one global camera/selection;
- sequenced scientific-event journal with explicit UI lag/catch-up behavior;
- projection perf metrics.

**Exit:** stalled readers remain bounded and snapshot publication does not alter world digest.

#### 2.7 control/server migration

- API reads snapshot/status, never world lock;
- mutations await receipt or return explicit accepted status endpoint;
- monotonic revision replaces audit length;
- truthful ASCII/PNG artifacts;
- REST/MCP conformance.

**Exit:** paused/no-renderer server applies and acknowledges commands.

#### 2.8 storage lifecycle migration

- blocking FrankenSQLite boundary with one worker-owned connection;
- bounded lossless command/lifecycle journal queue;
- pause-before-overflow and fail-closed timeout policy when storage remains unavailable;
- admitted/applied/durable receipt semantics;
- flush obligations, recovery, and errors;
- multi-run schema/migrations;
- provenance and command/digest persistence.

**Exit:** overload/worker-failure tests are bounded, visible, and cancel-clean.

### Phase 3 — Presentation Frontends (`P1`)

#### 3.1 legacy TUI adapter

- make current Ratatui consume snapshots/clients;
- remove direct stepping/world locks;
- fix reports/current tick;
- semantic parity shadow tests.

**Exit:** current UI behaves correctly before visual rewrite.

#### 3.2 FrankenTUI source/license decision

**Status:** [Completed — exact Git revision `fccff2a7e51d39a927bced882877a45aef5c8d39`, minimal feature graph, rider-preserving distribution boundary, remote check/test spike, and lifecycle conformance blocker recorded, Codex, 2026-07-12]

- verify published-crate versus exact-Git source availability;
- complete licensing/distribution review;
- select and pin one immutable source;
- record API/version and rollback boundary.

**Exit:** an approved immutable dependency decision blocks every implementation bead.

#### 3.3 FrankenTUI shell and harness

- Model/update/view/subscriptions;
- ProgramSimulator;
- command palette;
- terminal RAII.

**Exit:** shell state transitions, command receipts, teardown, and minimal goldens pass.

#### 3.4 FrankenTUI responsive world view

- responsive shell/breakpoints;
- canvas layers;
- inspector;
- charts/timeline;
- accessible themes;
- mouse/resize/input storm.

**Exit:** World view goldens and PTY/input tests pass at required sizes/capabilities.

#### 3.5 Evolution Lab science screens

- Lineages screen;
- Brain Arena screen;
- Experiments screen;
- Replay screen;
- Environment screen;
- Diagnostics screen;
- checkpoint/export workflows.

**Exit:** a user can run, inspect, intervene, checkpoint, and export without leaving the TUI.

#### 3.6 current-version GUI live characterization

- minimal real Bevy render-app/framebuffer capture on the current version;
- minimal current GPUI window/paint/direct-texture feasibility evidence;
- distinguish code/API failures from unavailable hardware;
- create platform/backend metadata goldens where the current path can run.

**Exit:** renderer migrations have an honest current-version live harness or an explicit red blocker.

#### 3.7 Bevy presentation adapter

- remove simulation worker;
- snapshot inbox;
- prove renderer repaint/window count cannot advance ticks;
- selection/action intents use host receipts.

**Exit:** current-version Bevy consumes canonical snapshots without owning the simulation.

#### 3.8 Bevy GPU instancing

- GPU instancing;
- shared materials/batches;
- 1k/10k agent profiling;
- LOD/culling only from measured need.

**Exit:** agent presentation meets entity/material/memory/frame budgets.

#### 3.9 Bevy terrain, camera, HUD, and capture

- revisioned terrain;
- camera/picking;
- UI actions/receipts;
- actual capture.

These become separate implementation beads for terrain, camera/input, HUD, and capture.

**Exit:** the full Bevy frontend meets live-path correctness and visual gates.

#### 3.10 GPUI feasibility/decision

- exact-rev direct-texture spike;
- measured comparison;
- keep dashboard, keep renderer, or retire decision;
- no deletion until permission.

**Exit:** one documented primary GUI architecture remains.

#### 3.11 live visual E2E

- tracked actual-render goldens;
- input/camera tests;
- platform launch probes;
- device failure/recovery;
- screenshot/export parity;
- soak tests.

**Exit:** claims about GUI appearance come from the shipped path.

### Phase 4 — Replay and Experiments (`P1`)

#### 4.1 command/event journal and checkpoint implementation

- persist the already-defined canonical manifest/digest schemas;
- sequenced commands and status transitions;
- nonempty domain events;
- evaluator/RNG/world checkpoint state;
- checkpoint resume and first-divergence verification.

**Exit:** tick-zero and checkpoint replay match in the bit-exact lane and empty evidence is rejected.

#### 4.2 deterministic run-bundle assembler

- manifest/source provenance;
- command/event/checkpoint/digest files;
- FrankenSQLite run database;
- artifact index;
- import/verify command.

**Exit:** a bundle verifies on a second clean checkout with the same pinned toolchain.

#### 4.3 experiment runner

- matched seeds/variants;
- bounded concurrency;
- resume/checkpoint/branch;
- progress/status;
- report summary.

**Exit:** MLP versus DWRAON sample experiment produces reproducible comparable runs.

#### 4.4 lineage and phenotype analysis

- lifetime reproductive success;
- founder contribution;
- interaction graphs;
- phenotype feature table;
- optional FrankenNetworkX/FrankenSciPy adapters;
- confidence intervals/effect sizes.

**Exit:** UI and exported report explain one observed evolutionary change.

#### 4.5 intervention studies

- drought/resource shock;
- temperature shift;
- closed-world toggle;
- checkpoint branch;
- paired outcome comparison.

**Exit:** intervention commands replay at the same ticks and produce verified branch bundles.

### Phase 5 — Web and External Agents (`P2`)

#### 5.1 renderer-neutral web projection

- consume the same snapshot schema;
- keep browser persistence separate;
- no duplicated world semantics;
- WASM/native digest comparison.

#### 5.2 agent-friendly API/MCP

- status/health/read snapshots;
- command receipts;
- scenario/experiment tools;
- checkpoint and artifact discovery;
- bounded responses and pagination;
- schema/version discovery.

#### 5.3 browser live renderer

- only after native runtime and snapshot contracts are stable;
- real Canvas/WebGPU tests;
- capability/fallback matrix;
- requestAnimationFrame never owns scientific time.

### Phase 6 — Documentation, Packaging, and Release (`P1`)

- README truth rewrite from verified commands;
- archive/supersede stale completion claims without deleting history;
- install/run/feature matrix;
- architecture and contribution docs;
- scenario/experiment tutorials;
- troubleshooting/doctor output;
- release build and bundle verification;
- all work lands on `main`; after any authorized push to `main`, immediately mirror that exact commit to the legacy branch as required by AGENTS.

---

## 13. Verification Architecture

Verification is a product feature. A simulation laboratory is only as useful as its ability to explain what happened and prove that the same inputs reproduce it.

### 13.1 gate hierarchy

Use the cheapest relevant gate first, then widen:

1. format and static structure;
2. focused unit test;
3. crate test/check;
4. feature/cfg variant;
5. workspace check/test/clippy;
6. semantic golden/metamorphic suite;
7. real frontend E2E;
8. performance/resource benchmark;
9. platform matrix;
10. soak/replay bundle verification.

No low-level green result substitutes for a relevant high-level gate. A CPU raster cannot prove a GPU application; numeric JSON cannot prove a TUI layout.

For substantive code changes, the repository-wide handoff gate follows AGENTS exactly where platform prerequisites permit:

- `cargo check --workspace --all-targets`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo fmt --check`;
- `cargo test --workspace`;
- `cargo test --workspace --all-features`;
- relevant real frontend/scenario/replay E2E.

Use RCH for compute-intensive gates when healthy; otherwise run sequentially in the external scratch target and record infrastructure limits. Run `ubs <changed-files>` before every commit, not just the first slice. A platform/native-library blocker is attached verbatim to the bead; it is never silently relabeled green.

### 13.2 core unit/oracle matrix

| Surface | Minimum tests |
|---|---|
| Position/math | wrap, toroidal delta, rotation, boundary and non-finite rejection |
| Index | empty, one, dense, tiny torus, radius > half extent, no duplicates, brute-force parity |
| Eyes | center, boundary, outside, rotations, colors, 8+ neighbors, scalar/SIMD |
| Blood | FOV boundaries, wound scaling, toroidal target |
| Brain inputs/outputs | typed mapping, range, finite values, known fixtures |
| Movement | differential drive, boost, wrap, terrain slope |
| Food/ecology | growth/decay/diffusion/intake, ledger, caps, starvation |
| Combat | reach/facing/cost/damage, boost channel, death attribution |
| Reproduction | gates, cost, placement, lineage, genome mutation/crossover, bound evaluator |
| Population | floor/injection/closed-world semantics |
| Persistence | cadence aggregation, writer failure, bounded queue |
| Replay | nonempty events, checkpoint, first divergence, final digest |
| Config | every invalid boundary, NaN/infinity, patch conflicts, scenario round trip |

### 13.3 cross-feature determinism matrix

For fixed manifests and command traces, run the appropriate determinism tier across:

- `parallel` on/off;
- `simd_wide` on/off;
- one Rayon thread versus several;
- storage enabled/disabled;
- persistence interval variants;
- UI snapshot cadence variants;
- null/TUI/Bevy/GPUI adapters;
- native and WASM-supported subset where floating semantics permit;
- release and debug assertions where behavior should match.

Exact digests are required only inside the declared bit-exact lane. Cross-feature/cross-target runs compare field-specific tolerances, stage oracles, and invariants. Whole-world “approximately equal” is not enough for bit-exact replay, while requiring identical chaotic final hashes across different floating paths is also dishonest.

### 13.4 command/runtime tests

Required deterministic cases:

- start paused/running;
- pause while running;
- resume while paused;
- one step while paused;
- one step while running atomically pauses, advances exactly once, and remains paused;
- speed zero and maximum;
- config update while paused;
- optimistic revision conflict;
- 1, 32, 33, and 1,000 command bursts;
- slow/no snapshot consumer;
- slow/failing storage worker;
- multiple clients with total-order receipts;
- shutdown with empty and nonempty queues;
- auto-pause by age, population, spike, and combined triggers;
- renderer disconnect/reconnect;
- command sent with no renderer;
- repeated idempotent shutdown/reset policy.

### 13.5 TUI golden matrix

Capture full buffers for:

- viewports `40x12`, `60x20`, `80x24`, `80x36`, `120x40`, `160x50`, `200x60`;
- initial meadow;
- ten and one hundred ticks;
- paused and one-step receipts;
- command palette open/filter/error/success;
- selected agent and missing/dead selection;
- lineages, brain, experiment, replay, environment, diagnostics screens;
- empty/extinct world;
- disconnected host and storage lag warning;
- ASCII-only, 16-color, 256-color, truecolor, and wide-character profiles;
- resize sequences and narrow-to-wide state preservation.

Goldens include seed, viewport, capability profile, model digest, world digest, frame hash, and command receipts.

### 13.6 PTY lifecycle tests

Spawn the actual terminal binary and prove:

- raw mode and alternate screen activate;
- keyboard, mouse, paste, and resize are decoded;
- shortcuts change model/host state once;
- normal quit restores terminal;
- error return restores terminal;
- interrupt/termination path restores terminal within deadline;
- repeated startup/shutdown leaves no lingering task;
- 1,000-event storm remains responsive and bounded.

### 13.7 GUI live-path matrix

For Bevy and any retained GPUI frontend:

- construct actual application/window or offscreen render app;
- ingest canonical snapshots;
- run synchronization systems;
- exercise camera/input state;
- render actual shader/material pipeline;
- capture actual output;
- compare semantic regions and image metrics;
- detect GPU/device/adapter failures;
- prove repaint count has no effect on ticks;
- prove two windows have no effect on ticks.

Separate CPU semantic references remain useful for debugging, but filenames/tests state that scope.

### 13.8 API/MCP conformance

The REST and MCP surfaces share one service contract. Test:

- schema/version discovery;
- status/current snapshot;
- list/apply scenario;
- valid/invalid config patch;
- accepted/applied/rejected receipt lifecycle;
- pause/resume/step/speed;
- selected agent and lineage query;
- checkpoint/export;
- ASCII and PNG capture with real content;
- pagination and response size limits;
- disconnected/full queue/storage failure;
- graceful shutdown.

### 13.9 replay bundle E2E

1. Run a fixed scenario with nonzero births, deaths, combat, and interventions.
2. Save manifest, commands, events, checkpoints, digests, database, and artifacts.
3. Replay from tick zero and from a checkpoint.
4. Compare per-stage and final digests.
5. Mutate one command in a copy and prove first-divergence diagnostics.
6. Verify on a second checkout/target using the recorded toolchain and lock.

### 13.10 CI lanes

Keep lanes honest and resource-aware:

- **fast core:** format, unit tests, default/no-default core features;
- **static:** clippy, unsafe/policy scanners, feature manifest checks;
- **workspace CPU:** all crates and CPU-safe tests;
- **TUI:** buffer goldens and PTY on Linux/macOS/Windows where supported;
- **GPU:** real Bevy/retained GPUI capture on explicit Metal/Vulkan/D3D12 runners;
- **WASM:** build and browser digest/front-end smoke;
- **replay:** bundle verification;
- **performance:** scheduled/manual with stored baselines, not every noisy PR;
- **dependency:** lock consistency, advisories, outdated report without auto-upgrade.

Invalid commands and nonexistent features fail a CI manifest linter before expensive jobs start.

---

## 14. Performance and Resource Budgets

Budgets are initial targets and must be recalibrated from captured baseline hardware. They prevent architecture regressions; they are not unsupported marketing promises.

### 14.1 simulation budgets

For the standard benchmark scenario:

- 1k agents: sustain at least 60 simulation ticks/s on the designated reference machine;
- 10k agents: publish measured TPS and avoid catastrophic scaling;
- scalar/SIMD fixes: no more than 10% regression without correctness/perf justification;
- one tick allocates no per-agent brain activation maps unless inspection is requested;
- spatial queries deliver each neighbor at most once;
- deterministic mode has a measured, documented overhead.

### 14.2 snapshot/projection budgets

- dynamic snapshot p95 under 4 ms at 1k agents and under 16 ms at 10k on reference hardware;
- terrain/static layers allocate only on revision;
- latest-snapshot backlog is bounded to one or a tiny fixed number;
- projection lock/ownership model has no world mutex in frontends;
- no full-agent sorting for panels that need only top-K;
- selected brain introspection is on demand.

### 14.3 TUI budgets

- input-to-frame p95 under 50 ms;
- 60 Hz event storm does not starve snapshot/control receipt handling;
- stable frame allocation after warm-up through grow-only buffers;
- no unbounded log/event history;
- CPU use while paused remains low;
- buffer work scales with visible viewport, not full world resolution.

### 14.4 GUI budgets

- 60 FPS target at 1k agents on reference GPU;
- 30 FPS minimum target at 10k agents in performance preset;
- p95 frame time and hitch p99 recorded;
- draw calls and materials scale by batches/classes, not agents;
- no live GPU readback;
- resize does not recreate global resources continuously;
- screenshot capture may read back asynchronously and reports its cost;
- 30-minute soak has bounded CPU/GPU memory and snapshot queues.

### 14.5 storage budgets

- lifecycle and command events are lossless;
- queue capacity and saturation policy are explicit;
- strict runs pause before lossless high-water overflow and fail closed after the configured unrecoverable-storage deadline;
- writer lag is visible;
- flush on graceful shutdown completes within configured deadline or reports failure;
- batch size balances FrankenSQLite transaction throughput without cloning render data;
- multi-run queries have indexes/ordering backed by measured plans.

### 14.6 build budgets

- default TUI/server product does not compile Bevy, GPUI, WGPU, Candle, Tch, or Tract;
- each optional rich backend is separately measurable;
- no mutable dependency source invalidates caches unpredictably;
- record clean/check incremental times and peak memory;
- RCH is preferred when healthy; local target artifacts remain on the external scratch volume.

---

## 15. Observability and Failure UX

Every frontend exposes a small health model:

- host state and last tick deadline;
- command queue depth/capacity;
- last applied/rejected command;
- snapshot revision/age;
- storage queue/lag/error;
- deterministic mode and last digest;
- active scenario/run/seed;
- renderer/backend/adapter;
- dropped UI snapshots;
- non-finite/invariant warnings;
- unbound/invalid brain count.

Errors are actionable:

- renderer unavailable says which compiled products exist;
- GPU adapter failure offers terminal/server fallback where authorized;
- command full distinguishes retryable overload from rejection;
- storage failure says whether scientific events are buffered, blocked, or lost;
- replay divergence names the first tick/stage/agent;
- missing golden gives the exact explicit bless command, never auto-generates silently;
- unsupported run schema identifies migration/reader options.

Structured tracing uses run ID, command ID, agent UID, tick, and component fields. Per-frame/per-agent info logging is prohibited in normal mode.

---

## 16. Risk Register

| Risk | Likelihood | Impact | Mitigation / decision gate |
|---|---:|---:|---|
| Runtime rewrite changes simulation behavior | High | High | characterize first, null adapter, digest traces, small semantic beads |
| Existing tests encode bugs | High | High | legacy micro-oracles and target contracts before fixes |
| De-monolith move hides semantic changes | Medium | High | one mechanical extraction, full isomorphism gates, separate commits |
| RCH remains unavailable | Medium | Medium | external local target dir, sequential builds, record infrastructure blocker |
| GPUI direct texture is unavailable | High | Medium | time-box spike; make Bevy primary, dashboard-only GPUI |
| Bevy instancing migration is large | High | High | snapshot boundary first, one release at a time, actual goldens/perf |
| FrankenTUI pre-1.0 API/license friction | Medium | High | exact pin, license review, façade, simulator tests |
| Asupersync/Tokio coexistence becomes complex | Medium | Medium | use Asupersync only behind runtime/storage ports initially |
| RNG replacement changes all traces | High | High | versioned algorithm, new baseline epoch, matched oracle tests |
| Brain genome redesign loses old scaffolding | Medium | High | no users/backcompat required, but fixture/codec tests and explicit schema |
| FrankenSQLite schema migration loses data | Medium | High | new run-scoped schema, transactional migration tests, copy/verify, no destructive commands |
| Large sibling crates inflate build | High | Medium | exact optional adapters, offline-only features, size/time spike |
| Scientific defaults are overfit to one seed | Medium | High | seed cohorts, viability envelopes, effect sizes |
| Visual goldens are platform-noisy | High | Medium | actual path plus semantic regions/tolerances and backend-specific metadata |
| GPU CI unavailable/flaky | Medium | Medium | CPU-safe PR lane plus explicit GPU runners; no substitute claims |
| Dirty shared worktree conflicts | Medium | High | preserve unrelated diffs, path-scoped edits/commits, no stash/reset |
| Broad dependency upgrade exceeds review | High | High | library-updater circuit breakers, one dependency/family per bead |
| Historical docs continue misleading users | High | Medium | supersession notice now; truth rewrite after evidence gates |

---

## 17. Bead Model and Dependency Graph

The tracker should contain one top-level epic and focused child epics. Beads carry full descriptions, acceptance criteria, design notes, and links back to this plan. Dependencies represent real prerequisites, not an arbitrary phase list.

### 17.1 top-level epics

| Symbol | Bead | Epic | Priority |
|---|---|---|---:|
| `REVIVE` | `bd-2z0` | Revive ScriptBots into Evolution Lab | P0 |
| `TRUTH` | `bd-2z0.1` | Baseline, reproducibility, and honest tests | P0 |
| `CORE` | `bd-2z0.2` | Scientific kernel correctness | P0 |
| `BRAINS` | `bd-2z0.3` | Heritable brain genomes and families | P0 |
| `RUNTIME` | `bd-2z0.4` | Sole-owner simulation host and protocol | P0 |
| `DATA` | `bd-2z0.5` | Replay, persistence, and experiment data | P1 |
| `TUI` | `bd-2z0.6` | FrankenTUI Evolution Lab | P1 |
| `GUI` | `bd-2z0.7` | Primary native GPU Evolution Lab | P1 |
| `DEPS` | `bd-2z0.8` | Dependency reproducibility and serialized modernization | P0 |
| `MONO` | `bd-2z0.9` | Isomorphic monolith extraction | P1 |
| `SCENARIOS` | `bd-2z0.10` | Curated scenarios and first-run product | P1 |
| `SCIENCE` | `bd-2z0.11` | Experiments and offline analysis | P2 |
| `WEB` | `bd-2z0.12` | Web and agent interfaces | P2 |
| `DOCS` | `bd-2z0.13` | Truthful docs, packaging, and release | P1 |

The validated tracker snapshot contains 103 issues: 14 epic containers and 89
focused executable leaves. The 184 blocking edges are leaf-to-leaf prerequisites;
the remaining dependency records are hierarchy. `br dep cycles --json` reported
zero cycles. All expensive `bv --robot-insights` metrics reported `computed`.
Because `bv 0.16.0` currently treats `parent-child` hierarchy as a blocker in
`--robot-next`, execution uses `br ready --json` as the actionability authority
and `bv` for centrality, critical-path, parallel-track, and priority analysis.
Unfinished epics are not falsely closed merely to work around that viewer bug.

### 17.2 critical path

```text
TRUTH baseline
   +--> lock + dated toolchain + Git-source pins
           +--> CORE sensing/output/persistence accounting fixes
                   +--> AgentUid + minimal RandomStream protocol
                           +--> brain genome/evaluator-state protocol
                                   +--> family adapters + inheritance
                                           +--> canonical digest/checkpoint skeleton
                                                   +--> core side-effect extraction
                                                           +--> runtime ports + Asupersync decision
                                                                   +--> pure HostCore
                                                                           +--> native scheduler
                                                                                   +--> snapshot hub/projections
                                                                                           +--> TUI / GUI / control / DATA

TRUTH manifest/scenario startup -------------------------------> scenario workflows
canonical digest + honest brains + RNG ------------------------> meadow/scenario tuning
live current-version GUI harness + runtime ownership ----------> Bevy/WGPU migrations
confirmed retention decision ---------------------------------> GPUI/legacy-TUI extraction
```

Monolith inventory can proceed after explicit confirmation. A mechanical extraction lands only from a fresh exact-HEAD baseline and depends on semantic edits to the same surface being complete/frozen. Major dependency families wait for the boundaries and live tests they need. Research beads may run in parallel; dependency manifest/lock mutations form one exclusive serialized lane.

### 17.3 first executable slice

The first implementation slice should be small enough to prove the workflow and valuable enough to change reality:

1. Create/validate the bead graph.
2. Complete the reproducibility prerequisite: baseline, tracked lock, dated toolchain, and exact current GPUI revision. Preserve the unrelated existing `.gitignore` change through explicit coordination/minimal hunk editing.
3. Claim the highest ready P0 correctness bead selected by `bv --robot-triage`.
4. Prefer the default SIMD eyesight oracle/fix if graph ranking agrees, because it is a contained scientific defect that makes agents effectively blind and has a crisp red test.
5. Run focused default and no-SIMD tests, workspace check, clippy, format, and UBS on changed files.
6. Close only with evidence attached to the bead.
7. Re-run `bv --robot-triage` for the next ready foundation bead.

The large `SimulationHost` work begins with characterization/contract tests, not a giant replacement commit.

### 17.4 bead quality rules

Every implementation bead includes:

- problem/evidence with file references;
- desired behavior;
- explicit non-goals;
- affected modules/features/platforms;
- acceptance tests and commands;
- performance/resource gate when relevant;
- dependency relationships;
- rollback/isomorphism note;
- documentation impact;
- no-deletion reminder where retirement is involved.

Roadmap headings are not automatically one bead. Split any change likely to exceed ten files into research/decision, characterization, protocol, individual adapter, migration, and E2E beads. Use `--parent` only for hierarchy; never make children depend on an open epic. Dependency mutation beads reserve their manifests plus `Cargo.lock`, declare the exact old/target package and expected lock delta, and stop on unrelated changes.

### 17.5 graph validation

After creation:

- `br dep cycles` must report none;
- `br ready --json` must show sensible first work;
- `bv --robot-triage` must identify P0 foundations rather than polish;
- `bv --robot-plan` must expose parallel tracks;
- `bv --robot-insights` status fields must be checked for computed versus timed-out metrics;
- priority misalignments and missing dependencies are corrected;
- tracker JSONL is flushed and reviewed.

---

## 18. Definition of Done

The revival is complete only when all of the following are true.

### 18.1 it works

- default launch succeeds and shows a living, understandable meadow scenario;
- explicit terminal and GUI modes work on supported platforms;
- pause/resume/speed/step behave identically from UI, REST, and MCP;
- single-step advances exactly once;
- closing/occluding/repainting frontends never changes scientific time;
- errors and fallbacks are truthful.

### 18.2 it evolves

- every default agent has a real heritable brain genome;
- children inherit/mutate/cross brain state;
- MLP, DWRAON, and Assembly can be compared honestly;
- placeholder backends are absent from default populations;
- lineage/provenance is visible and persisted.

### 18.3 it is reproducible

- toolchain, lock, Git dependencies, features, seed, config, and brain versions are recorded;
- checkpoint/replay digests match;
- empty evidence cannot pass;
- scalar/SIMD and serial/parallel results meet the declared contract;
- run bundles verify on a clean checkout.

### 18.4 it looks cool

- FrankenTUI has responsive canvas, inspector, timeline, charts, palette, and science screens;
- primary GUI uses actual batched GPU presentation, rich terrain/agents, camera, inspector, and experiments;
- both have accessibility and low-power/capability fallbacks;
- screenshots/goldens come from actual shipped paths.

### 18.5 it is useful

- curated scenarios tell distinct ecological stories;
- experiments compare matched seeds/variants;
- lineage, phenotype, interaction, and intervention analyses are available;
- FrankenSQLite and exports include provenance;
- external agents can inspect and intervene through acknowledged commands.

### 18.6 it is maintainable

- no god renderer or core file remains without an extraction plan/gate;
- mechanical and semantic changes remain separable;
- default dependency graph is lean and reproducible;
- every direct dependency has a disposition and update evidence;
- CI commands/features are real;
- README and help match verified behavior;
- no files were deleted without explicit written permission.

---

## 19. Immediate Next Actions

1. **Done:** complete and record the current host-target build/test baseline.
2. **Done:** review this plan in four adversarial passes: correctness, dependency/sequence, test honesty, and product usefulness.
3. **Done:** add a supersession pointer to the older primary port plan.
4. **Done:** convert the roadmap into 14 epics and 89 focused executable Beads through `br`.
5. **Done:** validate 184 blocking edges with `br` and robot-only `bv`; zero cycles and all graph metrics computed.
6. **Done:** freeze the existing lock/GPUI source in `37bca1f`, then prove and pin `nightly-2026-07-09` with MSRV 1.88 in its own bead.
7. **Done:** implement the bounded `RunManifestV0`/component-digest/trace foundation with two reviewed fixed-seed sequences and a real CLI probe.
8. **In progress:** the across-board dependency/feature ledger is complete;
   execute its serialized dead/redundant-declaration slices, then the eyesight
   oracle/fix as the first scientific slice.
9. Await explicit confirmation before creating the de-monolith skill workspace.
10. Continue until both frontends and the scientific runtime have real end-to-end evidence—not merely compilation or substitute images.

---

## Appendix A — Observed Baseline on 2026-07-11

This is observed evidence, not an evergreen status claim. Beads update it as the baseline changes.

### A.1 repository/tracker

- branch `main` matched `origin/main` at commit `38d59d6` when the audit began;
- pre-existing user changes were limited to `.beads/.gitignore` and `.gitignore` and were preserved;
- Beads database was healthy but empty (`br list --json` and `br ready --json` returned `[]`);
- `cargo fmt --all --check` passed before source edits;
- `Cargo.lock` existed but was ignored and untracked.
- the active compiler was floating `rustc 1.99.0-nightly (375b1431b 2026-07-10)` with `cargo 1.99.0-nightly (59800466c 2026-07-07)` on `aarch64-apple-darwin`;
- `rust-toolchain.toml` named undated `nightly`, requested `rustfmt`, `clippy`, and `rust-analyzer`, and installed only the `x86_64-unknown-linux-gnu` extra target;
- `cargo metadata --no-deps` reported eleven workspace crates. Core defaults enabled `parallel` and `simd_wide`; application defaults enabled placeholder `ml`, `neuro`, and `fast-alloc` while omitting both GUI backends.

### A.2 build infrastructure

- `rch doctor` initially reported healthy configuration, but all workers were unreachable and status fell back local-only;
- RCH’s canonical project root was on `/Volumes/USBNVME16TB/data/projects`, while this checkout is under `/Users/jemanuel/projects`;
- fail-open injected a Linux target into a macOS local build and failed in Wayland/pkg-config, so it was not accepted as project evidence;
- a dangling `~/.cargo/git` target directory was repaired by creating the missing external cache directory only; no repository file changed;
- authoritative local artifacts use `/Volumes/USBNVME16TB/temp_agent_space/rust_scriptbots_host_target`.

### A.3 full host-target check

Command:

```bash
env CARGO_TARGET_DIR=/Volumes/USBNVME16TB/temp_agent_space/rust_scriptbots_host_target \
  cargo check --workspace --all-targets --target aarch64-apple-darwin
```

Result: failed in `scriptbots-render` against the captured floating GPUI resolution after roughly thirteen minutes. Observed errors included:

- `flex_grow` now requires an `f32` argument;
- `gpui::Application::new` no longer exists;
- stale test initializers use removed `AgentInstance::size` fields;
- stale integer selection values no longer match `f32`;
- one unused-assignment warning in renderer code;
- one dead-code warning in the app’s non-Linux GUI helper.

This proves the GUI feature graph does not compile from the current mutable dependency source.

### A.4 default app/TUI tests

The default-feature app unit tests compiled and passed their small unit groups. The integration run then entered `terminal_headless_applies_control_updates`:

- it creates 48 agents;
- reproduction rates are 140 and a queued update asks for 420;
- headless mode does not drain that queued update;
- the initial rates alone cause explosive reproduction;
- the test consumed roughly seven CPU cores for several minutes and did not finish;
- it was interrupted safely and recorded as a bounded resource/runaway failure.

The focused `terminal_smoke` suite passed two tests. That green result is misleading by design: bootstrap advances 120 ticks, headless draws/steps twelve more frames, but the completion log reads the last persistence history at tick 120. The test’s `final_tick=120` assertion therefore confirms stale reporting rather than correct tick semantics.

### A.5 core tests

Command:

```bash
env CARGO_TARGET_DIR=/Volumes/USBNVME16TB/temp_agent_space/rust_scriptbots_host_target \
  cargo test -p scriptbots-core --target aarch64-apple-darwin -- --nocapture
```

Result: 41 passed, 1 failed. `world_state_initialises_from_config` hard-codes a `100x100` food grid, while the current default `6000x3000` world and 50-unit cells correctly construct `120x60`. This is another stale test/claim, independent of the newly discovered scientific defects.

### A.6 workspace test lane

Command:

```bash
env CARGO_TARGET_DIR=/Volumes/USBNVME16TB/temp_agent_space/rust_scriptbots_host_target \
  cargo test --workspace --all-targets --target aarch64-apple-darwin
```

Result: failed before test execution in `scriptbots-render` after compiling the
full default/Bevy/GPUI/legacy-storage graph. The decisive errors were the two missing
`flex_grow(f32)` arguments and removed `Application::new`; the renderer also
reported the unused `selected_after` assignment. This is a code/dependency API
failure, not a test assertion or unavailable GPU.

### A.7 clippy lane

Command:

```bash
env CARGO_TARGET_DIR=/Volumes/USBNVME16TB/temp_agent_space/rust_scriptbots_host_target \
  cargo clippy --workspace --all-targets --target aarch64-apple-darwin -- -D warnings
```

Result: failed in `scriptbots-core` before reaching the renderer. The floating
nightly promotes ten `chunks_exact_to_as_chunks` findings to errors under
`-D warnings`; Cargo also reports the unused `num-bigint-dig` patch. These are a
code/toolchain-lint baseline and a dependency-manifest baseline, respectively.

### A.8 executable CLI and TUI lifecycle

The default binary compiled and `--help` exited zero. Its public mode enum lists
`auto`, `gui`, `bevy`, and `terminal`; there is no explicit `headless` mode even
though replay, determinism, and profiling paths run headlessly.

Command:

```bash
env CARGO_TARGET_DIR=/Volumes/USBNVME16TB/temp_agent_space/rust_scriptbots_host_target \
  cargo run -p scriptbots-app --target aarch64-apple-darwin --bin scriptbots-app -- \
  --mode terminal --storage memory --low-power
```

Result under a real PTY: the terminal entered alternate-screen mode, rendered,
accepted `q`, restored the terminal, and exited zero. The first visible screen
already reported tick 120 and 17 agents, confirming the hidden bootstrap and
stale-history concerns. This is a green lifecycle probe with red product/time
semantics; it is not evidence that TUI controls or rendering are correct.

### A.9 bounded explicit macOS GUI probe

Command:

```bash
gtimeout 20s env \
  CARGO_TARGET_DIR=/Volumes/USBNVME16TB/temp_agent_space/rust_scriptbots_host_target \
  cargo run -p scriptbots-app --features gui --target aarch64-apple-darwin \
  --bin scriptbots-app -- --mode gui --storage memory --low-power
```

Result: exit 124 after twenty seconds while compiling the GUI-specific graph;
no window launched. The already completed workspace check/test lanes prove that
the same source graph is code-red in `scriptbots-render`, so the timeout is
classified as a bounded launch/build failure, not a working GUI or a hardware
failure. No live renderer golden can be generated until that compile baseline is
repaired.

### A.10 tracker conversion and graph validation

- `br sync --flush-only` exported 103 issues and 286 total dependency records;
- 184 records are explicit blocking edges and the remainder are hierarchy;
- `br dep cycles --json` returned `count: 0`;
- `bv --robot-insights` reported PageRank, betweenness, eigenvector, HITS,
  critical path, cycles, k-core, articulation, and slack as `computed`;
- the critical path begins `bd-2z0.1.1` (baseline), `bd-2z0.1.2` (lock/Git
  pins), `bd-2z0.1.8` (dated toolchain), and `bd-2z0.1.6` (manifest/digest);
- `br ready --json`, filtered to non-epics, exposed only `bd-2z0.1.1` before it
  was claimed.

### A.11 known static contradictions awaiting focused red tests

- GPUI two-window repaint stepping;
- GPUI command queue never drained;
- TUI direct-plus-queued double step;
- headless TUI command queue never drained;
- SIMD eye chunk factor inverted;
- scalar eye heading double-added;
- offspring brain unbound;
- combat boost reads color output;
- replay emits no events;
- lifecycle records cleared between persistence intervals;
- macOS explicit GUI misdetected as headless;
- visual goldens absent and live render paths untested.

### A.12 dated-toolchain proof

`nightly-2026-07-09` was proven in a separate clean external target directory
before it became the repository default. It identifies itself as `rustc
1.99.0-nightly (14cae6813 2026-07-08)` and Cargo commit `59800466c`.

- locked metadata and formatting pass;
- the default application host-target check passes with its pre-existing
  non-Linux helper warning;
- `terminal_smoke` passes 2/2 and the core suite reproduces 41 pass/1 stale-grid
  failure;
- the full workspace/all-targets check reaches the same `scriptbots-render`
  GPUI/test-initializer failures, with no MSRV or language incompatibility;
- pinned clippy stops first at the existing collapsible nested `if` in
  `scriptbots-index`, while the floating nightly stopped at newer core
  `chunks_exact_to_as_chunks` lints. This difference is classified as a moving
  lint-toolchain baseline, not a semantic regression;
- all eight CI/release toolchain inputs now name the same date. Mutable action
  revisions remain red and belong to the dedicated CI-truth bead.

### A.13 RunManifestV0 and characterization oracle proof

`bd-2z0.1.6` now has a real headless artifact path rather than another replay
claim over empty data:

- `CharacterizationDigestV0` hashes separate agent, food, terrain, hydrology,
  cloned-RNG-probe, and brain-registry components plus an overall boundary
  digest;
- the encoder uses explicit domain tags, sequence lengths, little-endian integer
  bytes, fixed enum tags, and `f32::to_bits()` with FNV-1a-64 V0. Reference
  vectors and `+0.0` versus `-0.0` are tested;
- agents are sorted by raw generational handle. The digest rejects pending
  deaths, spawns, or simulation commands instead of silently hashing a partial
  tick;
- selection/indicator state, mutation-log strings, history, analytics,
  persistence, scratch/index state, wall-clock map timestamps, opaque evaluator
  state, and restorable RNG state are explicitly excluded. The manifest labels
  this `characterization_only`, ties comparisons to one pinned build lane, and
  names `WorldDigestV1` as the superseding format;
- `RunManifestV0` records the explicit root seed, normalized effective config
  and a separately RON-encoded config digest, ordered config-layer content
  digests, temporary scenario/population recipe, sorted brain roster,
  package/source/toolchain/lock/features/target/thread provenance, tracked Git
  diff and status digests, and machine-readable limitations. Entropy-seeded
  characterization is rejected. Missing or dirty provenance is surfaced and
  never upgraded to a reproducibility claim;
- `--characterize-v0 TICKS` captures tick zero through the requested boundary,
  is capped at 256, skips the hidden 120-tick bootstrap and every
  storage/server/renderer path, and emits canonical JSON to stdout or an
  explicitly requested file.

Focused host-target evidence on pinned `nightly-2026-07-09`:

- core characterization/FNV/float-bit tests: 5/5 pass;
- application manifest/trace tests: 4/4 pass;
- CLI parsing/constraint tests: 2/2 pass;
- a fresh-environment no-default-feature application library check passes with
  no new warning;
- two inline two-agent traces are frozen for seeds `0xC0FFEE` and `0xBAD5EED`;
- `ubs --diff --only=rust .` reports zero critical findings and exits zero;
- clippy reaches only the previously recorded `scriptbots-index`
  `collapsible_if`, core `chunks_exact_to_as_chunks`, server sort, and terminal
  nested-if/sort baselines. No new characterization lint appeared before those
  known gates.

Real binary probe:

```bash
RAYON_NUM_THREADS=1 SCRIPTBOTS_MAX_THREADS=1 \
  CARGO_TARGET_DIR=/Volumes/USBNVME16TB/temp_agent_space/rust_scriptbots_nightly_2026_07_09_target \
  cargo run --quiet --target aarch64-apple-darwin -p scriptbots-app \
  --bin scriptbots-app --no-default-features -- \
  --rng-seed 12648430 --characterize-v0 4
```

The canonical artifact contained exactly ticks `[0,1,2,3,4]`, population recipe
`legacy_4x4_grid_v0`, `bootstrap_ticks: 0`, and overall digests
`cdead2e86466d7ab`, `7e92bca0ab0ed4ef`, `539de76e125fc73d`,
`75cd2ff76928a8c0`, and `ec0e84c64e13d02a`. It correctly set
`reproducible: false` with one warning because the shared checkout was dirty.
The dirty status and tracked diff are themselves fingerprinted; untracked file
contents remain explicitly outside V0.

RCH was healthy but still rejected this `/Users/jemanuel/projects` checkout
against its `/Volumes/USBNVME16TB/data/projects` canonical root, then attempted
local fallback. That fallback was interrupted and every accepted artifact was
rebuilt explicitly for `aarch64-apple-darwin` in the external NVMe target. No
dependency or new source file was added. Physical extraction of the temporary
V0 implementation remains gated on the user's required de-monolith workspace
confirmation.

When an unrelated process saturated the shared USB Cargo cache, the final fresh
proof used `/tmp/rust_scriptbots_manifest_cargo_home` and
`/tmp/rust_scriptbots_manifest_target` with incremental compilation disabled.
Those directories were left intact. The cold application graph downloaded and
compiled Arrow, the former bundled native database, three Crossterm generations, two Ratatui-era
stacks, HTTP/TLS, and MCP dependencies even with application default features
disabled; that observed cost is direct input to `bd-2z0.8.1` dependency pruning.

### A.14 dependency and feature-graph disposition proof

`bd-2z0.8.1` audited all twelve manifests, 140 member dependency edges,
31 root workspace policies, the unused patch, all features/targets, and the
1,184-package lock. `UPGRADE_LOG.md` now assigns all 70 unique directly named
external crates and all 22 local path edges a keep/remove/relocate/hold/migrate
decision, owner bead, license/MSRV risk, and proof gate.

The final online Cargo dry run proposed no compatible updates and identified 47
packages behind a breaking or graph-constrained boundary. It did not mutate the
lock. Crates.io metadata was refreshed for every direct registry crate. No
incompatible direct license was found. Three latest candidates exceed the
declared Rust 1.88 floor—Wide 1.5 requires 1.89, Tract 0.23 requires 1.91, and
Bevy 0.19 requires 1.95—so none can be treated as a routine bump.

The audit distinguished declaration grep from actual feature semantics:

- core getrandom 0.3 and web getrandom 0.2 have no API references but
  intentionally activate the correct WASM backend on two rand_core generations;
- render's direct Naga edge similarly activates `termcolor` on WGPU 27 and
  needs shader-diagnostic characterization before removal;
- twelve member declarations and six unused/redundant root policies are proven
  dead; the num-bigint-dig Git patch is separately proven unused by Cargo;
- app rand and render's brain edge are test-only and should become dev
  dependencies; libc/windows-sys should become target-specific.

The feature graph is not product-isolated. On macOS, app no-default still has
325 normal package-version nodes; defaults have 332, GUI-only 551, Bevy-only
531, GUI+Bevy 701, and all features 708. No-default still includes bundled
the former native analytics stack, TUI, HTTP/MCP/Swagger, Tokio/Reqwest, core parallel/SIMD defaults,
and platform helpers. The browser graph re-enables core defaults through
`scriptbots-brain`. macOS GUI feature-unifies both Metal and Vulkan on WGPU
27, while Bevy brings its broad default feature set.

This graph also explains a primary executable failure: app defaults omit the
GUI feature, but Auto selects GUI on a desktop display and then the uncompiled
GUI runner aborts. Explicit GUI mode follows a different fallback path.
Selection happens only after storage setup, a 120-tick bootstrap, and control
thread startup. The fix is owned by `bd-2z0.1.5`; dependency cleanup must not
mask it.

The next mutation is deliberately narrow: `bd-2z0.8.2` removes or relocates
only proven declarations, one manifest slice at a time, with reviewed lock
deltas and affected feature/target proof. Major library migrations remain
blocked behind their family beads.

---

## Appendix B — Sibling Repository Study and Safety Decision

The sibling repositories were inspected read-only. No stash, pull, checkout, reset, deletion, or tracked mutation was performed.

- `frankentui`: local/upstream/remote commit `fccff2a7e51d39a927bced882877a45aef5c8d39`; unrelated `.gitignore` and untracked files preserved. It is current, so no pull was needed.
- `asupersync`: bounded MPSC, blocking bridge, cancellation, and deterministic lab APIs were inspected. Use selectively around runtime/storage lifecycle, not as a scheduler for simulation math.
- `franken_numpy`: checkout is 433 commits behind and has a modified fuzz seed, but the relevant `fnp-random` source was verified identical to current `origin/main`; no pull was needed to evaluate it.
- `franken_networkx`: suitable for optional post-run lineage/interaction graph analysis, not the hot loop.
- `frankenpandas`: contains extensive existing working-tree changes and is too large/eager for the runtime; use external reporting handoff only.
- `frankenscipy`: contains extensive existing working-tree changes; selected statistics/FFT/clustering are optional post-run candidates with known caveats.

Remote-content inspection supplied the needed current API evidence without disturbing valuable sibling work. Any future pinned dependency adoption uses an immutable source/revision and its own build/license/conformance bead.
