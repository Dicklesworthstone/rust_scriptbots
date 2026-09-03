# AGENTS.md — rust_scriptbots

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE 0 - THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. "I think it's safe" is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for a confirmation that your understanding is correct. Only then may you execute it—if anything remains ambiguous, refuse and escalate.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time. If that record is absent, the operation did not happen.

---

## Git Branch: ONLY Use `main`, NEVER `master`

**The default branch is `main`. The `master` branch exists only for legacy URL compatibility.**

- **All work happens on `main`** — commits, PRs, feature branches all merge to `main`
- **Never reference `master` in code or docs** — if you see `master` anywhere, it's a bug that needs fixing
- **The `master` branch must stay synchronized with `main`** — after pushing to `main`, also push to `master`:
  ```bash
  git push origin main:master
  ```

**If you see `master` referenced anywhere:**
1. Update it to `main`
2. Ensure `master` is synchronized: `git push origin main:master`

---

## Toolchain: Rust & Cargo

We only use **Cargo** in this project, NEVER any other package manager.

- **Edition:** Rust 2024 (nightly required — see `rust-toolchain.toml`)
- **Dependency versions:** Explicit versions for stability
- **Configuration:** Cargo.toml workspace with `workspace = true` pattern
- **Unsafe code:** Warned (`#![warn(unsafe_code)]`)

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `gpui` | Zed's GPU-accelerated UI framework (native rendering backend) |
| `bevy` | ECS game engine (alternative rendering backend) |
| `wgpu` | Low-level GPU abstraction for custom world rendering |
| `tokio` | Async runtime for server/API/MCP endpoints |
| `axum` | HTTP API server framework |
| `ratatui` | Terminal UI framework for console mode |
| `fsqlite` (`=0.1.16`, rev `e536d7f8ca102b3eb5236bef48514582379f9346`) | FrankenSQLite persistence for simulation metrics, replay, and run artifacts |
| `rayon` | Data parallelism for simulation tick processing |
| `rand` | RNG with `SmallRng` for agent behavior |
| `slotmap` | Generational arena for agent handles (`AgentId`) |
| `candle-core` + `candle-nn` | ML inference for neural network brains |
| `tract-onnx` | ONNX model loading for brain inference |
| `tch` | PyTorch bindings for brain training/inference |
| `neuroflow` | Lightweight neural network library for agent brains |
| `serde` + `serde_json` | Serialization |
| `thiserror` | Ergonomic error type derivation |
| `tracing` | Structured logging and diagnostics |
| `wasm-bindgen` | WebAssembly bindings for browser target |
| `clap` | CLI argument parsing |
| `fastmcp-rust` | MCP server integration |
| `kira` | Audio engine (optional) |
| `wide` | Portable SIMD for vectorized simulation math |

### Release Profile

The release build optimizes for performance:

```toml
[profile.release]
opt-level = 3       # Maximum performance optimization
lto = "thin"        # Thin link-time optimization
codegen-units = 1   # Single codegen unit for better optimization
strip = true        # Remove debug symbols
panic = "abort"     # Abort on panic (smaller binary)
incremental = false # Disable incremental for release
```

---

## Code Editing Discipline

### No Script-Based Changes

**NEVER** run a script that processes/changes code files in this repo. Brittle regex-based transformations create far more problems than they solve.

- **Always make code changes manually**, even when there are many instances
- For many simple changes: use parallel subagents
- For subtle/complex changes: do them methodically yourself

### No File Proliferation

If you want to change something or add a feature, **revise existing code files in place**.

**NEVER** create variations like:
- `mainV2.rs`
- `main_improved.rs`
- `main_enhanced.rs`

New files are reserved for **genuinely new functionality** that makes zero sense to include in any existing file. The bar for creating new files is **incredibly high**.

### Signals Must Name What They Observed

A **signal** is anything a later reader trusts without re-deriving it: a log line, a return value, a test name, a report, a schema, a doc comment. **Before asserting an outcome, compute the observation that would falsify it:**

- **A guard** — a case that FAILS it. A guard you deleted also passes correct data.
- **A coverage report over a matrix** — that its cells DIFFER. A collapsed matrix reads as a thorough one.
- **A declared capability** (table, config knob, public API) — a consumer. Grep for a writer AND a reader.
- **A bound, count, or width** — derive it from the declaration it describes. **NEVER** copy it from prose; a bead sentence goes stale, a type does not.

**If you cannot compute it, do not drop the qualifier.** Name the evidence you actually have — "admitted", not "persisted" — and file the gap. An unqualified past-tense claim is read as observed.

Context: `bd-0oro` catalogues 18 instances of this in this codebase, with no shared author. The two existing mechanical defences are disjoint and cover under half; this form is the one no scanner can reach, because the discriminating value was never computed.

---

## Backwards Compatibility

We do not care about backwards compatibility—we're in early development with no users. We want to do things the **RIGHT** way with **NO TECH DEBT**.

- Never create "compatibility shims"
- Never create wrapper functions for deprecated APIs
- Just fix the code directly

---

## Compiler Checks (CRITICAL)

**After any substantive code changes, you MUST verify no errors were introduced:**

```bash
# Check for compiler errors and warnings (workspace-wide)
cargo check --workspace --all-targets

# Check for clippy lints (pedantic + nursery are enabled)
cargo clippy --workspace --all-targets -- -D warnings

# Verify formatting
cargo fmt --check
```

If you see errors, **carefully understand and resolve each issue**. Read sufficient context to fix them the RIGHT way.

---

## Testing

### Testing Policy

Every component crate includes inline `#[cfg(test)]` unit tests alongside the implementation. Tests must cover:
- Happy path
- Edge cases (empty input, max values, boundary conditions)
- Error conditions

Integration tests live in per-crate `tests/` directories (e.g., `crates/scriptbots-core/tests/`).

### Unit Tests

```bash
# Run all tests across the workspace
cargo test --workspace

# Run with output
cargo test --workspace -- --nocapture

# Run tests for a specific crate
#
# scriptbots-core: use --features economy-faults (bd-dz37). The economy fault-injection
# machinery is gated on that feature alone. It used to be reachable via
# cfg(any(test, feature = "economy-faults")), which gave every test build a `fault` field on
# ResourceLedgerState that production does not have — the tested binary was structurally not
# the shipped one. Without the feature, the bd-16g.11.2 mutation suite does not compile in and
# therefore does not run.
cargo test -p scriptbots-core --features economy-faults
cargo test -p scriptbots-brain
cargo test -p scriptbots-brain-ml
cargo test -p scriptbots-brain-neuro
cargo test -p scriptbots-storage
cargo test -p scriptbots-render
cargo test -p scriptbots-app
cargo test -p scriptbots-index
cargo test -p scriptbots-world-gfx
cargo test -p scriptbots-bevy
cargo test -p scriptbots-web

# Run tests with all features enabled
cargo test --workspace --all-features
```

### Test lanes on this host (RCH) — read before running the full workspace [2026-09-03, bd-rch-full-workspace-lane-5mff]

- The RCH worker fleet currently LACKS `libudev.pc`, so a plain `cargo test --workspace`
  fails on every worker at the `libudev-sys` build script (bevy 0.17 → gilrs gamepad
  stack; first observed 2026-09-03 on worker hz4, RCH-E307, after a 462s build). A local
  fallback is NOT an acceptable substitute on this host (see the RCH section: bd-e6ff).
- The documented remote lane until workers are provisioned with `libudev-dev` is:

  ```bash
  cargo test --workspace --exclude scriptbots-bevy --no-fail-fast
  ```

  This lane skips ONLY `scriptbots-bevy` (~187 of ~1,895 workspace tests, ≈10%). Every
  artifact, commit message, or report produced from this lane MUST say
  "workspace-minus-bevy" — never quote it as "workspace". `scriptbots-bevy`'s own lane
  stays `cargo test -p scriptbots-bevy` on a machine with libudev, or through the pinned
  DSR profile.
- RCH admission is bursty: `insufficient_total_slots` refusals (exit 103) are
  infrastructure, not test results. Retry with ~60s backoff; never record an exit-103
  run as a red or green suite.

### Test Categories

| Crate | Focus Areas |
|-------|-------------|
| `scriptbots-core` | World simulation, agent lifecycle, tick processing, spatial indexing, food/terrain systems, evolution, genome serialization |
| `scriptbots-brain` | Brain trait contracts, MLP forward pass, DWRAON network, Assembly brain, mutation/crossover |
| `scriptbots-brain-ml` | Candle/Tract/Tch dependency probes and sensor-copy placeholder; model inference remains open |
| `scriptbots-brain-neuro` | Neuroflow brain wrapper, training, serialization |
| `scriptbots-index` | Uniform-grid neighbor queries and boundary conditions; R-tree/k-d backends remain open |
| `scriptbots-storage` | FrankenSQLite persistence, metric/replay recording, bounded admission, flush/shutdown receipts, analytics snapshots |
| `scriptbots-render` | GPUI rendering, camera controls, world visualization, audio integration |
| `scriptbots-world-gfx` | wgpu pipeline, shader compilation, offscreen readback, compute binning |
| `scriptbots-bevy` | Bevy ECS integration, entity spawning, system scheduling |
| `scriptbots-app` | CLI parsing, server startup, TUI mode, MCP endpoints, control commands |
| `scriptbots-web` | WASM bindings, browser interop, postcard serialization |

### Performance Budget Gate

`scripts/perf_gate.sh` is the executable regression sentinel for the recovery plan's CPU budgets. It runs fresh deterministic worlds for the production-default MLP and NeuroFlow families, uses three warmups plus five measured repetitions, records every raw nanosecond sample, and gates on the median of the five whole-run TPS values so infrequent cadence work cannot disappear behind a median of short windows. Each profiled simulation tick records five separately timed dynamic snapshots, giving the default 200-tick repetition 1,000 raw snapshot observations; the per-run p95 therefore has enough upper-tail observations for the unchanged 5% cross-run CV gate without averaging away individual latency samples. It emits:

- `perf_result.json` — scenario inputs, raw TPS windows, raw snapshot samples, per-stage timings, digests, and derived statistics;
- `fingerprint.json` — the comparison machine class plus exact raw host evidence, toolchain, build target, thread budget, filesystem, lockfile blob, and source commit;
- `perf_verdict.json` — the machine-readable pass/fail/advisory/refusal decision;
- `perf_summary.md` — the human comparison report retained in the DSR evidence bundle;
- `perf_baseline.json` — only for an explicitly requested, admissible baseline candidate.

The short lane covers 1k agents; the full lane covers 1k and 5k. Both cover every default brain family. That is the explicit `bd-2z0.8.18` regression-gate matrix; the plan's separate 10k publication target remains with the `bd-h33` optimization/baseline program and is not silently claimed by this harness. The stable gates are a regression greater than 10% from the exact-class baseline, less than 60 TPS at 1k, or dynamic-snapshot p95 greater than or equal to 4 ms at 1k. Per-metric run-to-run CV over 5% makes only that noisy metric advisory; it must not conceal a stable failure in another metric or scenario. A dirty checkout makes the whole local comparison advisory.

Performance execution is DSR-only. Never invoke the wrapper, its underlying Cargo benchmark,
RCH, or a hosted workflow directly. Put the desired self-test, record, or comparison command in a
pinned DSR repository profile and launch that profile with a unique version and `--no-sync`:

```bash
dsr build --tool <pinned-scriptbots-profile> \
  --target darwin/arm64 --no-sync --version <unique-proof-version>
```

The DSR profile must bind the clean `main` checkout to an expected source commit, use an external
unique proof directory, retain every JSON artifact and raw log, and assert the required typed
verdict with `jq`; process exit zero alone is not sufficient. The wrapper derives the executable
host target from `rustc -vV`; `SCRIPTBOTS_PERF_BUILD_TARGET` is the explicit override inside the
profile. The checked-in golden's exact DSR machine class is the comparison authority. A different
CPU, OS, filesystem, toolchain, memory bucket, or build configuration must refuse comparison; do
not spoof hosted-runner fields or hand-normalize the fingerprint.

Comparisons are exact-class only. A CPU, runner image, OS/architecture, filesystem, memory-capacity bucket, full `rustc -Vv` identity, effective Cargo target/linker/rustflags configuration, Rayon/thread budget, feature set, or scenario/science-digest mismatch returns refusal (exit 2); the harness never invents a cross-class delta. The comparison class rounds installed memory upward to the next 256 MiB capacity tier so reserved-page `MemTotal` jitter cannot create a false class, while `fingerprint.memory` retains the exact raw value for audit. Exit 1 is a stable budget failure. Exit 0 covers pass, explicit bootstrap-required, baseline-candidate, and advisory results, so automation must also inspect `perf_verdict.json` when it expects a particular proof state.

DSR treats that typed verdict more strictly than the executable contract. A final short/full proof
accepts only `pass`, requires both process and typed exit code to be zero, independently requires
`perf_result.json.fingerprint.git_dirty` to be `false`, and binds the artifact source and
fingerprint commit to the profile's expected commit. It fails closed on missing artifacts,
advisory, bootstrap-required, baseline-candidate, stable failure, class refusal, or an unknown
status. Baseline recording is a separate DSR pass and can never approve itself as final comparison
evidence.

Re-baselining is a reviewed golden-file operation:

1. Run a clean-`main`, expected-commit DSR profile in explicit baseline-recording mode with a
   non-empty justification and a unique external proof directory.
2. Require `baseline_candidate`, verify byte identity between `perf_result.json` and
   `perf_baseline.json`, then review all raw repetitions, CVs, absolute budgets, fingerprint fields,
   digest stability, and exact source/lockfile identity. Run a same-class readback comparison before
   promotion.
3. Materialize the reviewed artifact byte-for-byte as `ci/fixtures/perf_baseline.json` only at the
   end of that DSR pass. Commit it with a nonempty
   `Perf-Baseline-Justification: <reason>` trailer.
4. Update the profile's expected commit, switch it to comparison mode, and run a second clean DSR
   pass that requires typed `pass` against the checked-in golden.
5. Never bless a dirty, noisy, synthetic-delay, cross-class, or absolute-budget-failing result.
   Never hand-edit aggregate values; validation recomputes them from raw samples.

Any synthetic-delay proof is also run only inside DSR, remains ephemeral, and must never be
committed. `bd-h33` owns making the numbers better; `bd-2z0.8.18` owns detecting when they silently
get worse.

---

## Third-Party Library Usage

If you aren't 100% sure how to use a third-party library, **SEARCH ONLINE** to find the latest documentation and current best practices.

---

## rust_scriptbots — This Project

**This is the project you're working on.** rust_scriptbots is a transformative port of the original C++ ScriptBots evolutionary agent simulation into modern, idiomatic Rust. The original C++ code is preserved in `original_scriptbots_code_for_reference/` for reference.

### What It Does

Simulates a 2D world populated by autonomous agents with neural network brains that evolve over generations. Agents perceive their environment through eyes, make decisions via pluggable brain architectures (MLP, DWRAON, Assembly, Neuroflow, ML backends), and compete for food/survival. The simulation supports real-time visualization (GPUI, Bevy, wgpu, TUI), web deployment (WASM), an HTTP API with Swagger docs, and MCP server integration.

### Planning Document

The authoritative guide is `PLAN_TO_REARCHITECT_AND_REVIVE_RUST_SCRIPTBOTS.md`. The older GPUI port plan is retained as historical evidence and must not override the recovery plan, current source, executable tests, or Beads. Whenever you start a task from the active plan, immediately mark it in place with a bracketed notation such as `[Currently In Progress]` to avoid conflicts with concurrent agents.

In general, you should also try to follow all suggested best practices listed in `RUST_SYSTEM_PROGRAMMING_BEST_PRACTICES.md`.

For anything touching the franken-family libraries (fsqlite/asupersync/ftui/fnx/frankenpandas/fsci/ft/fnp), read `docs/franken_integration.md` first — program verdicts, constraint matrix, boundary rules, and CI guards (umbrella bead `bd-2js6`); licenses: `docs/licenses.md`.

### Architecture

```
User Input → CLI/API/MCP → ┬─ Control Commands ──→ WorldState (tick loop)
                           └─ Config Changes ────→ ScriptBotsConfig
                                                        │
WorldState::tick() ────→ ┬─ Sensor Collection (eyes, proximity, blood)
                         ├─ Brain Evaluation (pluggable: MLP/DWRAON/Assembly/ML/Neuro)
                         ├─ Agent Actions (movement, eating, reproduction, combat)
                         ├─ Food/Terrain/Hydrology Updates
                         ├─ Evolution (selection, crossover, mutation)
                         └─ Analytics (bounded FrankenSQLite storage worker)
                                    │
Render Layer ──────────→ ┬─ GPUI (native GPU-accelerated UI)
                         ├─ Bevy (ECS game engine)
                         ├─ wgpu (custom world renderer)
                         ├─ Ratatui (terminal TUI)
                         └─ WASM (browser via wasm-bindgen)
```

### Workspace Structure

```
rust_scriptbots/
├── Cargo.toml                              # Workspace root
├── crates/
│   ├── scriptbots-core/                    # World simulation, agents, evolution, spatial indexing
│   ├── scriptbots-brain/                   # Brain trait + impls (MLP, DWRAON, Assembly)
│   ├── scriptbots-brain-ml/                # ML dependency probes; inference placeholder
│   ├── scriptbots-brain-neuro/             # Neuroflow brain backend
│   ├── scriptbots-index/                   # Uniform-grid spatial indexing
│   ├── scriptbots-storage/                 # FrankenSQLite persistence worker + analytics snapshots
│   ├── scriptbots-render/                  # GPUI rendering + audio (kira)
│   ├── scriptbots-world-gfx/              # wgpu custom world renderer
│   ├── scriptbots-bevy/                    # Bevy ECS rendering backend
│   ├── scriptbots-app/                     # CLI, HTTP API, TUI, MCP server, main binary
│   └── scriptbots-web/                     # WASM/browser target
├── original_scriptbots_code_for_reference/ # Original C++ source
├── docs/                                   # Performance data, rendering references, WASM docs
├── scripts/                                # Build/run helper scripts
└── ci/                                     # CI configuration
```

### Key Files by Crate

| Crate | Key Files | Purpose |
|-------|-----------|---------|
| `scriptbots-core` | `src/lib.rs` | `WorldState`, `AgentData`, `AgentArena`, `AgentId`, `FoodGrid`, `TerrainLayer`, `ScriptBotsConfig`, `BrainRegistry`, evolution, tick loop |
| `scriptbots-core` | `tests/world_integration.rs` | World simulation integration tests |
| `scriptbots-core` | `benches/world_bench.rs` | Tick performance benchmarks |
| `scriptbots-brain` | `src/lib.rs` | `Brain` trait, `BrainKind`, `BrainTelemetry` |
| `scriptbots-brain` | `src/mlp.rs` | `MlpBrain` — multi-layer perceptron implementation |
| `scriptbots-brain` | `src/dwraon.rs` | `DwraonBrain` — DWRAON network implementation |
| `scriptbots-brain` | `src/assembly.rs` | `AssemblyBrain` — assembly-style brain with instruction set |
| `scriptbots-brain-ml` | `src/lib.rs` | ML feature selection and current sensor-copy placeholder |
| `scriptbots-brain-neuro` | `src/lib.rs` | Neuroflow neural network brain adapter |
| `scriptbots-index` | `src/lib.rs` | `NeighborhoodIndex` trait and `UniformGridIndex`; alternate backends remain open |
| `scriptbots-storage` | `src/lib.rs` | `Storage`, `StoragePipeline`, FrankenSQLite schema, metric/replay persistence, immutable analytics snapshots |
| `scriptbots-render` | `src/lib.rs` | GPUI rendering, camera system, world visualization, agent drawing |
| `scriptbots-world-gfx` | `src/lib.rs` | wgpu pipeline, WGSL shaders, offscreen readback for GPUI composition |
| `scriptbots-bevy` | `src/lib.rs` | Bevy ECS plugin, entity management, system scheduling |
| `scriptbots-app` | `src/main.rs` | CLI entry point, mode dispatch (GUI/TUI/headless/server) |
| `scriptbots-app` | `src/servers.rs` | Axum HTTP API, MCP server, Swagger/OpenAPI |
| `scriptbots-app` | `src/control.rs` | Simulation control commands, config management |
| `scriptbots-app` | `src/terminal/` | Ratatui TUI implementation |
| `scriptbots-web` | `src/lib.rs` | WASM bindings, browser-side simulation interface |

### Core Types Quick Reference

| Type | Purpose |
|------|---------|
| `WorldState` | Central simulation state — agents, food, terrain, hydrology, tick loop |
| `AgentData` | Per-agent state: position, velocity, health, energy, genome, brain binding |
| `AgentArena` | `SlotMap<AgentId, AgentData>` generational arena for all agents |
| `AgentId` | Stable generational handle for agents (`slotmap::new_key_type!`) |
| `Brain` | Core trait — `tick(inputs) -> outputs`, `mutate()`, `crossover()`, `snapshot_activations()` |
| `BrainRunner` | Batch brain evaluation trait for the tick loop |
| `BrainRegistry` | Registry of brain families and their factories |
| `BrainGenome` | Serializable genome with layer specs, hyperparams, provenance |
| `ScriptBotsConfig` | All simulation tuning knobs (mutation rates, food, terrain, rendering) |
| `FoodGrid` | Spatial grid of food cells with growth/decay dynamics |
| `TerrainLayer` | Terrain types (land, water, hazard) with procedural generation |
| `Storage` | Same-thread FrankenSQLite persistence boundary; owns the connection and typed SQL conversions |
| `StoragePipeline` | Bounded worker with a durable file outbox, per-batch admission identities, monotonic admitted/applied/durable watermarks, recovery, and flush/shutdown receipts |
| `AnalyticsSnapshot` | Immutable latest-value read model published lock-free to GUI, TUI, and API consumers |
| `NeighborhoodIndex` | Trait currently implemented by the uniform-grid spatial index |
| `Tick` | Newtype wrapper for simulation time step (`u64`) |
| `ControlCommand` | Enum of simulation control actions |
| `MutationRates` | Per-genome mutation rate parameters |
| `DeathCause` | Enum: starvation, old age, combat, etc. |
| `SelectionMode` | Evolution selection strategy enum |

### Console Output Style

We want all console output to be informative, detailed, stylish, colorful, etc. by fully leveraging the relevant Rust libraries (`owo-colors`, `ratatui`, `supports-color`) wherever possible.

### Key Design Decisions

- **Pluggable brain architecture** — `Brain` trait allows MLP, DWRAON, Assembly, ML, and Neuroflow backends to coexist and compete
- **Generational slot map (`slotmap`)** for agent handles — O(1) lookup, safe reuse, no dangling references
- **FrankenSQLite for persistence and analytics** — one SQLite-compatible run database, isolated behind a bounded worker and immutable read models
- **Multiple rendering backends** — GPUI (native), Bevy (ECS), wgpu (custom), Ratatui (terminal), WASM (browser)
- **Rayon for data parallelism** — agent tick processing parallelized with configurable thread budgets
- **SIMD via `wide`** — vectorized math for simulation hot paths
- **Feature-gated backends** — ML/Neuro/GUI/Bevy/audio are all optional features to minimize compile times
- **WASM target** — `scriptbots-web` compiles to WebAssembly for browser deployment via `wasm-pack`
- **MCP server integration** — agents can interact with the simulation via MCP protocol
- **Workspace-level lint config** — clippy pedantic + nursery enabled, consistent across all crates

### FrankenSQLite Storage Contract

- **One engine:** use the public `fsqlite` facade at package version `0.1.16`, pinned to immutable revision `e536d7f8ca102b3eb5236bef48514582379f9346` from `https://github.com/Dicklesworthstone/frankensqlite`. The workspace declaration uses `version = "=0.1.16"`, `default-features = false`, and `features = ["native"]` until the lean native feature qualification is complete.
- **Thread and writer ownership:** `fsqlite::Connection` is deliberately `!Send + !Sync`. Construct, use, explicitly close, and drop it inside the storage worker thread. Never place a connection or connection-owning `Storage` inside cross-thread `Arc<Mutex<_>>` state. Every file writer holds a nonblocking OS advisory lease on a stable companion lock file for the connection-owning lifetime; the process-local path/inode registry is defense in depth, not the cross-process authority. Recovery holds and revalidates an existing-file descriptor, binds FrankenSQLite's writable open to that expected descriptor identity before any header or journal access, and verifies the exact supported migration set and structural schema before enabling writes.
- **Bounded admission and explicit proof:** `StoragePipeline` carries a bounded persistence queue, and configurable `StorageDeadlines` bound controller waits for startup, gate acquisition, enqueue, admission, flush, and shutdown acknowledgements. They cannot cancel a FrankenSQLite call already executing on the owner thread, and supervised reaping may wait for that call to finish. Validation, closed-gate, queue-send, and rolled-back outbox failures are definitely `NotAdmitted`; the world latches the fault, retains the exact completed batch, and prevents later science ticks until an explicit retry admits that batch. A lost or timed-out acknowledgement remains typed as `Indeterminate` at the world boundary, and the retained exact payload is retry-safe because an identical tick/BLAKE3 identity reuses its stable batch ID while a changed payload is rejected. Timed-out shutdown keeps its original receipt receiver and worker handle for retry; controller drop transfers them to the supervised reaper without moving the connection. `submit_with_receipt` assigns the batch identity only after the exact payload commits to the worker outbox (`Durable` for `file`, `CommittedVolatile` for `memory`). Admission is still not application: the worker atomically applies contiguous outbox batches to the scientific tables, advances the applied watermark, then advances the separate durable watermark and compacts only payloads covered by that marker. Startup replays admitted-but-unapplied batches in order and idempotently finalizes applied-but-not-durable batches. Flush and shutdown receipts expose all three watermarks; readers and immutable analytics snapshots can query them without touching the worker connection. Same-thread `Storage::persist` uses this identical identity/stage/apply/finalize protocol, while its raw agent-insert SQL stays private and the isolated FrankenSQLite workload test owns its own explicitly non-production SQL. Terminal worker errors share the exact immutable typed FrankenSQLite cause across the first receipt and later shutdown/join (`bd-2z0.8.9.4.4`).
- **Lock-free reads:** the worker atomically publishes immutable `Arc<AnalyticsSnapshot>` latest values. GUI, TUI, and API consumers load them without a mutex; rendering and paint paths never acquire a database lock or issue SQL.
- **Modes and files:** the application storage targets are `file` and `memory`. `file` exclusively reserves `SCRIPTBOTS_STORAGE_PATH` or a unique `runs/scriptbots-<unix-ms>-<pid>.sqlite`; startup refuses an existing database or stale SQLite sidecar instead of reusing a prior run. `memory` opens `:memory:` through the same FrankenSQLite implementation. The app prints the selected file path; use that exact path for later reads and exports. `--recover-storage FILE` (or `SCRIPTBOTS_RECOVER_STORAGE`) is the explicit exception for opening an existing file: it holds the OS writer lease, binds the writable FrankenSQLite handle to the leased file identity before recovery, verifies the exact supported migration and structural schema manifests plus persistence invariants, refuses aliases/unrelated files, replays/finalizes persistence, and exits without claiming to reconstruct or resume the in-memory world.
- **Maintenance:** same-thread `Storage::optimize` flushes before `VACUUM`, and explicit close handles checkpointing on the connection-owning thread. The asynchronous pipeline currently exposes only flush and shutdown barriers. `PRAGMA integrity_check` is a conformance-test gate, not a runtime maintenance command. Never run database maintenance in a UI path or claim unsupported pragmas performed it.

---

## MCP Agent Mail — Multi-Agent Coordination

A mail-like layer that lets coding agents coordinate asynchronously via MCP tools and resources. Provides identities, inbox/outbox, searchable threads, and advisory file reservations with human-auditable artifacts in Git.

### Why It's Useful

- **Prevents conflicts:** Explicit file reservations (leases) for files/globs
- **Token-efficient:** Messages stored in per-project archive, not in context
- **Quick reads:** `resource://inbox/...`, `resource://thread/...`

### Same Repository Workflow

1. **Register identity:**
   ```
   ensure_project(project_key=<abs-path>)
   register_agent(project_key, program, model)
   ```

2. **Reserve files before editing:**
   ```
   file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true)
   ```

3. **Communicate with threads:**
   ```
   send_message(..., thread_id="FEAT-123")
   fetch_inbox(project_key, agent_name)
   acknowledge_message(project_key, agent_name, message_id)
   ```

4. **Quick reads:**
   ```
   resource://inbox/{Agent}?project=<abs-path>&limit=20
   resource://thread/{id}?project=<abs-path>&include_bodies=true
   ```

### Macros vs Granular Tools

- **Prefer macros for speed:** `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`
- **Use granular tools for control:** `register_agent`, `file_reservation_paths`, `send_message`, `fetch_inbox`, `acknowledge_message`

### Common Pitfalls

- `"from_agent not registered"`: Always `register_agent` in the correct `project_key` first
- `"FILE_RESERVATION_CONFLICT"`: Adjust patterns, wait for expiry, or use non-exclusive reservation
- **Auth errors:** If JWT+JWKS enabled, include bearer token with matching `kid`

---

## Beads (br) — Dependency-Aware Issue Tracking

Beads provides a lightweight, dependency-aware issue database and CLI (`br` - beads_rust) for selecting "ready work," setting priorities, and tracking status. It complements MCP Agent Mail's messaging and file reservations.

**Important:** `br` is non-invasive—it NEVER runs git commands automatically. You must manually commit changes after `br sync --flush-only`.

### Conventions

- **Single source of truth:** Beads for task status/priority/dependencies; Agent Mail for conversation and audit
- **Shared identifiers:** Use Beads issue ID (e.g., `br-123`) as Mail `thread_id` and prefix subjects with `[br-123]`
- **Reservations:** When starting a task, call `file_reservation_paths()` with the issue ID in `reason`

### Typical Agent Flow

1. **Pick ready work (Beads):**
   ```bash
   br ready --json  # Choose highest priority, no blockers
   ```

2. **Reserve edit surface (Mail):**
   ```
   file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true, reason="br-123")
   ```

3. **Announce start (Mail):**
   ```
   send_message(..., thread_id="br-123", subject="[br-123] Start: <title>", ack_required=true)
   ```

4. **Work and update:** Reply in-thread with progress

5. **Complete and release:**
   ```bash
   br close 123 --reason "Completed"
   br sync --flush-only  # Export to JSONL (no git operations)
   ```
   ```
   release_file_reservations(project_key, agent_name, paths=["src/**"])
   ```
   Final Mail reply: `[br-123] Completed` with summary

### Mapping Cheat Sheet

| Concept | Value |
|---------|-------|
| Mail `thread_id` | `br-###` |
| Mail subject | `[br-###] ...` |
| File reservation `reason` | `br-###` |
| Commit messages | Include `br-###` for traceability |

---

## bv — Graph-Aware Triage Engine

bv is a graph-aware triage engine for Beads projects. In this repository,
`.beads/issues.jsonl` is the authoritative tracked export; the similarly named
`.beads/beads.jsonl` is not an authority. Always invoke BV through
`scripts/bv_authoritative.sh`, which builds an isolated read-only view and
cross-checks BR/BV issue, status, dependency, native actionability, BR readiness,
and hash state before emitting JSON.

**Scope boundary:** bv handles *what to work on* (triage, priority, planning). For agent-to-agent coordination (messaging, work claiming, file reservations), use MCP Agent Mail.

**CRITICAL: Use ONLY `scripts/bv_authoritative.sh --robot-*`. Bare `bv` launches an interactive TUI, and direct robot invocations can silently select a stale alternate snapshot.**

### The Workflow: Start With Triage

**`scripts/bv_authoritative.sh --robot-triage` is your single entry point.** It returns:
- `quick_ref`: at-a-glance native BV counts and up to 3 claim-safe top picks
- `recommendations`: graph-ranked issues with scores, reasons, unblock info (including blocked work)
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
scripts/bv_authoritative.sh --robot-triage  # THE MEGA-COMMAND: start here
br ready --json                             # The sole actionability and claim authority
```

BV 0.16.0 and current BR deliberately disagree about actionability: BV includes
in-progress work and treats hierarchy differently, while `br ready` returns
open, unblocked, non-deferred claims. The wrapper proves BV's own actionable
count and complete plan agree, proves every BR-ready issue is represented in
that plan, and then refuses a claim-oriented result unless `--robot-next`, an
unscoped plan, a scoped plan, or a triage top pick is respectively BR-ready,
exactly BR-ready, a BR-ready subset, or BR-ready. An empty BV top-pick list is
valid graph-analysis output, not evidence that the BR queue is empty. A BV
recommendation is graph analysis, not authorization to claim work.

### Command Reference

**Planning:**
| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks only when the emitted set agrees with BR readiness; otherwise fails closed |
| `--robot-priority` | Priority misalignment detection with confidence |

**Graph Analysis:**
| Command | Returns |
|---------|---------|
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core, articulation points, slack |
| `--robot-label-health` | Per-label health: `health_level`, `velocity_score`, `staleness`, `blocked_count` |
| `--robot-label-flow` | Cross-label dependency: `flow_matrix`, `dependencies`, `bottleneck_labels` |
| `--robot-label-attention [--attention-limit=N]` | Attention-ranked labels |

**History and change tracking:** `--robot-history` and `--robot-diff` are
refused because BV 0.16.0 resolves their historical JSONL from the checkout
instead of honoring the wrapper's isolated source. Its preferred historical
filename can therefore mix stale data with an authoritative report hash.

**Other:**
| Command | Returns |
|---------|---------|
| `--robot-burndown <sprint>` | Sprint burndown, scope changes, at-risk items |
| `--robot-forecast <id\|all>` | ETA predictions with dependency-aware scheduling |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions |
| `--robot-graph --graph-format=json` | Dependency graph JSON; non-JSON graph output is refused by the wrapper |

### Scoping & Filtering

```bash
scripts/bv_authoritative.sh --robot-plan --label backend            # Scope; still refuses non-BR-ready items
scripts/bv_authoritative.sh --robot-insights --label backend        # Inspect one label's subgraph
scripts/bv_authoritative.sh --recipe actionable --robot-plan        # BV filter; BR-ready subset gate still applies
scripts/bv_authoritative.sh --recipe high-impact --robot-triage     # Pre-filter: top PageRank
scripts/bv_authoritative.sh --robot-triage --robot-triage-by-track  # Group by parallel work streams
scripts/bv_authoritative.sh --robot-triage --robot-triage-by-label  # Group by domain
```

### Understanding Robot Output

**All robot JSON includes:**
- `data_hash` — Fingerprint of the isolated authoritative `issues.jsonl` view
- `status` — Per-metric state: `computed|approx|timeout|skipped` + elapsed ms

Historical `--as-of` analysis is deliberately refused by this current-authority
wrapper. Use a separately audited historical snapshot workflow when that is the
actual task.

**Two-phase analysis:**
- **Phase 1 (instant):** degree, topo sort, density
- **Phase 2 (async, 500ms timeout):** PageRank, betweenness, HITS, eigenvector, cycles

### jq Quick Reference

```bash
scripts/bv_authoritative.sh --robot-triage | jq '.triage.quick_ref'           # At-a-glance summary
scripts/bv_authoritative.sh --robot-triage | jq '.triage.recommendations[0]'  # Top recommendation
scripts/bv_authoritative.sh --robot-plan | jq '.plan.summary.highest_impact'  # Best BR-aligned target, or fail closed
scripts/bv_authoritative.sh --robot-insights | jq '.status'                   # Check metric readiness
scripts/bv_authoritative.sh --robot-insights | jq '.Cycles'                   # Circular deps (must fix!)
```

---

## UBS — Ultimate Bug Scanner

**Golden Rule:** `ubs <changed-files>` before every commit. Exit 0 = safe. Exit >0 = fix & re-run.

### Commands

```bash
ubs file.rs file2.rs                    # Specific files (< 1s) — USE THIS
ubs $(git diff --name-only --cached)    # Staged files — before commit
ubs --only=rust,toml src/               # Language filter (3-5x faster)
ubs --ci --fail-on-warning .            # CI mode — before PR
ubs .                                   # Whole project (ignores target/, Cargo.lock)
```

### Output Format

```
  Category (N errors)
    file.rs:42:5 - Issue description
    Suggested fix
Exit code: 1
```

Parse: `file:line:col` -> location | fix hint -> how to fix | Exit 0/1 -> pass/fail

### Fix Workflow

1. Read finding -> category + fix suggestion
2. Navigate `file:line:col` -> view context
3. Verify real issue (not false positive)
4. Fix root cause (not symptom)
5. Re-run `ubs <file>` -> exit 0
6. Commit

### Bug Severity

- **Critical (always fix):** Memory safety, use-after-free, data races, SQL injection
- **Important (production):** Unwrap panics, resource leaks, overflow checks
- **Contextual (judgment):** TODO/FIXME, println! debugging

---

## RCH — Remote Compilation Helper

RCH offloads `cargo build`, `cargo test`, `cargo clippy`, and other compilation commands to a fleet of 8 remote Contabo VPS workers instead of building locally. This prevents compilation storms from overwhelming csd when many agents run simultaneously.

**RCH is installed at `~/.local/bin/rch` and is hooked into Claude Code's PreToolUse automatically.** Most of the time you don't need to do anything if you are Claude Code — builds are intercepted and offloaded transparently.

To manually offload a build:
```bash
rch exec -- cargo build --release
rch exec -- cargo test
rch exec -- cargo clippy
```

Quick commands:
```bash
rch doctor                    # Health check
rch workers probe --all       # Test connectivity to all 8 workers
rch status                    # Overview of current state
rch queue                     # See active/waiting builds
```

**RCH FAILS OPEN TO A LOCAL BUILD — TREAT EVERY `rch exec` AS CAPABLE OF BECOMING ONE (`bd-e6ff`).**
When the remote leg fails, rch logs `Remote execution failed: ... running locally` and runs the
build on this host. That is benign on a healthy machine and dangerous on this one: with
`CARGO_TARGET_DIR` on the exFAT volume and `UVFSService` wedged, a local `rustc` blocks in
uninterruptible (`U`) state, where `SIGKILL` does not apply — `timeout` and `pkill` are both no-ops
and every attempt leaks a permanent process. Agents have leaked wedged `rustc` while deliberately
running no local cargo, because rch ran it for them.

`force_remote = true` does NOT prevent this; the sync-failure path does not consult it. There is no
config-level guard. Mitigations that actually work:

- Read the `Executing command remotely: ... on <worker>` line and the sync result before believing
  any outcome. A permission/rsync failure is an infrastructure result, not a test result.
- `rch workers drain <id> --yes` any worker that returns `mkdir: Permission denied` or rsync code 23
  (reverse with `rch workers enable <id>`). Fewer bad dispatches means fewer silent local fallbacks.
- `rch workers probe --all` proves SSH reachability only. It does not write to the project tree,
  so a worker that fails every job in 1.6s still probes green — "6/6 healthy" is not evidence.
- Never pipe `rch exec` into `tail`/`head`: you get the pipe's exit code, not the build's. Redirect
  to a file, then check `$?`.

**Note for Codex/GPT-5.2:** Codex does not have the automatic PreToolUse hook, but you can (and should) still manually offload compute-intensive compilation commands using `rch exec -- <command>`. This avoids local resource contention when multiple agents are building simultaneously.

---

## ast-grep vs ripgrep

**Use `ast-grep` when structure matters.** It parses code and matches AST nodes, ignoring comments/strings, and can **safely rewrite** code.

- Refactors/codemods: rename APIs, change import forms
- Policy checks: enforce patterns across a repo
- Editor/automation: LSP mode, `--json` output

**Use `ripgrep` when text is enough.** Fastest way to grep literals/regex.

- Recon: find strings, TODOs, log lines, config values
- Pre-filter: narrow candidate files before ast-grep

### Rule of Thumb

- Need correctness or **applying changes** -> `ast-grep`
- Need raw speed or **hunting text** -> `rg`
- Often combine: `rg` to shortlist files, then `ast-grep` to match/modify

### Rust Examples

```bash
# Find structured code (ignores comments)
ast-grep run -l Rust -p 'fn $NAME($$$ARGS) -> $RET { $$$BODY }'

# Find all unwrap() calls
ast-grep run -l Rust -p '$EXPR.unwrap()'

# Quick textual hunt
rg -n 'println!' -t rust

# Combine speed + precision
rg -l -t rust 'unwrap\(' | xargs ast-grep run -l Rust -p '$X.unwrap()' --json
```

---

## Morph Warp Grep — AI-Powered Code Search

**Use `mcp__morph-mcp__warp_grep` for exploratory "how does X work?" questions.** An AI agent expands your query, greps the codebase, reads relevant files, and returns precise line ranges with full context.

**Use `ripgrep` for targeted searches.** When you know exactly what you're looking for.

**Use `ast-grep` for structural patterns.** When you need AST precision for matching/rewriting.

### When to Use What

| Scenario | Tool | Why |
|----------|------|-----|
| "How is the neural network implemented?" | `warp_grep` | Exploratory; don't know where to start |
| "Where is the GPUI rendering loop?" | `warp_grep` | Need to understand architecture |
| "Find all uses of `spawn`" | `ripgrep` | Targeted literal search |
| "Find files with `println!`" | `ripgrep` | Simple pattern |
| "Replace all `unwrap()` with `expect()`" | `ast-grep` | Structural refactor |

### warp_grep Usage

```
mcp__morph-mcp__warp_grep(
  repoPath: "/data/projects/rust_scriptbots",
  query: "How does the bot brain neural network work?"
)
```

Returns structured results with file paths, line ranges, and extracted code snippets.

### Anti-Patterns

- **Don't** use `warp_grep` to find a specific function name -> use `ripgrep`
- **Don't** use `ripgrep` to understand "how does X work" -> wastes time with manual reads
- **Don't** use `ripgrep` for codemods -> risks collateral edits

---

## cass — Cross-Agent Session Search

`cass` indexes prior agent conversations (Claude Code, Codex, Cursor, Gemini, ChatGPT, Aider, etc.) into a unified, searchable index so you can reuse solved problems.

**NEVER run bare `cass`** — it launches an interactive TUI. Always use `--robot` or `--json`.

### Quick Start

```bash
# Check if index is healthy (exit 0=ok, 1=run index first)
cass health

# Search across all agent histories
cass search "GPUI rendering" --robot --limit 5

# View a specific result (from search output)
cass view /path/to/session.jsonl -n 42 --json

# Expand context around a line
cass expand /path/to/session.jsonl -n 42 -C 3 --json

# Learn the full API
cass capabilities --json      # Feature discovery
cass robot-docs guide         # LLM-optimized docs
```

### Key Flags

| Flag | Purpose |
|------|---------|
| `--robot` / `--json` | Machine-readable JSON output (required!) |
| `--fields minimal` | Reduce payload: `source_path`, `line_number`, `agent` only |
| `--limit N` | Cap result count |
| `--agent NAME` | Filter to specific agent (claude, codex, cursor, etc.) |
| `--days N` | Limit to recent N days |

**stdout = data only, stderr = diagnostics. Exit 0 = success.**

### Robot Mode Etiquette

- Prefer `cass --robot-help` and `cass robot-docs <topic>` for machine-first docs
- The CLI is forgiving: globals placed before/after subcommand are auto-normalized
- If parsing fails, follow the actionable errors with examples
- Use `--color=never` in non-TTY automation for ANSI-free output

### Pre-Flight Health Check

```bash
cass health --json
```

Returns in <50ms:
- **Exit 0:** Healthy—proceed with queries
- **Exit 1:** Unhealthy—run `cass index --full` first

### Exit Codes

| Code | Meaning | Retryable |
|------|---------|-----------|
| 0 | Success | N/A |
| 1 | Health check failed | Yes—run `cass index --full` |
| 2 | Usage/parsing error | No—fix syntax |
| 3 | Index/DB missing | Yes—run `cass index --full` |

Treat cass as a way to avoid re-solving problems other agents already handled.

<!-- bv-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking. Issues are stored in `.beads/` and tracked in git.

**Important:** `br` is non-invasive—it NEVER executes git commands. After
`br sync --flush-only`, commit the exported paths through the reviewed-snapshot
protocol below.

### Essential Commands

```bash
# Graph-aware robot triage over the authoritative tracked export
scripts/bv_authoritative.sh --robot-triage

# CLI commands for agents (use these instead)
br ready              # Show issues ready to work (no blockers)
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br create --title="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason "Completed"
br close <id1> <id2>  # Close multiple issues at once
br sync --flush-only  # Export to JSONL (NO git operations)
```

### Workflow Pattern

1. **Start**: Run `br ready` to find actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Run `br sync --flush-only` then manually commit

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers, not words)
- **Types**: task, bug, feature, epic, question, docs
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status

# Freeze the exact code snapshot, read EVERY printed hunk, then copy its token.
AGENT_NAME=<mail-name> scripts/shared_tree_commit.py review \
  -m "fix(crate): what and why (bd-xxxx)" -- <exact-code-files>
AGENT_NAME=<mail-name> scripts/shared_tree_commit.py commit --review <token>

# Export and commit tracker state as a separate reviewed snapshot.
br sync --flush-only
AGENT_NAME=<mail-name> scripts/shared_tree_commit.py review \
  -m "chore(beads): close bd-xxxx" -- .beads/issues.jsonl .beads/last-touched
AGENT_NAME=<mail-name> scripts/shared_tree_commit.py commit --review <token>

# AGENT_NAME is required for the PUSH too, not just the commits above (bd-tat5).
AGENT_NAME=<mail-name> git push origin main
AGENT_NAME=<mail-name> git push origin main:master
```

**`git push` without `AGENT_NAME` fails, and the error does not say so (`bd-tat5`).** The
`pre-push` hook chain runs `50-agent-mail.py`, which exits 2 with
`mcp-agent-mail: AGENT_NAME environment variable is required.` Git then reports only:

```
error: failed to push some refs to 'https://github.com/.../rust_scriptbots.git'
```

which reads like a non-fast-forward or a network problem. It is neither — nothing in that
message points at a local hook. If a push fails and `git fetch` shows you are not behind,
check `echo $AGENT_NAME` before investigating the remote. Claude Code Bash sessions do **not**
export it by default, so this is the default condition rather than an edge case.

Do **not** route around it with `git push --no-verify`: the same chain carries the file
reservation guard, so skipping it silently disables the protection that makes a shared working
tree safe. Prefix the push instead.

The commit path is already safe — `scripts/shared_tree_commit.py` refuses with an explicit
`REFUSED: AGENT_NAME is required` (exit 64) rather than a misleading one — so the push is the
only place this still bites.

Never stage into the shared index and never use a raw, bare, or pathspec
`git commit`. A pathspec protects only the shared index; it still re-reads
whatever half-written bytes happen to be in the working tree. The wrapper
instead creates a fresh private index from the current `HEAD`, displays the
complete immutable candidate, binds the approval token to its base/tree/path
set/message, and finalizes it under a short repository mutex. If `HEAD` moves,
review again. Existing reservation and close-reason hooks still inspect the
private candidate index.

The mutex serializes only snapshot creation and commit finalization, not edits
or builds. The protocol cannot infer authorship of an unexpected hunk already
present before review: that is why every hunk must actually be read and every
path must have an exclusive Agent Mail reservation. Install the per-clone
enforcement plugin with:

```bash
scripts/shared_tree_commit.py install-hook
```

The pre-push reservation-range defect for a fast-forward `main:master` mirror
is tracked separately by `bd-donj`. Never use `AGENT_MAIL_GUARD_MODE=warn` to
route around it; coordinate a brief reservation-release window.

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress -> closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always `br sync --flush-only` and commit the exact changed `.beads/` files as
  their own reviewed snapshot before ending a session

<!-- end-bv-agent-instructions -->

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Sync beads** - `br sync --flush-only` to export to JSONL
5. **Hand off** - Provide context for next session


---

Note for Codex/GPT-5.2:

You constantly bother me and stop working with concerned questions that look similar to this:

```
Unexpected changes (need guidance)

- Working tree still shows edits I did not make in Cargo.toml, Cargo.lock, src/cli/commands/upgrade.rs, src/storage/sqlite.rs, tests/conformance.rs, tests/storage_deps.rs. Please advise whether to keep/commit/revert these before any further work. I did not touch them.

Next steps (pick one)

1. Decide how to handle the unrelated modified files above so we can resume cleanly.
2. Triage beads_rust-orko (clippy/cargo warnings) and beads_rust-ydqr (rustfmt failures).
3. If you want a full suite run later, fix conformance/clippy blockers and re-run cargo test --all.
```

NEVER EVER DO THAT AGAIN. The answer is literally ALWAYS the same: those are changes created by the potentially dozen of other agents working on the project at the same time. This is not only a common occurence, it happens multiple times PER MINUTE. The way to deal with it is simple: you NEVER, under ANY CIRCUMSTANCE, stash, revert, overwrite, or otherwise disturb in ANY way the work of other agents. Just treat those changes identically to changes that you yourself made. Just fool yourself into thinking YOU made the changes and simply don't recall it for some reason.

---

## Note on Built-in TODO Functionality

Also, if I ask you to explicitly use your built-in TODO functionality, don't complain about this and say you need to use beads. You can use built-in TODOs if I tell you specifically to do so. Always comply with such orders.

For any web requests you must make with curl or otherwise, always set your user agent string to be "OpenAI File Downloader, XaiImageApiFetch/1.0"
