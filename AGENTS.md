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
| `fsqlite` (`=0.1.16`, rev `cd9990bb16291d8c7c247b75b47faae8d7701adb`) | FrankenSQLite persistence for simulation metrics, replay, and run artifacts |
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
| `mcp-protocol-sdk` | MCP server integration |
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
cargo test -p scriptbots-core
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
| `StoragePipeline` | Bounded batch-admission worker with explicit flush/shutdown commit receipts; durable per-batch watermarks remain open |
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

- **One engine:** use the public `fsqlite` facade at package version `0.1.16`, pinned to immutable revision `cd9990bb16291d8c7c247b75b47faae8d7701adb` from `https://github.com/Dicklesworthstone/frankensqlite`. The workspace declaration uses `version = "=0.1.16"`, `default-features = false`, and `features = ["native"]` until the lean native feature qualification is complete.
- **Thread ownership:** `fsqlite::Connection` is deliberately `!Send + !Sync`. Construct, use, explicitly close, and drop it inside the storage worker thread. Never place a connection or connection-owning `Storage` inside cross-thread `Arc<Mutex<_>>` state.
- **Bounded admission and explicit proof:** `StoragePipeline` carries a bounded persistence queue. A synchronous rejection is definitely `NotAdmitted`; the world latches the fault, retains the exact completed batch, and prevents later science ticks until an explicit retry admits that batch. Successful enqueue proves only admission, not commit or crash durability. Flush and shutdown receipts prove that all earlier admitted transactions committed (`CommittedVolatile` for `memory`, `Durable` for `file`). The durable outbox and per-batch applied/durable watermarks required for an end-to-end lossless claim remain open work; do not describe the current asynchronous path as lossless.
- **Lock-free reads:** the worker atomically publishes immutable `Arc<AnalyticsSnapshot>` latest values. GUI, TUI, and API consumers load them without a mutex; rendering and paint paths never acquire a database lock or issue SQL.
- **Modes and files:** the application storage targets are `file` and `memory`. `file` exclusively reserves `SCRIPTBOTS_STORAGE_PATH` or a unique `runs/scriptbots-<unix-ms>-<pid>.sqlite`; startup refuses an existing database or stale SQLite sidecar instead of reusing a prior run. `memory` opens `:memory:` through the same FrankenSQLite implementation. The app prints the selected file path; use that exact path for later reads and exports.
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

bv is a graph-aware triage engine for Beads projects (`.beads/beads.jsonl`). It computes PageRank, betweenness, critical path, cycles, HITS, eigenvector, and k-core metrics deterministically.

**Scope boundary:** bv handles *what to work on* (triage, priority, planning). For agent-to-agent coordination (messaging, work claiming, file reservations), use MCP Agent Mail.

**CRITICAL: Use ONLY `--robot-*` flags. Bare `bv` launches an interactive TUI that blocks your session.**

### The Workflow: Start With Triage

**`bv --robot-triage` is your single entry point.** It returns:
- `quick_ref`: at-a-glance counts + top 3 picks
- `recommendations`: ranked actionable items with scores, reasons, unblock info
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command
```

### Command Reference

**Planning:**
| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks with `unblocks` lists |
| `--robot-priority` | Priority misalignment detection with confidence |

**Graph Analysis:**
| Command | Returns |
|---------|---------|
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core, articulation points, slack |
| `--robot-label-health` | Per-label health: `health_level`, `velocity_score`, `staleness`, `blocked_count` |
| `--robot-label-flow` | Cross-label dependency: `flow_matrix`, `dependencies`, `bottleneck_labels` |
| `--robot-label-attention [--attention-limit=N]` | Attention-ranked labels |

**History & Change Tracking:**
| Command | Returns |
|---------|---------|
| `--robot-history` | Bead-to-commit correlations |
| `--robot-diff --diff-since <ref>` | Changes since ref: new/closed/modified issues, cycles |

**Other:**
| Command | Returns |
|---------|---------|
| `--robot-burndown <sprint>` | Sprint burndown, scope changes, at-risk items |
| `--robot-forecast <id\|all>` | ETA predictions with dependency-aware scheduling |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions |
| `--robot-graph [--graph-format=json\|dot\|mermaid]` | Dependency graph export |
| `--export-graph <file.html>` | Interactive HTML visualization |

### Scoping & Filtering

```bash
bv --robot-plan --label backend              # Scope to label's subgraph
bv --robot-insights --as-of HEAD~30          # Historical point-in-time
bv --recipe actionable --robot-plan          # Pre-filter: ready to work
bv --recipe high-impact --robot-triage       # Pre-filter: top PageRank
bv --robot-triage --robot-triage-by-track    # Group by parallel work streams
bv --robot-triage --robot-triage-by-label    # Group by domain
```

### Understanding Robot Output

**All robot JSON includes:**
- `data_hash` — Fingerprint of source beads.jsonl
- `status` — Per-metric state: `computed|approx|timeout|skipped` + elapsed ms
- `as_of` / `as_of_commit` — Present when using `--as-of`

**Two-phase analysis:**
- **Phase 1 (instant):** degree, topo sort, density
- **Phase 2 (async, 500ms timeout):** PageRank, betweenness, HITS, eigenvector, cycles

### jq Quick Reference

```bash
bv --robot-triage | jq '.quick_ref'                        # At-a-glance summary
bv --robot-triage | jq '.recommendations[0]'               # Top recommendation
bv --robot-plan | jq '.plan.summary.highest_impact'        # Best unblock target
bv --robot-insights | jq '.status'                         # Check metric readiness
bv --robot-insights | jq '.Cycles'                         # Circular deps (must fix!)
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

If rch or its workers are unavailable, it fails open — builds run locally as normal.

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

**Important:** `br` is non-invasive—it NEVER executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

### Essential Commands

```bash
# View issues (launches TUI - avoid in automated sessions)
bv

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
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads to JSONL
git add .beads/         # Stage beads changes
git commit -m "..."     # Commit everything together
git push                # Push to remote
```

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress -> closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always `br sync --flush-only && git add .beads/` before ending session

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
