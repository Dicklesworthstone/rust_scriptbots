## Rust ScriptBots

ScriptBots is a modern Rust reimagining of Andrej Karpathy’s classic agent-based evolution simulator. Our objective is to deliver a faithful, deterministic port with a GPU-accelerated UI, pluggable brain implementations, and first-class analytics. This repository is a multi-crate Cargo workspace that separates the simulation core, brains, storage, rendering, and application shell.

For full design intent and roadmap, see `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md` (the project “bible”). We target Linux, macOS, and Windows 11 (native and WSL2), with ongoing validation of UI performance across all three.

### Why this exists
- **Determinism and safety**: Replace legacy C++/GLUT and global state with idiomatic Rust, zero `unsafe` in v1, and reproducible runs.
- **Performance at scale**: Use data-parallelism (Rayon) and cache-friendly data layouts to simulate thousands of agents efficiently.
- **Modern UX**: Swap immediate-mode OpenGL for a declarative, GPU-accelerated GPUI interface with an inspector, overlays, and smooth camera controls.
- **Observability**: Persist metrics and snapshots to DuckDB for replay, analytics, and regression testing.
- **Extensibility**: Hot-swap brain implementations (MLP, DWRAON, experimental Assembly, plus optional NeuroFlow) without rewriting the world loop.

## Architecture at a glance

The workspace is organized for clear boundaries and fast incremental builds:

```
rust_scriptbots/
├── Cargo.toml                # Workspace manifest, shared deps/lints/profiles
├── rust-toolchain.toml       # Pinned toolchain (Rust 1.85)
├── crates/
│   ├── scriptbots-core       # Core types and (future) simulation model/traits
│   ├── scriptbots-brain      # Brain trait + telemetry; concrete brains later
│   ├── scriptbots-brain-ml   # Optional ML backends (Candle, Tract, tch) via features
│   ├── scriptbots-index      # Pluggable spatial indices (grid, rstar, kd-tree)
│   ├── scriptbots-storage    # DuckDB-backed persistence & analytics hooks
│   ├── scriptbots-render     # GPUI integration and visual layer
│   └── scriptbots-app        # Binary crate wiring everything together
└── original_scriptbots_code_for_reference/  # Upstream C++ snapshot for parity
```

### Crate roles
- **`scriptbots-core`**: Foundational types (e.g., `Tick`, `AgentId`). Will house `WorldState`, `AgentState`, configuration, and the deterministic tick pipeline.
- **`scriptbots-brain`**: Defines the `Brain` trait and telemetry. Will host concrete brains: `MlpBrain`, `DwraonBrain`, and experimental `AssemblyBrain` behind features.
- **`scriptbots-brain-ml`**: Optional ML backends (Candle, Tract, tch) for alternative or accelerated brain implementations, all feature-gated.
- **`scriptbots-index`**: Spatial indexing implementations. Default uniform grid with optional `rstar` (R-tree) and `kd` (kiddo) features.
- **`scriptbots-storage`**: A thin persistence layer over DuckDB. Currently exposes `Storage::open` and `record_tick`; will evolve to buffered/batched writers and richer schemas.
- **`scriptbots-render`**: GPUI application scaffolding. Today it boots a minimal `Application`; planned work includes a `WorldView`, canvas-based rendering, overlays, and an inspector.
- **`scriptbots-app`**: The binary shell. Initializes tracing/logging, config, and orchestrates the simulation/render loop.

## Current status (alpha scaffold)
- Workspace, profiles, and shared lint settings are in place.
- `scriptbots-app` starts and initializes tracing; calls a minimal GPUI demo.
- `scriptbots-render` spins up a bare `gpui::Application` (no views/windows yet).
- `scriptbots-core` provides initial types; `WorldState`/`AgentState` and the tick loop are not yet implemented.
- `scriptbots-brain` exposes the `Brain` trait and telemetry; no concrete brains merged yet.
- `scriptbots-storage` opens DuckDB and records simple per-tick metrics.

Refer to the migration roadmap in `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md` for staged milestones (core data structures → world mechanics → concurrency → brains → storage → rendering → integration & polish).

## Getting started

### Prerequisites
- Rust toolchain: pinned in `rust-toolchain.toml` (Rust 1.85). Install via `rustup`.
- OS: Linux, macOS, or Windows 11 (native or WSL2). GPU drivers should be up to date for best GPUI performance (wgpu backends: Metal/macOS, Vulkan/Linux, D3D12 or Vulkan/Windows).

### Build
```bash
cargo check
```

### Run the app shell
```bash
cargo run -p scriptbots-app
```

Set logging verbosity with `RUST_LOG`, for example:
```bash
RUST_LOG=info cargo run -p scriptbots-app
```

### Windows quickstart (native)
1. Install Rust (MSVC toolchain):
   - Download `rustup-init.exe` and select the MSVC target, or run in PowerShell:
   ```powershell
   rustup default stable-x86_64-pc-windows-msvc
   rustup component add clippy rustfmt
   ```
2. Install Visual Studio Build Tools (2022+):
   - Select the "Desktop development with C++" workload (includes MSVC, Windows 10/11 SDK).
3. Update GPU drivers (NVIDIA/AMD/Intel) to latest. Ensure D3D12 is available; Vulkan runtime optional.
4. Build and run:
   ```powershell
   cargo run -p scriptbots-app
   ```
5. Troubleshooting: If linking fails with MSVC or SDK errors, re-run the VS installer to include the Windows 11 SDK and C++ toolset (v143+).

### Windows via WSL2 (optional)
- Windows 11 with WSLg supports Linux GUI apps out of the box; GPUI rendering generally works, but performance may vary. If you see blank windows, update your GPU drivers and WSL kernel, then retry.

### Feature flags & variants (current crates)
- `scriptbots-render`:
  - `audio` → enables Kira-driven audio output in the render layer (app-level toggle pending wiring).
- `scriptbots-index` (pluggable spatial indices):
  - Default: `grid`
  - Optional: `rstar` (R-tree), `kd` (kiddo-based k-d tree)
  - Example: `cargo build -p scriptbots-index --features rstar`
- `scriptbots-brain-ml` (ML backends for brains):
  - `candle`, `tract`, `tch` (all optional; off by default)
  - Examples:
    - `cargo build -p scriptbots-brain-ml --features candle`
    - `cargo build -p scriptbots-brain-ml --features tract`
    - `cargo build -p scriptbots-brain-ml --features tch`

Note: App-level switches to choose specific index/brain backends will be added as integration lands. For now, features can be compiled per-crate while the wiring is implemented.

### Commands cheat sheet
```bash
# Build the whole workspace
cargo build --workspace

# Run the UI shell
cargo run -p scriptbots-app

# Lint and format
cargo clippy --workspace --all-targets --all-features
cargo fmt --all

# Run tests (as they land)
cargo test --workspace

# Build optional crates with features
cargo build -p scriptbots-index --features rstar
cargo build -p scriptbots-brain-ml --features candle
```

## How the simulation will work (high level)
The world executes a deterministic, staged tick pipeline:
1. Aging and periodic tasks
2. Food respawn and diffusion
3. Event flag resets
4. Sense neighbors via a spatial index (uniform grid)
5. Brain evaluation (`Brain::tick`) for each agent without per-tick allocations
6. Actuation (movement, colors, sharing, boosting) using double-buffered state
7. Food intake and sharing with deterministic reductions
8. Combat and death (queued, then committed)
9. Reproduction (crossover/mutation) applied in a stable order
10. Persistence hooks (metrics and snapshots batched to DuckDB)

Determinism is preserved by avoiding shared mutable state in parallel sections, gathering results in thread-local buffers, then committing in a defined order.

## Rendering & UX (planned)
- GPUI window with a canvas renderer for food tiles and agents (circles/spikes).
- Camera: right-drag pan, middle-drag zoom; keyboard shortcuts (Pause, Toggle Draw, Speed ±, etc.).
- Overlays: population charts, selection highlights, brain heatmaps, metrics HUD.
- Inspector: per-agent stats, genome/brain views, mutation history.
- Audio (via `kira`): optional event-driven cues (reproduction, kills, starvation) with channel toggles.

## Storage & analytics (planned)
- DuckDB schema for ticks, agents, events, metrics; compacted via batched writers.
- Deterministic replay tooling (future crate may be introduced) to branch and compare runs.
- Optional Parquet/Arrow exports for heavy analytics.

## Development workflow
- **Coding standards**: See `RUST_SYSTEM_PROGRAMMING_BEST_PRACTICES.md`. Embrace `Result`-based errors, clear traits, and avoid `unsafe`.
- **Linting**: `cargo clippy --workspace --all-targets --all-features -W clippy::all -W clippy::pedantic -W clippy::nursery`
- **Formatting**: `cargo fmt --all`
- **Tests**: `cargo test --workspace` (simulation and GPUI tests will be added as systems land)
- **Profiles**: Release uses LTO, single codegen unit, and abort-on-panic for optimal binaries.

## Contributing
- Keep changes scoped to the relevant crate; prefer improving existing files over creating siblings unless functionality is truly new.
- Add or update documentation where it helps future maintainers understand decisions and invariants.
- For larger tasks, reference and update `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md` inline to mark progress.

## Licensing
Licensed under `MIT OR Apache-2.0` (see workspace manifest).

## Credits
- Original ScriptBots by Andrej Karpathy (reference snapshot included under `original_scriptbots_code_for_reference/`).
- This Rust port is an independent, from-scratch implementation guided by parity goals and modern Rust/GPUI best practices.

## FAQ
- **Does it run a full simulation today?** Not yet. The scaffolding builds and runs; core simulation, UI, and brains are being implemented incrementally.
- **What platforms are supported?** Linux, macOS, and Windows 11 are targeted. Windows is supported natively (MSVC toolchain) and via WSL2. Early UI milestones may see platform-specific polish arriving at different times.
- **Where do I start hacking?** `scriptbots-core` for the world model; `scriptbots-render` for the GPUI view; `scriptbots-brain` for brain interfaces; `scriptbots-storage` for persistence.

## Troubleshooting
- **MSVC/SDK link errors on Windows**: Ensure VS Build Tools "Desktop development with C++" and Windows 11 SDK are installed. Then run `rustup default stable-x86_64-pc-windows-msvc`.
- **Blank or crashing window**: Update GPU drivers. On WSL2, update the WSL kernel and try again. Verify that your system supports D3D12 (Windows) or Vulkan/Metal (Linux/macOS).
- **DuckDB file lock errors**: Close any external tools accessing the DB file and retry. Prefer unique DB paths per run while developing.
- **Determinism regressions**: Ensure you haven't introduced unordered parallel reductions; stage results and apply in a stable commit phase.

## Roadmap (condensed)
1. Core data structures (`WorldState`, `AgentState`), config port, and unit tests.
2. World mechanics (food grid, sensing, reproduction, death cleanup).
3. Concurrency + spatial indices (uniform grid first; opt-in `rstar`/`kd`).
4. Brains: MLP baseline; optional ML backends in `scriptbots-brain-ml`.
5. Persistence layer expansions (DuckDB schema, batched writers, analytics helpers).
6. Rendering layer: GPUI window, canvas renderer, overlays, inspector, audio (feature-gated).
7. Integration, UX polish, determinism/benchmark suites, packaging.
