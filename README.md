## Rust ScriptBots

ScriptBots is a modern Rust reimagining of Andrej Karpathy’s classic agent-based evolution simulator. Our goal is a faithful, deterministic port with a GPU-accelerated UI, pluggable brain implementations, and first-class analytics. This is a multi-crate Cargo workspace separating simulation core, brains, storage, rendering, and the application shell.

For design intent and the living roadmap, see `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md` (the project “bible”). A sibling WebAssembly plan lives in `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md`.

### Philosophy & purpose
- **Why this exists**: ScriptBots is a minimalist artificial life laboratory. By rebuilding the original simulator with rigorously deterministic Rust systems, we can observe, measure, and reproduce emergent behavior at scale—without undefined behavior or global state muddying results.
- **What we learn**: How simple sensory channels and local rules produce complex population dynamics—cooperation vs. predation, resource gradients shaping migration, lineage divergence under different mutation schedules, and the role of perception in survival.
- **LLM-in-the-loop science**: The REST API, CLI, and MCP HTTP server expose the full control surface (knobs, patches, snapshots). This lets an external LLM agent act as an autonomous lab assistant: steering experiments, sweeping parameter spaces, logging observations into DuckDB, and drafting human-readable reports.
  - Example workflows:
    - Parameter sweeps: vary `mutation.{primary,secondary}` and temperature gradients; record birth/death ratios and equilibrium populations.
    - Interventions: toggle `closed` worlds, inject carnivore cohorts, or freeze food diffusion to test resilience.
    - Reporting: ingest DuckDB tables to auto-generate charts/tables describing discovered phenomena (e.g., altruistic giving thresholds that stabilize mixed diets).
- **A brain testbed**: The `Brain` trait and registry allow swapping decision engines—handwritten controllers, MLP/DWRAON/Assembly, or NeuroFlow—while holding the environment constant. This enables fair comparisons of:
  - Perception encoding (multi-eye vision, smell/sound/blood) and how architectures exploit them.
  - Locomotion control (differential drive) and energy/health trade-offs.
  - Evolutionary operators (mutation/crossover) and speciation pressures.
- **Reproducible research**: Deterministic pipelines + a growing replay roadmap mean results can be shared and re-run bit-for-bit, making the project a solid platform for pedagogy, papers, and benchmarking new brain designs.

### Why this exists
- **Determinism and safety**: Replace legacy C++/GLUT and global state with idiomatic Rust, zero `unsafe` in v1, and reproducible runs.
- **Performance at scale**: Data-parallelism (Rayon) and cache-friendly layouts to simulate thousands of agents efficiently.
- **Modern UX**: Declarative, GPU-accelerated GPUI interface with an inspector, overlays, and smooth camera controls.
- **Observability**: Persist metrics and snapshots to DuckDB for replay, analytics, and regression testing.
- **Extensibility**: Hot-swap brain implementations (MLP, DWRAON, experimental Assembly, plus optional NeuroFlow) without rewriting the world loop.

## Architecture at a glance

The workspace is organized for clear boundaries and fast incremental builds:

```
rust_scriptbots/
├── Cargo.toml                # Workspace manifest, shared deps/lints/profiles
├── rust-toolchain.toml       # Pinned toolchain (Rust 1.85)
├── crates/
│   ├── scriptbots-core       # Simulation core (WorldState, AgentState, tick pipeline, config)
│   ├── scriptbots-brain      # Brain trait + base implementations (mlp, dwraon, assembly)
│   ├── scriptbots-brain-ml   # Optional ML backends (Candle/Tract/tch), feature-gated
│   ├── scriptbots-brain-neuro# NeuroFlow brain (optional), feature-gated
│   ├── scriptbots-index      # Pluggable spatial indices (grid, rstar, kd-tree)
│   ├── scriptbots-storage    # DuckDB-backed persistence & analytics hooks
│   ├── scriptbots-render     # GPUI integration and visual layer (HUD, canvas renderer)
│   ├── scriptbots-app        # Binary crate wiring everything together
│   └── scriptbots-web        # Sibling WebAssembly harness (wasm-bindgen bindings; experimental)
└── docs/
    └── wasm/                 # ADRs, browser matrix, multithreading notes, rendering spikes
└── original_scriptbots_code_for_reference/  # Upstream C++ snapshot for parity
```

### Architecture diagram (high-level)
Data flows left-to-right; control surfaces are orthogonal and non-invasive:

```
                           ┌───────────────────────────────────────┐
                           │  scriptbots-brain family              │
                           │  (brain, brain-ml, brain-neuro)       │
                           └──────────────┬────────────────────────┘
                                          │ BrainRegistry (attach by key)
┌──────────────────────────────────────────▼──────────────────────────────────────────┐
│                  scriptbots-core (WorldState, Tick Pipeline)                        │
│  - SoA AgentColumns · Spatial index (scriptbots-index)                              │
│  - Deterministic: sense → brains → actuation → persistence hooks                    │
└───────────────┬───────────────────────────┬───────────────────────────┬─────────────┘
                │ AgentSnapshots            │ PersistenceBatch          │ CommandDrain (in-tick)
                │                           │                           │
        ┌───────▼────────┐           ┌──────▼──────────┐          ┌─────▼──────────┐
        │ Renderer (GUI) │           │ scriptbots-     │          │ CommandBus     │
        │ GPUI window    │           │ storage         │          │ (crossfire MPMC)│
        │ or Terminal TUI│           │  StoragePipeline│          └─────┬───────────┘
        │ (console text) │           │  (async worker) │                │
        └───────┬────────┘           └──────┬──────────┘                │
                │ World snapshots     └──────┬──────────┘                │
                │ HUD metrics                 │                          │
        ┌───────▼────────┐                   │                           │
        │ scriptbots-    │                   ▼                           │
        │ render         │             ┌────────────┐                     │
        └────────────────┘             │  DuckDB    │                     │
                                       └────────────┘                     │
                                                                          │
                                 ┌─────────────────────────────────────────▼────────────────────┐
                                 │ scriptbots-app (orchestrator)                                │
                                 │ - launches ControlRuntime (Tokio thread)                     │
                                 │ - selects Renderer (CLI flag/env)                            │
                                 │ - seeds world, installs brains, primes history               │
                                 └───────────────┬───────────────────────────────┬──────────────┘
                                                 │ REST (axum + Swagger UI)      │ MCP HTTP (mcp_protocol_sdk)
                                                 │ /api/knobs /api/config        │ tools: list_knobs,get_config,
                                                 │ /api/knobs/apply PATCH config │ apply_updates,apply_patch
                                                 │                               │
                                                 │                               │
                                          ┌──────▼───────┐                       │
                                          │ control_cli  │ (reqwest; TUI watch)  │
                                          │ list/get/set │  -> REST               │
                                          └──────────────┘                       │
                                                                                 │
┌────────────────────────────────────────────────────────────────────────────────▼─────────────┐
│ scriptbots-web (wasm)                                                                          │
│ - wasm-bindgen: default_init_options/init_sim/tick/snapshot/reset/registerBrain                │
│ - snapshot_format: json | binary (Postcard) · wasm-vs-native parity tests                      │
│ - feeds JS renderer (WebGPU/Canvas)                                                            │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

- Background workers: StoragePipeline (async writer) and ControlRuntime (Tokio) are isolated; the core drains commands inside the tick loop for deterministic application.
- Renderers are read-only consumers of world snapshots; they do not mutate simulation state directly.
- Control surfaces are transport-agnostic; both REST and MCP use the same safe `ControlHandle` and enqueue commands with back-pressure.

### Crate roles
- **`scriptbots-core`**: Simulation core with `WorldState`, `AgentState`, deterministic staged tick pipeline, config, sensor/actuation scaffolding, and brain registry bindings.
- **`scriptbots-brain`**: `Brain` trait + baseline implementations and adapters; experimental `assembly` behind a feature.
- **`scriptbots-brain-ml`**: Optional ML backends (Candle, Tract, tch) for alternative/accelerated inference (feature-gated).
- **`scriptbots-brain-neuro`**: Optional NeuroFlow-based brain; controllable at runtime via config/env (see below).
- **`scriptbots-index`**: Spatial indexing implementations; default uniform grid, optional `rstar` (R-tree) and `kd` (kiddo).
- **`scriptbots-storage`**: DuckDB persistence with buffered writes (`ticks`, `metrics`, `events`, `agents`) plus analytics helpers (e.g., `top_predators`, `latest_metrics`).
- **`scriptbots-render`**: GPUI UI layer with a window shell, HUD, canvas renderer for agents/food, selection highlights, and diagnostics overlay.
- **`scriptbots-app`**: Binary shell. Wires tracing/logging, config/env, storage pipeline, installs brains, seeds agents, and launches the GPUI shell.
- **`scriptbots-web`**: WebAssembly harness exposing bindings to init/tick/reset and snapshot the simulation; consumes `scriptbots-core` with `default-features = false` (sequential fallback; Rayon disabled on wasm).

## Current status
- Workspace scaffolding, shared lints, and profiles are in place.
- `scriptbots-core`: World state, agent runtime, staged tick, reproduction/combat hooks, history summaries, and brain registry integration are implemented; parity tasks are tracked in the plan doc.
- `scriptbots-render`: GPUI window + HUD + canvas renderer with camera controls, selection highlights, and diagnostics overlay; audio is optional via `kira` feature.
- `scriptbots-brain`: `MlpBrain` available; experimental `assembly` feature; DWRAON planned; registry wiring present.
- `scriptbots-brain-neuro`: NeuroFlow-backed brain available behind the `neuro` feature (runtime toggles below).
- `scriptbots-storage`: DuckDB persistence with buffered writes and analytics helpers.

See the migration roadmap in `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md` for staged milestones and parity checklists.

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
### Recommended defaults for performance

- Threads: By default, the core auto-budgets worker threads conservatively. Our profiling shows best throughput at 8 threads on a 32-core CPU for this workload. To match that:

```bash
SCRIPTBOTS_MAX_THREADS=8 cargo run -p scriptbots-app -- --storage memory --storage-thresholds 128,4096,1024,1024
```

- With servers disabled (avoid port conflicts/background overhead):

```bash
SCRIPTBOTS_CONTROL_REST_ENABLED=false \
SCRIPTBOTS_CONTROL_MCP=disabled \
SCRIPTBOTS_MAX_THREADS=8 \
cargo run -p scriptbots-app -- --mode terminal --storage memory --storage-thresholds 128,4096,1024,1024
```

- Profiling helpers (headless):

```bash
# No storage (isolates world.step performance)
SCRIPTBOTS_MAX_THREADS=8 cargo run -p scriptbots-app -- --profile-steps 1000

# With storage (memory) and tuned flush thresholds
SCRIPTBOTS_MAX_THREADS=8 cargo run -p scriptbots-app -- --profile-storage-steps 3000 --storage memory --storage-thresholds 128,4096,1024,1024
```


Set logging verbosity with `RUST_LOG`, for example:
```bash
RUST_LOG=info cargo run -p scriptbots-app
```

#### Terminal-only mode
- Force the emoji TUI renderer (useful on headless machines):
  ```bash
  SCRIPTBOTS_MODE=terminal cargo run -p scriptbots-app
  ```
- Auto fallback: `SCRIPTBOTS_MODE=auto` (default) will drop into terminal mode if no GUI backend is available (e.g., SSH sessions).
- Override detection:
  - `SCRIPTBOTS_FORCE_TERMINAL=1` → force terminal even when a display server is present.
  - `SCRIPTBOTS_FORCE_GUI=1` → keep GPUI even if no display variables are set (may still fail if the OS truly lacks a GUI).
- CI/headless smoke runs can bypass raw TTY requirements by setting `SCRIPTBOTS_TERMINAL_HEADLESS=1`, which drives the renderer against an in-memory buffer for a few frames.

- Emoji mode (terminal renderer):
  - Defaults ON when a modern UTF‑8 terminal is detected; press `e` to toggle at runtime.
  - Force enable via env: `SCRIPTBOTS_TERMINAL_EMOJI=1|true|yes|on`; force disable with `0|false|off|no`.
  - Heuristic: enabled if `TERM` is not `dumb/linux/vt100`, locale contains `utf-8|utf8`, and `CI` is unset.
  - Emoji mappings: terrain `🌊/💧/🏜/🌿/🌺/🪨` (lush swaps: `🐟`, `🌴`, `🌾`, barren `🥀`); agents single `🐇/🦝/🦊`, small groups `🐑/🐻/🐺`, large cluster `👥`, boosted `🚀`, spike peak `⚔` (underline). Heading arrows remain for single agents when available.
  - If emojis render as tofu/misaligned, install an emoji-capable font (e.g., Noto Color Emoji) or toggle off with `e`.
- Narrow symbols mode: press `n` to switch to width-1 friendly symbols while keeping emoji colors off-background; helpful for strict terminals/alignment.

## Quick start (platform scripts)

Use the convenience scripts in the repo root to launch ScriptBots with sensible defaults per OS. These scripts set appropriate targets, isolate build artifacts, and pick the right renderer.

### Linux — terminal mode
- Script: `run_linux_terminal_mode.sh`
- Usage:
  ```bash
  chmod +x ./run_linux_terminal_mode.sh
  ./run_linux_terminal_mode.sh
  ```
- What it does:
  - Detects CPU count into `THREADS` (`nproc`/`getconf` fallback; override by exporting `THREADS` beforehand)
  - Builds with native CPU optimizations (`RUSTFLAGS="-C target-cpu=native"`)
  - Forces terminal renderer (`SCRIPTBOTS_MODE=terminal`)
  - Runs release binary with cargo job parallelism `-j $THREADS` and passes `--threads $THREADS` to the app
- Customize:
  - Reduce CPU usage: `THREADS=2 ./run_linux_terminal_mode.sh`
  - Headless CI snapshot: export `SCRIPTBOTS_TERMINAL_HEADLESS=1` to render against an in-memory buffer
  - Logging: `RUST_LOG=info ./run_linux_terminal_mode.sh`

### macOS — terminal console
- Script: `run_macos_version_with_console.sh`
- Usage:
  ```bash
  chmod +x ./run_macos_version_with_console.sh
  ./run_macos_version_with_console.sh
  ```
- What it does:
  - Detects arch (`arm64` vs `x86_64`) and sets `--target` accordingly
  - Isolates artifacts per-arch via `CARGO_TARGET_DIR=target-macos-$ARCH`
  - Unsets any stray cross-compile/link flags for a clean native build
  - Uses all cores for build jobs and launches the app in terminal mode (`--mode terminal`)
- Customize:
  - Add app flags by appending to the final `-- ...` section (e.g., `--threads 8`)
  - Override logging: `RUST_LOG=info ./run_macos_version_with_console.sh`

### macOS — GPU GUI (Metal)
- Script: `run_macos_version_with_gui.sh`
- Usage:
  ```bash
  chmod +x ./run_macos_version_with_gui.sh
  ./run_macos_version_with_gui.sh
  ```
- What it does:
  - Same target/artifact isolation as console script
  - Prefers Metal backend for `wgpu` (`WGPU_BACKEND=metal`)
  - Builds with `--features gui` and launches GUI mode (`--mode gui`) using `--threads 8`
- Customize:
  - Tune threads: edit `--threads 8` or set `SCRIPTBOTS_MAX_THREADS` env
  - Troubleshoot rendering: you can add `--renderer-safe` to the app args if you see a black canvas

### Windows — terminal console (MSVC)
- Script: `run_windows_version_with_console.bat`
- Usage:
  - Double-click in Explorer, or run from a Developer PowerShell/Command Prompt:
    ```bat
    run_windows_version_with_console.bat
    ```
- What it does:
  - Uses MSVC target `x86_64-pc-windows-msvc`
  - Isolates artifacts under `target-windows-msvc`
  - Uses all cores for build jobs and launches terminal mode (`--mode terminal`)
- Prereqs:
  - Rust MSVC toolchain and Visual Studio Build Tools (Windows 11 SDK) installed

### Windows — GPU GUI (D3D12/Vulkan)
- Script: `run_windows_version_with_gui.bat`
- Usage:
  - Double-click in Explorer, or run from a Developer PowerShell/Command Prompt:
    ```bat
    run_windows_version_with_gui.bat
    ```
- What it does:
  - Same MSVC target/artifact isolation as console script
  - Builds with `--features gui` and launches GUI mode (`--mode gui`) using `--threads 8`
- Customize:
  - Adjust threads by editing the `--threads` value; add app flags after `--` as needed (e.g., `--debug-watermark`)

Notes (all platforms):
- The final `-- ...` segment in each script passes flags to the application binary. You can add flags like `--storage memory`, `--profile-steps 1000`, or `--det-check 200` there.
- To stream control API docs, ensure REST is enabled (default) and open `http://127.0.0.1:8088/docs` while the app runs.

### Build for Web (experimental)
```bash
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown -p scriptbots-web
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

### Feature flags & variants
- **`scriptbots-app` features**:
  - `ml` → enable `scriptbots-brain-ml`
  - `neuro` → enable `scriptbots-brain-neuro`
  - `fast-alloc` → enable mimalloc as the global allocator for improved multithreaded performance
  - Example: `cargo run -p scriptbots-app --features neuro`
- **`scriptbots-render`**:
  - `audio` → enable Kira-driven audio in the UI layer
- **`scriptbots-index`** (pluggable spatial indices):
  - Default: `grid`; optional: `rstar`, `kd`
  - Example: `cargo build -p scriptbots-index --features rstar`
- **`scriptbots-brain-ml`** (optional ML backends):
  - `candle`, `tract`, `tch` (all optional)
  - Examples:
    - `cargo build -p scriptbots-brain-ml --features candle`
    - `cargo build -p scriptbots-brain-ml --features tract`
    - `cargo build -p scriptbots-brain-ml --features tch`

Note: App-level switches for brain/index selection are wired in the binary; crate-specific features control availability.

### NeuroFlow runtime configuration (optional)
If built with the `neuro` feature, runtime toggles can be applied via env vars before launch:
```bash
SCRIPTBOTS_NEUROFLOW_ENABLED=true \
SCRIPTBOTS_NEUROFLOW_HIDDEN="64,32,16" \
SCRIPTBOTS_NEUROFLOW_ACTIVATION=relu \
cargo run -p scriptbots-app --features neuro
```
Valid activations: `tanh`, `sigmoid`, `relu`.

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

### Command-line options (scriptbots-app)
- `--mode {auto|gui|terminal}`: select renderer. Defaults to `auto` and can be set via `SCRIPTBOTS_MODE`.
  - `auto`: use GPUI when a display is detected; otherwise fall back to terminal.
  - `gui`: force GPUI; may fail on headless systems.
  - `terminal`: force emoji TUI.
 - `--dump-png <FILE>` (GUI builds): write an offscreen PNG and exit (no UI). Pair with `--png-size WxH`.
 - `--png-size WxH` (GUI builds): snapshot size for `--dump-png` (e.g., `1280x720`).
 - `--debug-watermark`: overlay a tiny diagnostics watermark in the render canvas.
 - `--renderer-safe`: force a conservative paint path (useful for troubleshooting black canvas on some Windows setups).
 - `--threads N`: cap simulation worker threads (overrides low-power defaults).
 - `--low-power`: prefer lower CPU usage (equivalent to `--threads 2` unless `--threads` is provided); also biases `auto` toward terminal.
 - `--profile-steps N`: headless `world.step()` profiling without persistence.
 - `--profile-storage-steps N`: headless profiling with selected storage mode.
 - `--storage-thresholds t,a,e,m`: override flush thresholds (tick, agent, event, metric).
 - `--profile-sweep N`: run a sweep of configurations for profiling and print a summary.
 - `--auto-tune N`: quick sweep to pick threads/thresholds for the chosen storage, then continue.

### Environment variables (quick reference)
- `RUST_LOG` — logging filter (e.g., `info`, `trace`, `scriptbots_core=debug`).
- `RAYON_NUM_THREADS` — set simulation thread pool size when `parallel` is enabled.
- `SCRIPTBOTS_MODE` — `auto|gui|terminal` (renderer selection).
- `SCRIPTBOTS_FORCE_TERMINAL` / `SCRIPTBOTS_FORCE_GUI` — hard override renderer detection (`1|true|yes`).
- `SCRIPTBOTS_TERMINAL_HEADLESS` — render TUI to an in-memory buffer for CI smoke tests.
- `SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES` — number of frames to render in headless mode (default 12; max 360).
- `SCRIPTBOTS_TERMINAL_HEADLESS_REPORT` — file path to write a JSON summary from a headless run.
- `SCRIPTBOTS_MAX_THREADS` — preferred maximum thread budget; core will cap Rayon to min of CPUs and this value (used unless `RAYON_NUM_THREADS` is already set).
- `SCRIPTBOTS_NEUROFLOW_ENABLED` — `true|false`.
- `SCRIPTBOTS_NEUROFLOW_HIDDEN` — comma-separated hidden sizes (e.g., `64,32,16`).
- `SCRIPTBOTS_NEUROFLOW_ACTIVATION` — `tanh|sigmoid|relu`.
- `SCRIPTBOTS_CONTROL_REST_ADDR` — REST bind address (default `127.0.0.1:8088`).
- `SCRIPTBOTS_CONTROL_SWAGGER_PATH` — Swagger UI path (default `/docs`).
- `SCRIPTBOTS_CONTROL_REST_ENABLED` — `true|false`.
- `SCRIPTBOTS_CONTROL_MCP` — `disabled|http` (default `http`).
- `SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR` — MCP HTTP bind address (default `127.0.0.1:8090`).

## Simulation overview
Deterministic, staged tick pipeline (seeded RNG; stable ordering):
1. Aging and scheduled tasks
2. Food respawn/diffusion
3. Reset runtime flags
4. Sense (spatial index snapshot)
5. Brain tick (no per-tick allocs)
6. Actuation (double-buffered state)
7. Food intake/sharing (deterministic reductions)
8. Combat and death (queued → commit)
9. Reproduction (mutation/crossover) in stable order
10. Persistence hooks (batched to DuckDB)

### Design principles & determinism
- **Zero undefined behavior**: no `unsafe` in v1; clear ownership and lifetimes.
- **Stable order of effects**: floating-point reductions and removals are staged and applied in a fixed order for bitwise-stable runs across thread counts.
- **Per-agent RNG**: seeds derive from a global seed + `AgentId`, keeping behavior stable as populations change and threads vary.
- **Feature-gated parallelism**: `scriptbots-core` defaults to `parallel` (Rayon), while web builds disable it for single-thread determinism.

#### Reproducible runs (seed control)
- Set a fixed seed in config: `rng_seed = <u64>`. At runtime you can apply via REST:
  ```json
  { "rng_seed": 42 }
  ```
- For CPU thread control during profiling, prefer the standard `RAYON_NUM_THREADS` env var.

### Data model & spatial indexing
- **SoA layout**: agents use cache-friendly columns (`AgentColumns`) for fast scans during sense/actuation.
- **Generational IDs**: slotmap-backed `AgentId` prevents stale references and enables stable iteration.
- **Spatial index**: uniform hash grid by default (opt-in `rstar`/`kd`). Sense builds a read-only snapshot; actuation writes into a double buffer to avoid races.

### Sensors & outputs
- **Sensors**: multi-eye vision cones (angular), smell/sound/blood channels with attenuation, temperature discomfort, and clock/age cues.
- **Outputs**: differential drive (wheel velocities), color/indicator pulses, spike length easing, give intent (altruistic food sharing), boost control, sound output.
- **Mapping**: outputs drive physics and side-effects (e.g., spike damage scales with spike length and speed) and are logged for analytics.

### Brains & evolution
- **Brain trait** with `tick`/`mutate`/`crossover`; implementations include MLP (production), `dwraon` (feature), `assembly` (experimental).
- **Brain registry**: per-run registry attaches runners by key, enabling hybrid populations and runtime selection.
- **Genome & genetics**: genomes capture topology/activations; mutation/crossover create hybrid births with lineage tracking and tests.
- **NeuroFlow** (optional): deterministic CPU MLP with runtime toggles; seed-stable outputs verified in tests.

### Environment: food, terrain, temperature
- **Food dynamics**: configurable growth, decay, diffusion, and fertility capacity; speed-based intake and reproduction bonuses mirror legacy behavior.
- **Topography**: tile-based terrain/elevation influence fertility and movement energy (downhill momentum/energy costs).
- **Temperature**: gradient and per-agent preference drive discomfort drains; exposed in config and analytics.
- **Closed worlds & seeding**: enforce closed ecosystems; maintain population floors and scheduled spawns.

### Combat & mortality analytics
- **Spikes**: damage scales with requested spike length and agent speed; collision resolution is staged for determinism.
- **Carcass sharing**: meat distribution honors age scaling and diet tendencies; events persisted for analysis.
- **Analytics**: attacker/victim flags (carnivore/herbivore), births/deaths, hybrid markers, age/boost tracking, and per-tick summaries feed the HUD and DuckDB.

## Rendering & UX
- GPUI window, HUD, and canvas renderer for food tiles and agents (circles/spikes).
- Camera controls: pan/zoom; keyboard bindings for pause, draw toggle, speed ±.
- Overlays: selection highlights, diagnostics panel; charts and advanced overlays are staged in the plan.
- Inspector: per-agent stats and genome/brain views (scoped to plan milestones).
- Optional audio via `kira` (feature `audio`).

### Accessibility & input
- **Colorblind-safe palettes** (deuteranopia/protanopia/tritanopia) and a high-contrast mode; UI elements and overlays respect palette transforms.
- **Keyboard remapping** with conflict resolution and capture mode; discoverable bindings in the HUD.
- **Narration hooks** prepared for future screen-reader integration; toggles surfaced in the inspector.

### Renderer abstraction
- The app selects a `Renderer` implementation at runtime (`gpui` or `terminal`) via `--mode {auto|gui|terminal}` or environment variables. Both renderers consume the same world snapshots and control bus.

### Keyboard shortcuts (GUI)
- Playback: `space` pause/resume, `+`/`-` speed up/down, `s` single-step
- Views: `d` toggle drawing, `f` toggle food overlay, `Ctrl+Shift+O` toggle agent outlines
- Spawning: `a` add crossover agents, `q`/`h` spawn carnivore/herbivore
- World: `c` toggle closed environment, `o` follow oldest, `s` follow selected
- Accessibility: `p` cycle color palettes (with keyboard rebinding support)

### Audio system
- Optional `kira`-backed mixer (feature `audio`) with event-driven cues (births, deaths, spikes) and accessibility toggles.
- Channels planned for ambience/effects; platform caveats apply on Linux/WSL2. Audio is disabled in wasm; use Web Audio API from JS if needed.

### Terminal mode (planned)
An emoji-rich terminal renderer is planned behind a `terminal` feature/CLI mode (`--mode {auto|gui|terminal}`) with fallback when GPUI cannot start. See the “Terminal Rendering Mode (Emoji TUI)” section in `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md`.

#### Terminal-only mode
- Force the emoji TUI renderer (useful on headless machines):
  ```bash
  SCRIPTBOTS_MODE=terminal cargo run -p scriptbots-app
  ```
- Auto fallback: `SCRIPTBOTS_MODE=auto` (default) will drop into terminal mode if no GUI backend is available (e.g., SSH sessions).
- Override detection:
  - `SCRIPTBOTS_FORCE_TERMINAL=1` → force terminal even when a display server is present.
  - `SCRIPTBOTS_FORCE_GUI=1` → keep GPUI even if no display variables are set (may still fail if the OS truly lacks a GUI).
- CI/headless smoke runs can bypass raw TTY requirements by setting `SCRIPTBOTS_TERMINAL_HEADLESS=1`, which drives the renderer against an in-memory buffer for a few frames.

- Emoji mode (terminal renderer):
  - Defaults ON when a modern UTF‑8 terminal is detected; press `e` to toggle at runtime.
  - Force enable via env: `SCRIPTBOTS_TERMINAL_EMOJI=1|true|yes|on`; force disable with `0|false|off|no`.
  - Heuristic: enabled if `TERM` is not `dumb/linux/vt100`, locale contains `utf-8|utf8`, and `CI` is unset.
  - Emoji mappings: terrain `🌊/💧/🏜/🌿/🌺/🪨` (lush swaps: `🐟`, `🌴`, `🌾`, barren `🥀`); agents single `🐇/🦝/🦊`, small groups `🐑/🐻/🐺`, large cluster `👥`, boosted `🚀`, spike peak `⚔` (underline). Heading arrows remain for single agents when available.
  - If emojis render as tofu/misaligned, install an emoji-capable font (e.g., Noto Color Emoji) or toggle off with `e`.
- Narrow symbols mode: press `n` to switch to width-1 friendly symbols while keeping emoji colors off-background; helpful for strict terminals/alignment.

Keybinds: space (pause), +/- (speed), s (single-step), ?/h (help), q/Esc (quit). The terminal HUD shows tick/agents/births/deaths/energy and an emoji world mini-map that adapts to color support.

## Storage & analytics
- DuckDB schema (`ticks`, `metrics`, `events`, `agents`) with buffered writes and maintenance (`optimize`, `VACUUM`).
- Analytics helpers: `latest_metrics`, `top_predators`.
- Deterministic replay tooling is planned in the roadmap.

### DuckDB tables and usage
- Tables created automatically on first run:
  - `ticks(tick, epoch, closed, agent_count, births, deaths, total_energy, average_energy, average_health)`
  - `metrics(tick, name, value)` (primary key `(tick,name)`)
  - `events(tick, kind, count)` (primary key `(tick,kind)`)
  - `agents(tick, agent_id, generation, age, position_x, position_y, velocity_x, velocity_y, heading, health, energy, color_r, color_g, color_b, spike_length, boost, herbivore_tendency, sound_multiplier, reproduction_counter, mutation_rate_primary, mutation_rate_secondary, trait_smell, trait_sound, trait_hearing, trait_eye, trait_blood, give_intent, brain_binding, food_delta, spiked, hybrid, sound_output, spike_attacker, spike_victim, hit_carnivore, hit_herbivore, hit_by_carnivore, hit_by_herbivore)`
- Example queries:
  ```sql
  -- Latest metrics snapshot
  select m.name, m.value
  from metrics m
  where m.tick = (select max(tick) from metrics)
  order by name;

  -- Top predators by average energy
  select agent_id, avg(energy) as avg_energy, max(spike_length) as max_spike_length
  from agents
  group by agent_id
  order by avg_energy desc
  limit 10;
  ```
Configuration: persistence buffers flush automatically; call `optimize()` periodically in long sessions (the app pipeline already triggers maintenance).

### Advanced analytics cookbook (DuckDB)
Population trend (10-tick moving average):
```sql
with ticks as (
  select tick, agent_count, row_number() over(order by tick) as rn
  from ticks
)
select t1.tick,
       avg(t2.agent_count) as population_ma10
from ticks t1
join ticks t2 on t2.rn between t1.rn-9 and t1.rn
group by t1.tick
order by t1.tick;
```
Kill ratios (carnivore vs herbivore):
```sql
select sum(case when hit_carnivore then 1 else 0 end) as carnivore_hits,
       sum(case when hit_herbivore then 1 else 0 end) as herbivore_hits
from agents;
```
Energy histogram at latest tick:
```sql
with latest as (select max(tick) as t from agents)
select width_bucket(energy, 0, 2.0, 20) as bucket,
       count(*)
from agents, latest
where agents.tick = latest.t
group by bucket
order by bucket;
```

### Storage evolution & maintenance
- Schema compatibility: additive changes only until v1; breaking changes guarded behind feature flags and migration scripts.
- Maintenance: the storage worker batches inserts and exposes `optimize()`/`VACUUM` hooks; long sessions should call `optimize()` periodically (the app pipeline already schedules it).

## Development workflow
- **Coding standards**: See `RUST_SYSTEM_PROGRAMMING_BEST_PRACTICES.md`. Embrace `Result`-based errors, clear traits, and avoid `unsafe`.
- **Linting**: `cargo clippy --workspace --all-targets --all-features -W clippy::all -W clippy::pedantic -W clippy::nursery`
- **Formatting**: `cargo fmt --all`
- **Tests**: `cargo test --workspace` (simulation and GPUI tests will be added as systems land)
- **Profiles**: Release uses LTO, single codegen unit, and abort-on-panic for optimal binaries.

## Testing & CI
- **Core tests**: unit and property tests for reproduction math, spike damage, food sharing/consumption; determinism tests run seeded scenarios and assert stable summaries.
- **Render tests**: GPUI compile-time view tests; terminal HUD headless smoke tests (`SCRIPTBOTS_TERMINAL_HEADLESS=1`).
- **Benchmarks**: `criterion` harness for ticks/sec at various agent counts.
- **CI**: matrix for macOS 14 and Ubuntu 24.04; wasm job builds `scriptbots-web`, runs parity tests in headless Chromium; release job uses `cargo dist` with optional macOS codesigning.

## Performance & profiling
- CPU profiling (Linux/macOS): run with `RUSTFLAGS='-g'` and use `perf record`/`perf report` or `dtrace`/Instruments; annotate hot paths in sense/actuation.
- Tracy (optional): integrate client in dev builds to visualize frame times and background worker activity.
- Threading: tune `RAYON_NUM_THREADS` to match physical cores; verify determinism with seeded runs.
- Rendering: measure HUD/canvas frame times; avoid per-frame allocations; prefer batched path building.

## Tracing & logging
- Logging uses `tracing` with `RUST_LOG` filters (e.g., `RUST_LOG=info,scriptbots_core=debug`).
- Categories of interest:
  - `scriptbots_core::world` — tick summaries, seeding, closed/open flips
  - `scriptbots_storage` — flushes, optimize/vacuum
  - `scriptbots_app::servers` — REST and MCP server lifecycle, tool invocations
  - `scriptbots_render` — window lifecycle, input bindings
- Prefer structured fields (e.g., `tick = summary.tick.0`) for machine-readable logs. Avoid panics in production; release profile uses `panic = abort`.

## Runtime control surfaces

### REST Control API (with Swagger UI)
- Default address: `http://127.0.0.1:8088` (override `SCRIPTBOTS_CONTROL_REST_ADDR`)
- Swagger UI path: `/docs` (override `SCRIPTBOTS_CONTROL_SWAGGER_PATH`)
- OpenAPI JSON: `/api-docs/openapi.json`
- Enable/disable: `SCRIPTBOTS_CONTROL_REST_ENABLED=true|false`
- Endpoints:
  - `GET /api/knobs` → list flattened config knobs
  - `GET /api/config` → fetch entire config snapshot
  - `PATCH /api/config` → apply JSON object patch `{ ... }`
  - `POST /api/knobs/apply` → apply list of `{ path, value }` updates
  - `GET /api/ticks/latest` → latest tick summary (JSON)
  - `GET /api/ticks/stream` → server-sent events stream of tick summaries (SSE)
  - `GET /api/ticks/ndjson` → newline-delimited JSON stream of tick summaries (NDJSON)
  - `GET /api/screenshot/ascii` → ASCII snapshot of terminal mini-map (text/plain)
  - `GET /api/screenshot/png` → offscreen PNG snapshot (requires GUI feature)
  - `GET /api/hydrology` → hydrology snapshot (flow directions, accumulation, basins) if available
  - `GET /api/events/tail` → recent events (birth/death/combat) ring buffer
  - `GET /api/scoreboard` → top carnivores and oldest agents at a glance
  - `GET /api/agents/debug` → lightweight agent debug table (filters: ids, diet, selection, brain)
  - `POST /api/selection` → queue a selection update (modes: set/add/remove; optional state)
  - `GET /api/presets` → list scenario presets
  - `POST /api/presets/apply` → apply preset by name
  - `GET /api/config/audit` → recent config patches (audit ring buffer)

REST quickstart:
```bash
# 1) Start the app
cargo run -p scriptbots-app

# 2) Open Swagger UI in a browser
#    http://127.0.0.1:8088/docs

# 3) List knobs
curl -s http://127.0.0.1:8088/api/knobs | jq '.[0:10]'

# 4) Patch configuration (enable NeuroFlow, set layers, activation)
curl -s -X PATCH http://127.0.0.1:8088/api/config \
  -H 'content-type: application/json' \
  -d '{"patch":{"neuroflow":{"enabled":true,"hidden_layers":[64,32,16],"activation":"relu"}}}' | jq .

# 5) Apply typed updates
curl -s -X POST http://127.0.0.1:8088/api/knobs/apply \
  -H 'content-type: application/json' \
  -d '{"updates":[{"path":"food_max","value":0.6}]}' | jq .

# 6) Stream ticks as NDJSON (Ctrl+C to stop)
curl -s http://127.0.0.1:8088/api/ticks/ndjson | head -n 5

# 7) Take an ASCII screenshot (text) and a PNG (requires GUI feature)
curl -s http://127.0.0.1:8088/api/screenshot/ascii > frame.txt
curl -s http://127.0.0.1:8088/api/screenshot/png > frame.png
```

Example PATCH body:
```json
{ "food_max": 0.6, "neuroflow": { "enabled": true, "hidden_layers": [64,32,16], "activation": "relu" } }
```

### Control CLI (`scriptbots-control`)
- Points to the REST API (default `SCRIPTBOTS_CONTROL_URL=http://127.0.0.1:8088`):
```bash
cargo run -p scriptbots-app --bin control_cli -- list
cargo run -p scriptbots-app --bin control_cli -- get
cargo run -p scriptbots-app --bin control_cli -- set neuroflow.enabled true
cargo run -p scriptbots-app --bin control_cli -- patch --json '{"food_max":0.6}'
cargo run -p scriptbots-app --bin control_cli -- watch --interval-ms 750
cargo run -p scriptbots-app --bin control_cli -- export metrics --db scriptbots.db --last 1000 --out latest_metrics.csv
# New commands:
cargo run -p scriptbots-app --bin control_cli -- presets
cargo run -p scriptbots-app --bin control_cli -- apply-preset arctic
cargo run -p scriptbots-app --bin control_cli -- screenshot --out screenshots/frame_0001.txt
cargo run -p scriptbots-app --bin control_cli -- screenshot --png --out screenshots/frame_0001.png
cargo run -p scriptbots-app --bin control_cli -- hydrology
```

### Scenario layering & deterministic replay CLI
- **Layered configs**: pass one or more `--config path/to/file.toml` (or `.ron`) flags—or set `SCRIPTBOTS_CONFIG` with semicolon-separated paths—to build scenarios from reusable fragments (e.g., `base.toml → arctic_biome.toml → evolution_study.toml`). Layers merge in order before env overrides, unlocking repeatable experiments without editing code.
- **Config inspection**: add `--print-config` to dump the merged configuration (default JSON) or `--write-config output.toml` to persist it; choose `--config-format json|toml|ron` and combine with `--config-only` for a dry run in CI/tooling workflows.
- **Replay verification**: `cargo run -p scriptbots-app -- --replay-db run.duckdb [--compare-db candidate.duckdb] [--tick-limit 500]` loads persisted events, re-simulates ticks headlessly, and reports colored diffs (tick/sequence mismatches, event payload divergences) together with event-type counts.
- **Storage helpers**: DuckDB accessors (`max_tick`, `load_replay_events`, `replay_event_counts`) underpin the CLI so analytics pipelines or external tools can reuse the same deterministic data.

### MCP HTTP server (Model Context Protocol)
- Default: `127.0.0.1:8090` over HTTP; disable with `SCRIPTBOTS_CONTROL_MCP=disabled`.
- Override bind address: `SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR=127.0.0.1:9090`.
- Tools exposed:
  - `list_knobs` → returns array of knob entries
  - `get_config` → returns full config snapshot
  - `apply_updates` → accepts `{ updates: [{ path, value }, ...] }`
  - `apply_patch` → accepts `{ patch: { ... } }`
Notes: Only HTTP transport is supported here; stdio/SSE are not used.

MCP quickstart:
- Start the app; verify MCP binds on `127.0.0.1:8090` (override via `SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR`).
- Connect an MCP HTTP client to the endpoint; available tools: `list_knobs`, `get_config`, `apply_updates`, `apply_patch`.
- Each tool returns structured JSON; use your MCP-compatible agent to orchestrate parameter sweeps and log findings to DuckDB.

## Configuration files & scenarios
- **Current state**: all configuration changes flow through the control surfaces (REST, MCP HTTP, CLI). Values persist only for the lifetime of the session unless you export them yourself.
- **Planned**: upcoming releases will load layered configuration files (TOML/RON) so experiments can stack reusable building blocks (e.g., base world defaults → biome overrides → study-specific tweaks).
  ```text
  base.toml → arctic_biome.toml → evolution_study.toml
  ```
- **Workaround today**: capture the desired configuration via the REST API (`GET /api/config`), version it externally, and apply deltas back with `PATCH /api/config` or the CLI.
- **Value**: researchers gain repeatable, composable scenarios without hand-editing large monolithic files, making it easy to swap biomes, mutation parameters, or analytics presets on demand.

## Deterministic replay roadmap
- **Already implemented**
  - ✅ Event-log schema (`replay_events` table) storing every tick’s RNG scope, brain outputs, and actions.
  - ✅ Type-safe encoding for replay artifacts shared by storage and analytics crates.
  - ✅ Storage plumbing to persist replay batches alongside standard metrics.
- **Planned**
  - ❌ Headless replay runner capable of deterministic re-simulation from stored events.
  - ❌ Branch/diff workflows comparing Rust vs. Rust PR builds vs. the legacy C++ baseline.
  - ❌ CLI tooling that surfaces divergence reports and snapshot diffs.
  - ❌ DuckDB-backed analysis views for quick triage of regressions and experiment outcomes.
- **Use cases**
  - Parity testing between the Rust port and the original C++ implementation.
  - Regression prevention by replaying critical seeds in CI or pre-merge checks.
  - Debugging elusive bugs by reproducing exact agent decisions tick-by-tick.
  - Long-running research experiments that demand bitwise-stable replays.

## Security & operations
- REST and MCP servers bind to loopback by default. If you expose them externally, front with TLS and configure CORS appropriately. The WASM path requires COOP/COEP headers only when enabling multithreading; single-thread builds avoid this.

### Configuration knobs (examples)
All configuration can be inspected and updated at runtime via REST/CLI/MCP. Common knobs:
- World: `world_width`, `world_height`, `closed`
- Population: `population_minimum`, `population_spawn_interval`
- Food: `food_max`, `food_regrowth_rate`, `food_diffusion`, `food_decay`, `fertility_strength`
- Temperature: `temperature_gradient`, `temperature_offset`
- Reproduction: `reproduction_rate_carnivore`, `reproduction_rate_herbivore`, `mutation.{primary,secondary}`
- Traits: `trait_modifiers.{smell,sound,hearing,eye,blood}`
- NeuroFlow: `neuroflow.{enabled,hidden_layers,activation}`

Use `GET /api/knobs` to discover the full flattened list with current values.

Runtime constraints:
- Changing `world_width`/`world_height` at runtime is rejected; restart with new dimensions.
- Some composite changes may be coerced (e.g., number/string parsing) but type mismatches are rejected with a clear error.

### Control bus architecture
- The app owns a bounded MPMC `CommandBus`; external surfaces (REST, MCP, CLI) enqueue `ControlCommand`s.
- The simulation drains the queue inside the tick loop before state mutation, guaranteeing coherent updates and avoiding data races.
- Back-pressure: when the queue is full, commands are rejected with a clear error; clients should retry with jitter.

## Contributing
- Keep changes scoped to the relevant crate; prefer improving existing files over adding new ones unless functionality is genuinely new.
- Update docs where it helps future maintainers understand decisions and invariants.
- For larger tasks, update `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md` inline to mark progress.

## WebAssembly (sibling crate plan)
We maintain a sibling browser-targeted crate, `scriptbots-web`, that reuses core crates without invasive changes. See `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` and `docs/wasm/` (ADRs, audits, capability matrix). Initial MVP runs single-threaded by disabling `scriptbots-core`’s `parallel` feature on wasm; WebGPU vs Canvas2D rendering is under evaluation.

> Quick peek: `crates/scriptbots-web/web/` ships a Canvas demo harness that consumes the wasm snapshots, surfaces live metrics, and can be served locally via `python -m http.server`. Binary snapshots (`snapshot_format: "binary"`) and custom seeding strategies are already wired in for experimentation.

Helpful docs:
- `docs/wasm/adrs/ADR-001-wasm-rendering.md` — rendering stack decision record
- `docs/wasm/adrs/ADR-002-browser-persistence.md` — browser persistence approach
- `docs/wasm/adrs/ADR-004-component-model.md` — component model/WASI Preview assessment
- `docs/wasm/browser_matrix.csv` — browser capabilities (WebGPU, SAB, SIMD)

### WASM snapshot format & APIs
- `snapshot_format`: `json` (default) or `binary` (Postcard `Uint8Array`).
- APIs: `default_init_options()`, `init_sim(options)`, `tick(steps)`, `snapshot()`, `reset(seed?)`, `registerBrain("wander"|"mlp"|"none")`.
- Determinism: wasm-vs-native parity tests compare snapshots for fixed seeds; single-thread fallback is default (Rayon disabled).

### WASM hosting guide (COOP/COEP)
- For multithreading (future), browsers require SharedArrayBuffer with headers:
  - `Cross-Origin-Opener-Policy: same-origin`
  - `Cross-Origin-Embedder-Policy: require-corp`
- Local dev: serve with a static server that sets these headers (or use a service worker). For now, single-thread builds avoid the requirement.
- CI: the wasm job runs parity tests in headless Chromium; see `.github/workflows/ci.yml`.

## Licensing
Licensed under `MIT OR Apache-2.0` (see workspace manifest).

## Credits
- Original ScriptBots by Andrej Karpathy (reference snapshot included under `original_scriptbots_code_for_reference/`).
- This Rust port is an independent, from-scratch implementation guided by parity goals and modern Rust/GPUI best practices.

## FAQ
- **What platforms are supported?** Linux, macOS, and Windows 11 are targeted. Windows is supported natively (MSVC toolchain) and via WSL2. Early UI milestones may see platform-specific polish arriving at different times.
- **Where do I start hacking?** `scriptbots-core` for the world model; `scriptbots-render` for the GPUI view; `scriptbots-brain` for brain interfaces; `scriptbots-storage` for persistence.

## Troubleshooting
- **MSVC/SDK link errors on Windows**: Ensure VS Build Tools "Desktop development with C++" and Windows 11 SDK are installed. Then run `rustup default stable-x86_64-pc-windows-msvc`.
- **Blank or crashing window**: Update GPU drivers. On WSL2, update the WSL kernel and try again. Verify that your system supports D3D12 (Windows) or Vulkan/Metal (Linux/macOS).
- **DuckDB file lock errors**: Close any external tools accessing the DB file and retry. Prefer unique DB paths per run while developing.
- **Determinism regressions**: Ensure you haven't introduced unordered parallel reductions; stage results and apply in a stable commit phase.

## Releases
- Releases are built via [`cargo dist`](https://github.com/axodotdev/cargo-dist) in the `release-builds` GitHub Actions workflow. Publish a new version by tagging the repository (`git tag v0.x.y && git push origin v0.x.y`) or running the workflow manually with a `tag` input.
- The workflow produces archives for Linux (`x86_64-unknown-linux-gnu`), Windows (`x86_64-pc-windows-msvc`), and a universal macOS build (Apple Silicon + Intel). Artifacts are uploaded as workflow run assets for review before attaching them to a GitHub Release.
- **macOS codesigning**: provide the following repository secrets for automatic signing (the job skips codesign if they are absent):
  - `MACOS_CERT_BASE64`: base64-encoded `.p12` Developer ID certificate.
  - `MACOS_CERT_PASSWORD`: password used to protect the `.p12`.
  - `MACOS_SIGNING_IDENTITY`: e.g. `"Developer ID Application: Example Corp (TEAMID1234)"`.
  - `MACOS_KEYCHAIN_PASSWORD` (optional): override keychain password used on the runner.
- Certificates are imported into a temporary keychain, the binaries (and `.app` bundles when present) are signed with the supplied identity, and the archives are repackaged. Add notarization credentials later if we adopt automated notarization.
- Release operators should verify the uploaded artifacts locally (`codesign --verify --deep` on macOS, `shasum -a 256` on every platform) before drafting public releases.

## Roadmap (condensed)
1. Core data structures and config (done); expand parity (metabolism, locomotion, food math, carcass sharing).
2. World mechanics and determinism under parallelism; spatial index tuning.
3. Brains: MLP shipped; DWRAON + Assembly (feature-gated) and NeuroFlow optional.
4. Storage: extend analytics, add replay hooks and regression tests.
5. Rendering: HUD/overlays/inspector polish; performance diagnostics.
6. Packaging/CI: release builds, binaries; wasm sibling crate scaffolding (non-invasive).

### Mixed brain families (default)
- The app now registers multiple brain families by default (MLP, DWRAON, Assembly experimental, NeuroFlow) and seeds mixed populations automatically. Random spawns are bound to a sampled brain family.
- NeuroFlow is enabled in the default config; edit at runtime via REST/CLI or env (e.g., SCRIPTBOTS_NEUROFLOW_*).
- Sexual reproduction only occurs within the same brain kind (species barrier). Cross-kind parents fall back to random spawns. This allows fair A/B comparisons between families.
- To force a single brain for new agents, bind a chosen `brain_key` to agents via `WorldState::bind_agent_brain` or modify the seeding function to always pick that key.
