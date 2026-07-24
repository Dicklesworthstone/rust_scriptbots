## Rust ScriptBots

ScriptBots is a modern Rust reimagining of Andrej Karpathy’s classic agent-based evolution simulator. Our goal is a faithful, deterministic port with a GPU-accelerated UI, pluggable brain implementations, and first-class analytics. This is a multi-crate Cargo workspace separating simulation core, renderer-neutral runtime protocol, brains, storage, rendering, and the application shell.

The authoritative roadmap is `PLAN_TO_REARCHITECT_AND_REVIVE_RUST_SCRIPTBOTS.md`. The older GPUI port plan is retained as historical design evidence, and a sibling WebAssembly plan lives in `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md`.

### Philosophy & purpose
- **Why this exists**: ScriptBots is a minimalist artificial life laboratory. By rebuilding the original simulator with rigorously deterministic Rust systems, we can observe, measure, and reproduce emergent behavior at scale—without undefined behavior or global state muddying results.
- **What we learn**: How simple sensory channels and local rules produce complex population dynamics—cooperation vs. predation, resource gradients shaping migration, lineage divergence under different mutation schedules, and the role of perception in survival.
- **LLM-in-the-loop science**: The REST API, CLI, and MCP HTTP server expose the full control surface (knobs, patches, snapshots). This lets an external LLM agent act as an autonomous lab assistant: steering experiments, sweeping parameter spaces, logging observations into FrankenSQLite, and drafting human-readable reports.
  - Example workflows:
    - Parameter sweeps: vary `mutation.{primary,secondary}` and temperature gradients; record birth/death ratios and equilibrium populations.
    - Interventions: toggle `closed` worlds, inject carnivore cohorts, or freeze food diffusion to test resilience.
    - Reporting: query or export FrankenSQLite tables to generate charts and tables describing discovered phenomena (e.g., altruistic giving thresholds that stabilize mixed diets).
- **A brain testbed**: The `Brain` trait and registry allow swapping decision engines—handwritten controllers, MLP/DWRAON/Assembly, or NeuroFlow—while holding the environment constant. This enables fair comparisons of:
  - Perception encoding (multi-eye vision, smell/sound/blood) and how architectures exploit them.
  - Locomotion control (exact legacy two-rotation parity by default, selectable differential drive) and energy/health trade-offs.
  - Evolutionary operators (mutation/crossover) and speciation pressures.
- **Reproducible research**: Deterministic pipelines + a growing replay roadmap mean results can be shared and re-run bit-for-bit, making the project a solid platform for pedagogy, papers, and benchmarking new brain designs.

### Why this exists
- **Determinism and safety**: Replace legacy C++/GLUT and global state with idiomatic Rust, tightly audited platform-boundary `unsafe`, and reproducible runs.
- **Performance at scale**: Data-parallelism (Rayon) and cache-friendly layouts to simulate thousands of agents efficiently.
- **Modern UX**: Declarative, GPU-accelerated GPUI interface with an inspector, overlays, and smooth camera controls.
- **Observability**: Persist metrics and snapshots to FrankenSQLite for replay, analytics, and regression testing while the UI consumes lock-free immutable read models.
- **Extensibility**: Hot-swap brain implementations (MLP, DWRAON, experimental Assembly, plus optional NeuroFlow) without rewriting the world loop.

## Architecture at a glance

The workspace is organized for clear boundaries and fast incremental builds:

```
rust_scriptbots/
├── Cargo.toml                # Workspace manifest, shared deps/lints/profiles
├── rust-toolchain.toml       # Pinned nightly-2026-07-09 toolchain (MSRV 1.89)
├── crates/
│   ├── scriptbots-core       # Simulation core (WorldState, AgentState, tick pipeline, config)
│   ├── scriptbots-runtime    # Sole-owner HostCore, protocol, fixed-deadline lifecycle, null frontend
│   ├── scriptbots-brain      # Brain trait + base implementations (mlp, dwraon, assembly)
│   ├── scriptbots-brain-ml   # Candle/Tract/tch probes plus the optional Frankentorch FtBrain
│   ├── scriptbots-brain-neuro# NeuroFlow brain (optional), feature-gated
│   ├── scriptbots-index      # Uniform-grid index; alternate backends are not implemented
│   ├── scriptbots-storage    # FrankenSQLite persistence worker & analytics snapshots
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
│  - Deterministic: sense → brains → actuation → persistence projection               │
└───────────────┬───────────────────────────┬───────────────────────────┬─────────────┘
                │ AgentSnapshots            │ StepOutcome                │ ControlCommand ↑ / disposition ↓
                │                           │ + Arc<PersistenceBatch>     │
        ┌───────▼────────┐          ┌────────▼───────────────┐     ┌─────▼──────────┐
        │ Renderer (GUI) │          │ Application runtime    │     │ Application   │
        │ GPUI window    │          │ PersistenceAdmission   │     │ command driver│
        │ or Terminal TUI│          │ Session / step driver  │     └─────┬───────────┘
        │ (console text) │          └────────┬───────────────┘           │
        └───────┬────────┘                   │ admitted batch             │
                │ World snapshots   ┌────────▼───────────┐               │
                │ HUD metrics       │ scriptbots-storage │               │
        ┌───────▼────────┐          │ StoragePipeline    │               │
        │ scriptbots-    │          └────────┬───────────┘               │
        │ render         │                   ▼                            │
        └────────────────┘       ┌──────────────────────┐                │
                                 │ FrankenSQLite        │                │
                                 │ run-name.sqlite      │                │
                                 └──────────────────────┘                │
                                                                          │
                                 ┌─────────────────────────────────────────▼────────────────────┐
                                 │ scriptbots-app (orchestrator)                                │
                                 │ - launches ControlRuntime (Tokio thread)                     │
                                 │ - owns CommandBus and drains one ordered vector per boundary │
                                 │ - selects Renderer (CLI flag/env)                            │
                                 │ - seeds world, installs brains, primes history               │
                                 └───────────────┬───────────────────────────────┬──────────────┘
                                                 │ REST (axum + Swagger UI)      │ MCP HTTP (fastmcp-rust)
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

- Background workers: `StoragePipeline` is a bounded-admission writer whose dedicated thread creates and exclusively owns its FrankenSQLite connection; `ControlRuntime` (Tokio) is separately isolated. A successful enqueue is not a durability receipt: flush or shutdown must acknowledge the earlier transactions. Application drivers drain one ordered command vector at a boundary; core applies world-owned mutations and returns normalized playback explicitly instead of owning a second queue.
- Startup is fail-closed and transactional. Renderer selection and control-environment validation happen first, then every enabled REST/MCP socket is prebound and held before configuration output, auto-tuning, process-priority changes, world construction, or storage reservation. Launch consumes those exact listeners, so a bind failure cannot leave config, tuning, or run-database artifacts behind and cannot race a later rebind.
- REST and MCP run as supervised sibling tasks. An unexpected error or clean task exit stops the sibling, preserves the original failure as the root cause, and publishes failed runtime health; the TUI, GPUI, and Bevy frontends observe that health and terminate with the same root failure. Graceful shutdown joins both tasks and releases both listeners.
- That supervision guarantee covers ordinary returned errors and task exits. Debug/test builds use unwinding boundaries to exercise panic reporting, while the shipped `panic = "abort"` release profile intentionally cannot recover from a panic or promise destructor-based cleanup after one.
- Frontends do not query FrankenSQLite or wait on a storage mutex during paint. GPUI now has exactly one session-level simulation driver, independent of either window's repaint cadence, that owns scientific stepping, command draining, and shared pause/speed state; both the HUD and world window are presentation-only projections. This closes the characterized GPUI double-drive and per-view control defects. `HostCore` and its native lifecycle are implemented, but TUI/GPUI/Bevy and the live server transports remain on their interim adapters until the dedicated migration beads move that ownership into the renderer-neutral host.
- `scriptbots-runtime` owns the renderer-neutral command, two-axis status, snapshot, event-cursor, and manual-drive contracts plus the exact sole-owner `HostCore`. Command lifecycle schema v1 retains the exact envelope, client namespace/sequence, optional admission order, typed control/scientific/config guards, and contiguous application transitions; every terminal runtime outcome has a journal obligation. Its optional `native-asupersync` feature drives that same `!Send` host as one current-thread root future at absolute deadlines. Commands and journal-ready signals wake the owner, catch-up is bounded and reported, quiescent paused worlds have no periodic timer, and ordered shutdown retains its exact host and journal obligations without spawning or detaching a simulation task. The lifecycle persistence source is committed under `bd-2z0.5.2`; centralized DSR proof is pending.
- Brain introspection is an explicit read-only projection, never a per-tick side effect. A client issues a revisioned request for up to eight stable `AgentUid` values; core and each brain family enforce independent bounds for layers, names, values, edges, source scalars, and retained payload. Responses carry the exact source tick and typed unavailable/clipped status, while TUI and GPUI cache by client, stable identity, and source tick. With no request, no brain is inspected and NeuroFlow performs no inspection JSON serialization; digest-neutrality and next-output purity are tested across the supported families.
- Control surfaces are transport-agnostic; both REST and MCP use the same safe `ControlHandle` and enqueue commands with back-pressure.

### Crate roles
- **`scriptbots-core`**: Simulation core with `WorldState`, `AgentState`, deterministic staged tick pipeline, config, sensor/actuation scaffolding, and brain registry bindings.
- **`scriptbots-runtime`**: Renderer-neutral command, immutable lifecycle evidence, two-axis status, snapshot, and event-cursor contracts with typed revision domains; the sole-owner `HostCore`; a pure injected-time fixed-deadline adapter; and an optional Asupersync `=0.3.9` native lifecycle. It depends on core but not storage, servers, or renderers. The default and WASM-facing surface remains runtime-neutral; Asupersync is activated only by `native-asupersync` on native targets.
- **`scriptbots-brain`**: `Brain` trait + baseline implementations and adapters; experimental `assembly` behind a feature.
- **`scriptbots-brain-ml`**: Feature selection and the retained `ml.placeholder` sensor-copy probe for Candle, Tract, and tch, plus the non-default code-first Frankentorch `FtBrain` family. The FtBrain source is implemented under `brain-ft`; its pinned DSR compile, determinism, and benchmark proof remains pending under `bd-2z0.3.12.3`.
- **`scriptbots-brain-neuro`**: Optional NeuroFlow-based brain; controllable at runtime via config/env (see below).
- **`scriptbots-index`**: Production uniform-grid neighborhood index. The declared `rstar` and `kd` dependency features are compile-time scaffolding, not implemented index backends.
- **`scriptbots-storage`**: FrankenSQLite persistence with transactional batched writes, bounded admission, explicit flush/shutdown commit receipts, and immutable latest-value analytics snapshots for frontends.
- **`scriptbots-render`**: GPUI UI layer with a window shell, HUD, canvas renderer for agents/food, selection highlights, and diagnostics overlay.
- **`scriptbots-app`**: Binary shell. Wires tracing/logging, config/env, storage pipeline, installs brains, seeds agents, and launches the GPUI shell.
- **`scriptbots-web`**: WebAssembly harness exposing bindings to init/tick/reset and snapshot the simulation; consumes `scriptbots-core` with `default-features = false` (sequential fallback; Rayon disabled on wasm).

## Current status
- Workspace scaffolding, shared lints, and profiles are in place.
- `scriptbots-core`: World state, agent runtime, staged tick, reproduction/combat hooks, history summaries, and brain registry integration are implemented; parity tasks are tracked in the plan doc.
- `scriptbots-runtime`: the protocol boundary, typed command/revision/status domains, schema-v1 lifecycle evidence, opaque client ports, cursors, manual-drive contract, sole-owner `HostCore`, pure fixed-deadline driver, and optional current-thread Asupersync lifecycle are implemented. The `bd-2z0.5.2` lifecycle source is centralized-DSR-pending; legacy frontend and transport migration remains pending.
- `scriptbots-render`: GPUI window + HUD + canvas renderer with camera controls, selection highlights, and diagnostics overlay; audio is optional via `kira` feature.
- `scriptbots-app`: explicit renderer selection, pre-storage control-socket reservation, supervised REST/MCP lifecycle, and frontend health propagation are implemented. The full cross-feature/platform startup matrix remains a Phase 0.4 acceptance gate.
- `scriptbots-brain`: MLP and DWRAON implementations are enabled by default; Assembly remains experimental; registry wiring is present.
- `scriptbots-brain-neuro`: NeuroFlow-backed brain available behind the `neuro` feature (runtime toggles below).
- `scriptbots-storage`: the exact FrankenSQLite source is pinned; its bounded worker now has a file-backed durable outbox, stable per-batch identities, monotonic admitted/applied/durable watermarks, ordered startup recovery, controller wait deadlines, supervised worker ownership, exact recovery identity/schema proof, and a V6 run-scoped base schema with V7 canonical host archives. Code-first V8 domain-event and V9 command-lifecycle projections are committed under `bd-2z0.5.2`, with centralized DSR proof pending. Same-thread writes use the same outbox protocol, and terminal receipt/join errors preserve one exact typed root cause. Deadlines do not cancel an in-flight database call or bound the supervised reaper. Strict-run host policy, pairwise interactions, checkpoint/replay integration, and run bundles remain open.

See the active recovery roadmap in `PLAN_TO_REARCHITECT_AND_REVIVE_RUST_SCRIPTBOTS.md` for staged milestones and acceptance gates. The older GPUI port plan is historical evidence, not current policy.

## Getting started

### Prerequisites
- Rust toolchain: `nightly-2026-07-09`, pinned in `rust-toolchain.toml`; the
  workspace and locked dependency graph declare a minimum Rust version of 1.89.
  Install through `rustup`.
- OS: Linux, macOS, or Windows 11 (native or WSL2). GPU drivers should be up to date for best GPUI performance (wgpu backends: Metal/macOS, Vulkan/Linux, D3D12 or Vulkan/Windows).

### Build
```bash
cargo check
```
> **CPU tuning note:** Workspace builds now default to a portable baseline so CI runners don’t require AVX2/“native” features. Set `RUSTFLAGS="-C target-cpu=native"` locally (all launch scripts already do this) if you want host-specific tuning.

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
- Auto selection: `SCRIPTBOTS_MODE=auto` (default) chooses only a renderer compiled into the binary—GPUI first, then Bevy—and only in a real native graphical session; otherwise it uses the terminal. Linux/Unix sessions require a display environment; local macOS sessions use the native window system without X11 variables, while SSH sessions choose the terminal.
- Auto-mode policy overrides:
  - `SCRIPTBOTS_FORCE_TERMINAL=1` → choose terminal in Auto mode even when a display server is present.
  - `SCRIPTBOTS_FORCE_GUI=1` → require compiled GPUI in Auto mode even if no display variables are set; unavailable features and launch failures are errors, never terminal fallbacks.
- CI/headless smoke runs can bypass raw TTY requirements by setting `SCRIPTBOTS_TERMINAL_HEADLESS=1`, which drives the renderer against an in-memory buffer for a few frames.

- Emoji mode (terminal renderer):
  - Defaults ON when a modern UTF‑8 terminal is detected; press `e` to toggle at runtime.
  - Force enable via env: `SCRIPTBOTS_TERMINAL_EMOJI=1|true|yes|on`; force disable with `0|false|off|no`.
  - Heuristic: enabled if `TERM` is not `dumb/linux/vt100`, locale contains `utf-8|utf8`, and `CI` is unset.
  - Emoji mappings: terrain `🌊/💧/🏜/🌿/🌺/🪨` (lush swaps: `🐟`, `🌴`, `🌾`, barren `🥀`); agents single `🐇/🦝/🦊`, small groups `🐑/🐻/🐺`, large cluster `👥`, boosted `🚀`, spike peak `⚔` (underline). Heading arrows remain for single agents when available.
  - If emojis render as tofu/misaligned, install an emoji-capable font (e.g., Noto Color Emoji) or toggle off with `e`.
- Narrow symbols mode: press `n` to switch to width-1 friendly symbols while keeping emoji colors off-background; helpful for strict terminals/alignment.

- Headless report (CI-friendly):
  ```bash
  SCRIPTBOTS_MODE=terminal \
  SCRIPTBOTS_TERMINAL_HEADLESS=1 \
  SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES=24 \
  SCRIPTBOTS_TERMINAL_HEADLESS_REPORT=terminal_report.json \
  cargo run -p scriptbots-app -- --storage memory --threads 2
  ```
  This renders offscreen for N frames and writes a JSON summary (frames, ticks, births/deaths, energy stats) to `terminal_report.json`.

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

### Linux — Bevy renderer (Vulkan/GL)
- Script: `run_linux_with_bevy.sh`
- Usage:
  ```bash
  chmod +x ./run_linux_with_bevy.sh
  ./run_linux_with_bevy.sh
  ```
- What it does:
  - Detects CPU count (caps default at 8) and passes the value to cargo (`-j`) and the app (`--threads`)
  - Prefers Vulkan via `WGPU_BACKEND=vulkan`, falling back to GL when Vulkan is unavailable
  - Enables the Bevy renderer (`--features bevy_render`) and launches with `--mode bevy`
  - Sets high-performance WGPU hints (`SB_WGPU_PRESENT_MODE=full`, bloom/tonemap/fog defaults) matching the Windows helper
- Customize:
  - Limit load: `THREADS=4 ./run_linux_with_bevy.sh`
  - Force GL: `WGPU_BACKEND=gl ./run_linux_with_bevy.sh`
  - Append extra flags after the final `--` (e.g., `./run_linux_with_bevy.sh -- --debug-watermark`)

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

### macOS — Bevy renderer (Metal)
- Script: `run_macos_version_with_bevy.sh`
- Usage:
  ```bash
  chmod +x ./run_macos_version_with_bevy.sh
  ./run_macos_version_with_bevy.sh
  ```
- What it does:
  - Selects the correct target triple (`aarch64-apple-darwin` on Apple Silicon, `x86_64-apple-darwin` otherwise)
  - Isolates build artifacts per-arch and clears stray cross-compilation flags
  - Forces the Metal backend, high-performance power preference, and Bevy feature flag (`--features bevy_render`)
  - Launches ScriptBots with `--mode bevy` and a default 8-thread budget (override with `THREADS` env)
- Customize:
  - Retina tweaks: set `SB_WGPU_RES_SCALE` to `0.5` or `2.0` before running
  - Lower CPU use: `THREADS=4 ./run_macos_version_with_bevy.sh`
  - Add Bevy-specific CLI flags after `--`, e.g., `./run_macos_version_with_bevy.sh -- --dump-semantic-png docs/rendering_reference/golden/bevy_default.png` (CPU semantic raster; for real offscreen GPU captures use `--dump-scene-png SCENE.toml`)

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

### Windows — Bevy renderer (Vulkan/D3D12)
- Script: `run_windows_version_with_bevy.bat`
- Usage:
  - Double-click in Explorer, or run from a Developer PowerShell/Command Prompt:
    ```bat
    run_windows_version_with_bevy.bat
    ```
- What it does:
  - Reuses the MSVC target (`x86_64-pc-windows-msvc`) with isolated artifacts under `target-windows-msvc`
  - Sets high-performance WGPU hints (`WGPU_BACKEND=Vulkan`, `WGPU_POWER_PREFERENCE=high_performance`)
  - Builds with `--features bevy_render` and launches the Bevy renderer (`--mode bevy`) using `--threads 8`
- Customize:
  - To force D3D12 instead of Vulkan: set `set WGPU_BACKEND=d3d12` before running
  - Add Bevy-only flags (e.g., `--dump-semantic-png`, `--dump-scene-png`) after the final `--` in the script

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
  - `brain-ft` → propagate the non-default pinned Frankentorch dependencies and code-first `FtBrain` family through `scriptbots-brain-ml`
  - `neuro` → enable `scriptbots-brain-neuro`
  - `fast-alloc` → enable mimalloc as the global allocator for improved multithreaded performance
  - Example: `cargo run -p scriptbots-app --features neuro`
  - Frankentorch compile, determinism, and benchmark acceptance is DSR-only through the pinned `rust_scriptbots` profile; GitHub Actions and direct Cargo invocations are not acceptance evidence.
  - Note: default features enable `ml`, `neuro`, and `fast-alloc`. To disable defaults, use `--no-default-features` and opt-in explicitly.
- **`scriptbots-render`**:
  - `audio` → enable Kira-driven audio in the UI layer
- **`scriptbots-index`**:
  - `grid` is the implemented backend; `rstar` and `kd` currently enable dependencies only.
  - Example: `cargo build -p scriptbots-index --features rstar`
- **`scriptbots-brain-ml`**:
  - `candle`, `tract`, and `tch` retain the dependency probes and `ml.placeholder`.
  - `brain-ft` exposes `FtBrainConfig`, `FtBrainFamily`, and `FT_BRAIN_KIND`.
  - The pinned Frankentorch `vector_to_parameters` path is not F32-safe, so FtBrain keeps a canonical flat F32 genome and materializes its layer slices without widening through F64. Resolution of that upstream gap and all compile/benchmark proof remain part of `bd-2z0.3.12.3`.

Note: NeuroFlow and the native brain implementations are functional. Candle/Tract/tch inference and alternate spatial-index backends remain tracked implementation work. FtBrain now has a real code-first inference implementation, but is not accepted as complete until its pinned DSR evidence is green.

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
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

# Run tests (as they land)
cargo test --workspace

# Build optional crates with features
cargo build -p scriptbots-index --features rstar
cargo build -p scriptbots-brain-ml --features candle # compile probe; inference is still a placeholder
```

### Command-line options (scriptbots-app)
- `--mode {auto|gui|bevy|terminal}`: select renderer. Defaults to `auto` and can be set via `SCRIPTBOTS_MODE`.
  - `auto`: choose GPUI, then Bevy, only when that backend is compiled and a real native graphical session is available; otherwise use the terminal.
  - `gui`: require GPUI; an uncompiled feature or native window launch failure is returned to the caller.
  - `bevy`: require the Bevy frontend and fail clearly unless built with the `bevy_render` application feature.
  - `terminal`: force emoji TUI.
- `--bootstrap-ticks N`: explicitly run `N` science ticks after seeding and before frontend launch (default `0`, so ordinary startup launches at tick zero).
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
 - `--det-check N`: run determinism self-check (1-thread vs N-threads summaries comparison).
 - `--set PATH=VALUE`: dotted-path configuration override in TOML syntax, repeatable (e.g., `--set world_width=800 --set neuroflow.enabled=true`; string values use TOML quotes). Configuration layers apply defaults → `--config` files (in order) → environment → CLI; every applied layer appends a kind-tagged content digest to the run manifest's `scenario.ordered_config_layer_digests`, and any field where one explicit layer displaced another is recorded in the manifest's `config_overrides`.
- `--quality TIER`: visual quality shortcut for `--set render.quality=TIER`, validated at parse time (`auto|potato|low|medium|high|ultra`). The unified `render.*` settings (quality tier, post stack bloom/vignette/fog/AA, day/night cycle, TUI theme, accessibility palette) are consumed by every frontend; `None` values defer to per-tier frontend defaults.
 - `--dump-png FILE` + `--png-size WxH` (GUI builds): write an offscreen PNG and exit.
 - `--dump-semantic-png FILE` (Bevy builds): write the CPU semantic projection raster and exit. This is a semantic reference only — it does NOT exercise the GPU pipeline (formerly `--dump-bevy-png`, renamed so it can never be mistaken for a GPU capture).
 - `--dump-scene-png SCENE.toml` (Bevy builds): render a scene manifest through the REAL offscreen Bevy GPU pipeline and write capture PNGs + provenance JSON + a JSON scene log under `captures/<scene>/`. Golden workflow: `RUST_REGEN_GOLDEN=1` blesses new goldens under `crates/scriptbots-app/tests/scenes/goldens/<scene>/`; an existing golden is compared with per-channel/perceptual thresholds (mismatch writes a `<name>.diff.png` heatmap and fails); a missing golden is an explicit failure with regeneration instructions, never an auto-bless. `SCRIPTBOTS_CAPTURE_CORRUPT=1` enables the alarm-test corruption mode (blacked-out lighting) used to prove a broken pipeline fails the harness.
 - `--storage {file|memory}`: select the FrankenSQLite target. `file` exclusively reserves `SCRIPTBOTS_STORAGE_PATH` or a fresh generated run path and refuses existing databases or stale sidecars; `memory` opens volatile `:memory:` through the same engine.
 - `--recover-storage FILE`: exclusively reopen a validated existing ScriptBots run, replay/finalize its durable outbox, print the admitted/applied/durable watermarks, and exit. This repairs persistence only; it does not reconstruct the in-memory world or resume simulation ticks.
 - Auto-pause (any renderer):
   - `--auto-pause-below COUNT` (or `SCRIPTBOTS_AUTO_PAUSE_BELOW`) pauses when population ≤ COUNT
   - `--auto-pause-age-above AGE` (or `SCRIPTBOTS_AUTO_PAUSE_AGE_ABOVE`) pauses when any agent’s age ≥ AGE
   - `--auto-pause-on-spike` (or `SCRIPTBOTS_AUTO_PAUSE_ON_SPIKE=true`) pauses on first spike hit event

### Environment variables (quick reference)
- `RUST_LOG` — logging filter (e.g., `info`, `trace`, `scriptbots_core=debug`).
- `RAYON_NUM_THREADS` — set simulation thread pool size when `parallel` is enabled.
- `SCRIPTBOTS_MODE` — `auto|gui|bevy|terminal` (renderer selection).
- `SCRIPTBOTS_FORCE_TERMINAL` / `SCRIPTBOTS_FORCE_GUI` — override Auto-mode renderer detection (`1|true|yes`); explicit `--mode` remains authoritative.
- `SCRIPTBOTS_BOOTSTRAP_TICKS` — explicit pre-frontend science ticks (default `0`, equivalent to `--bootstrap-ticks`; set a nonzero value only when an intentional warmup is part of the run).
- `SCRIPTBOTS_TERMINAL_HEADLESS` — render TUI to an in-memory buffer for CI smoke tests.
- `SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES` — number of frames to render in headless mode (default 12; max 360).
- `SCRIPTBOTS_TERMINAL_HEADLESS_REPORT` — file path to write a JSON summary from a headless run.
- `SCRIPTBOTS_MAX_THREADS` — preferred maximum thread budget; core will cap Rayon to min of CPUs and this value (used unless `RAYON_NUM_THREADS` is already set).
 - `SCRIPTBOTS_TERMINAL_EMOJI` — force emoji mode `1|true|yes|on` or disable with `0|false|off|no`.
 - `SCRIPTBOTS_RENDER_SAFE` — force conservative rendering path in GUI mode (also enabled by `--renderer-safe` or `--low-power`).
 - `SCRIPTBOTS_RENDER_WATERMARK` — overlay a tiny diagnostics watermark in the GUI canvas (also enabled by `--debug-watermark`).
- `SCRIPTBOTS_RENDER_QUALITY` — `auto|potato|low|medium|high|ultra` for `render.quality` (the `--quality` CLI flag outranks it).
- Legacy render knobs `SB_WGPU_TONEMAP|EXPOSURE|BLOOM|BLOOM_THRESH|BLOOM_INTENSITY|VIGNETTE|FOG|FOG_COLOR|FXAA` and `SCRIPTBOTS_TERMINAL_PALETTE` are mapped onto the typed `render.*` schema at startup with one INFO log per applied mapping; the canonical typed `SCRIPTBOTS_RENDER_*` variables and CLI flags outrank them.
- `SCRIPTBOTS_CONFIG_OVERRIDES` — inline TOML document merged as the most-general environment configuration layer (e.g., `world_width = 1000`). Beats `--config` files, loses to the typed `SCRIPTBOTS_*` variables and to every CLI flag; malformed content fails startup closed. Each applied layer appends a kind-tagged digest to the manifest, and cross-layer displacements land in the manifest's `config_overrides`.
- `SCRIPTBOTS_RNG_SEED` — environment-layer RNG seed; `--rng-seed` outranks it as a distinct CLI layer with its own provenance.
- `SCRIPTBOTS_NEUROFLOW_ENABLED` — `true|false`.
- `SCRIPTBOTS_NEUROFLOW_HIDDEN` — comma-separated hidden sizes (e.g., `64,32,16`).
- `SCRIPTBOTS_NEUROFLOW_ACTIVATION` — `tanh|sigmoid|relu`.
- `SCRIPTBOTS_CONTROL_REST_ADDR` — REST bind address (default `127.0.0.1:8088`).
- `SCRIPTBOTS_CONTROL_SWAGGER_PATH` — Swagger UI path (default `/docs`).
- `SCRIPTBOTS_CONTROL_REST_ENABLED` — `true|false`.
- `SCRIPTBOTS_CONTROL_MCP` — `disabled|http` (default `http`).
- `SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR` — MCP HTTP bind address (default `127.0.0.1:8090`).
- Control-server environment is validated before startup side effects. Malformed or non-Unicode control values fail closed; `https://` in either MCP transport/address setting is rejected because the embedded MCP listener is plaintext HTTP rather than silently claiming TLS.
- `SCRIPTBOTS_STORAGE_PATH` — optional new-run FrankenSQLite file path. Without it, ScriptBots creates a unique `runs/scriptbots-<unix-ms>-<pid>.sqlite`. In either case the app reserves the path with create-new semantics and refuses an existing database or stale `-wal`, `-shm`, `-journal`, `-wal-fec`, or lock sidecar. The selected path is printed as `Run database: ...`; save that exact value for later reads and exports.
- `SCRIPTBOTS_RECOVER_STORAGE` — existing file-backed run to repair and finalize, equivalent to `--recover-storage FILE`. Recovery exits after persistence repair; it does not resume the simulation. The core science checkpoint is a separate persistence-disabled API, while application run-bundle discovery and resume remain roadmap work.

### Dual-window mode (GUI)
- ScriptBots opens two GPUI windows as one transactional launch: a canvas window rendering the world and a HUD window with controls, charts, and inspector. If either window cannot be created (for example because of window-manager or remote-session limits), ScriptBots terminates the partial application lifetime and returns an actionable error; it never silently changes the requested layout. GPUI uses `QuitMode::LastWindowClosed`, and closing either member of the paired session closes the application rather than leaving one orphaned window or a hidden process alive.
- The paired session has one GPUI simulation driver that runs independently of painting; neither window advances the world from its render path, and pause/speed changes from either window update the same shared state. This fixes GPUI's double-drive and per-view control defects. The remaining `HostCore` migration is architectural rather than a second GPUI driver fix: it must move scientific-time and command-drain ownership out of the renderer-local adapter and into the renderer-neutral host shared by every frontend and live transport.

## Simulation overview
Deterministic, staged tick pipeline (six seeded, domain-separated RNG streams; explicit staged ordering):
1. Aging and scheduled tasks
2. Food respawn/diffusion
3. Sense (spatial index snapshot)
4. Brain tick
5. Actuation (double-buffered state)
6. Food intake/sharing (deterministic reductions)
7. Combat and death (queued → commit)
8. Reproduction (mutation/crossover) in stable order
9. History and persistence projection (batched to the bounded FrankenSQLite worker)
10. Preserve the final persistence tail and reset transient runtime flags

### Design principles & determinism
- **No undefined behavior**: workspace lints flag `unsafe`; the remaining platform/environment boundary blocks are explicit and reviewable.
- **Explicit order of effects**: floating-point reductions and removals are staged. Before each successful tick, the dense SoA is normalized to ascending stable `AgentUid`; stable compaction and monotonic insertion preserve that order through death and spawn commits. Physical slot allocation therefore cannot choose reduction, neighbor, parent, or child priority.
- **Restorable domain and agent-keyed RNG state**: the world owns six independently derived, versioned streams—Environment, Food, Population, Lineage, Mutation, and Crossover—plus a versioned agent-substream protocol and UID-keyed continuation counters. Agent operations derive isolated streams from the root seed, stable `AgentUid`, operation tag, and agent-local ordinal; offspring operations derive from ordered parent UIDs plus the primary parent's local birth ordinal. The global demographic `AgentIdentity::birth_ordinal` remains a separate run-wide lifecycle sequence and is never used as the offspring RNG identity.
- **Transactional stochastic ownership**: a reproduction attempt claims its parent's local attempt continuation, and an admitted child claims that parent's local birth continuation. Child body, runtime, brain genome, and evaluator-state operations each receive one distinct derived stream. A rejected transaction restores the exact counter preimage together with energy, reproduction progress, population inserts, and other staged state, so a failed child cannot consume randomness that a retry or unrelated agent would observe.
- **Feature-gated parallelism**: `scriptbots-core` defaults to `parallel` (Rayon), while web builds disable it for single-thread determinism.
- **Completed-boundary outcome and admission seam**: `WorldState` owns deterministic accumulators plus a payload-free `Open`/`Pending`/`Sealed` boundary marker; it no longer owns a sink, retry payload, acknowledgement error, or admission watermark. A one-lifetime `PersistenceAdmissionSession` owns those external concerns. Its step APIs stage the exact immutable `Arc<PersistenceBatch>` before returning, retain that same allocation across definite and indeterminate failures, and seal the world only after acknowledgement. Direct `WorldState` stepping is intentionally limited to persistence-disabled worlds. Application-owned drivers route TUI, GPUI, Bevy, headless, profiling, and WASM ticks through the matching session without exposing session state to renderers. Completed population faults still travel beside the full `StepOutcome`; the clock-free traced outcome keeps the same boundary, and profiled session stepping retains the reviewed v2 timing contract. The hidden playback transport queue is also retired: application drains return one ordered `Vec<ControlCommand>`, while core returns normalized playback as an explicit driver-owned disposition.

#### Canonical digest and first-divergence trace

- `WorldState::world_digest_v1()` emits the exact `scriptbots.world-digest.v1.6`/codec-6 science-state wire. Every agent transition is canonicalized by stable `AgentUid`, so physical slot/dense allocation is no longer science state and the temporary V1.1 execution-order lane is retired. The aggregate covers stable-UID agent bodies, brain genomes/evaluator coverage, food, terrain/hydrology, all six restorable RNG-domain checkpoints with per-domain diagnostics, the exact `AgentSubstreamProtocolV1`, UID-ordered `AgentRngCounterStateV1` rows carrying `AgentRngCountersV1`, global identity counters, legacy factory-state digests or protocol adapter semantic identities, the selected locomotion model and remaining scientific configuration, active effects, derived transition caches, and open ancestry origins. Product-admitted MLP, DWRAON, and Assembly adapters therefore report complete construction-semantics coverage. Operational rendering, analytics, persistence-admission policy, host receipts, and backend error prose do not enter the science digest.
- `WorldState::checkpoint_v1()` captures the bounded canonical `scriptbots.world-checkpoint.v1.3`/codec-5 `postcard+blake3-v5` core science envelope only when persistence is disabled and the world is at an open completed boundary. It persists the exact agent-substream protocol, selected locomotion model, and one counter object per stable UID. Decode and restore reject missing, duplicate, reordered, mismatched, or incompatible counter/protocol state before constructing any evaluator or agent. `WorldCheckpointV1::decode()` also rejects oversized, trailing, noncanonical, checksum-invalid, and semantically malformed bytes; `WorldState::restore_checkpoint_v1()` allocates fresh physical handles from stable `AgentUid` order and reconstructs full genome/evaluator state through a caller-prepared exact `BrainRegistry`. The registry's complete roster, including its next allocation key and every protocol adapter's exact `BrainAdapterIdentityV1`, must match before reconstruction. The identity is a family-authored semantic attestation, not executable-byte authentication: construction/evaluation changes must change it, while payload interpretation changes must additionally bump the family schema/codec. The checkpoint never deserializes executable adapters and does not claim to restore storage ownership, analytics/history, configuration audit provenance, UI/render state, or an application run bundle. Its unkeyed BLAKE3 checksum detects corruption and canonical drift; it is not authentication.
- `WorldState::step_traced()` is an opt-in, clock-free diagnostic API using `scriptbots.world-step-trace.v1.6`/codec-6. It records exactly six semantic checkpoints—Sense, Brains, Actuation, Food, DeathCleanup, and Population—with separate world, ordered deferred-work, pending output-tail, and resource-ledger hashes. The embedded world and transition lanes bind the same agent-substream protocol, UID-ordered counters, selected locomotion model, and queued offspring identities as the final digest. The trace has its own aggregate hash, typed capture failures, and `validate_contract()` enforcement for schema, cardinality, order, ticks, coverage, digest shape, and nested aggregate hashes.
- Both V1 and trace validation prove protocol self-consistency, not artifact authenticity. Their FNV-1a64 hashes are diagnostic first-divergence aids, not signatures or collision-resistant attestations; a durable run bundle will require separate provenance/authentication.
- DeathCleanup and Population retain the ordered queue state immediately before consumption, while the final completed-boundary V1 and resource report prove traced/untraced parity after finalization. The literal six-point golden is enabled only by the `SCRIPTBOTS_WORLD_DIGEST_GOLDEN=1` environment guard in its pinned DSR test lane, so ordinary `--all-features` tests remain portable; there is no product CLI or replay-file claim yet.

#### Reproducible runs (seed control)
- Set a fixed seed in config: `rng_seed = <u64>`. At runtime you can apply via REST:
  ```json
  { "rng_seed": 42 }
  ```
- The root seed deterministically derives all six domain streams and every stable agent/offspring substream. The canonical launch manifest is `scriptbots.run-manifest.v3.3`: it records the six-domain checkpoint, exact `AgentSubstreamProtocolV1`, and UID-ordered `AgentRngCounterStateV1` launch rows. Once the tick-zero V1.6 start digest is bound, the bootstrap form is `scriptbots.run-manifest.v3.5`; it must bind the same root, protocol, counters, registry semantics, locomotion model, and `WorldDigestV1` evidence rather than describing a different launch state. Neither form collapses the domains back into one ambiguous global stream.
- The `bd-1kxd` keyed-substream schema and the `bd-2i1` V1.6 locomotion contract are closed with pinned DSR evidence. The reviewed exact-class full performance baseline was promoted byte-for-byte in `39fc2f0`, and the final same-class comparison passed in DSR `0.1.0-bd2i1-perf-compare-quiet1.20260716T160817Z`. `bd-hiv1` remains open for the separate movement-noise and spike-speed consumers that must use legacy wheel effort rather than physical displacement.
- For CPU thread control during profiling, prefer the standard `RAYON_NUM_THREADS` env var.

### Data model & spatial indexing
- **SoA layout**: agents use cache-friendly columns (`AgentColumns`) for fast scans during sense/actuation.
- **Two identity layers**: slotmap-backed `AgentId` prevents stale references; monotonic `AgentUid` is the stable scientific identity used by digests, lineage, replay, and persistence.
- **Spatial index**: uniform hash grid. Sense builds a read-only snapshot; the declared `rstar`/`kd` features do not yet provide alternate implementations.

### Sensors & outputs
- **Sensors**: multi-eye vision cones (angular), smell/sound/blood channels with attenuation, temperature discomfort, and clock/age cues.
- **Outputs**: wheel velocities integrated by the exact legacy two-rotation model by default, with conventional differential drive selectable for experiments; color/indicator pulses, spike length easing, give intent (altruistic food sharing), boost control, and sound output.
- **Mapping**: outputs drive physics and side-effects (e.g., spike damage scales with spike length and speed) and are logged for analytics.

### Brains & evolution
- **Brain trait** with `tick`/`mutate`/`crossover`; implementations include MLP and DWRAON by default, plus experimental Assembly.
- **Brain registry**: per-run registry attaches runners by key, enabling hybrid populations and runtime selection. Random spawns draw from `BrainRegistry::random_key` for mixed-species runs; sexual crossover is gated to same-kind brains (species barrier). Brains can optionally expose activation snapshots for visualization.
- **Genome & genetics**: genomes capture topology/activations; mutation/crossover create hybrid births with lineage tracking and tests.
- **NeuroFlow** (optional): deterministic CPU MLP with runtime toggles; seed-stable outputs verified in tests.

### Environment: food, terrain, temperature
- **Food dynamics**: configurable growth, decay, diffusion, and fertility capacity; speed-based intake and reproduction bonuses mirror legacy behavior.
- **Topography**: tile-based terrain/elevation influence fertility and movement energy (downhill momentum/energy costs).
- **Temperature**: gradient and per-agent preference drive discomfort drains; exposed in config and analytics.
- **Closed worlds & seeding**: enforce closed ecosystems; maintain population floors and scheduled spawns.

### Procedural maps (WFC sandbox; in progress)
- The core ships a rule-based Wave Function Collapse (WFC) generator that produces deterministic terrain (`TerrainLayer`) and optional fertility/temperature fields from a tileset spec. This enables quick scenario bootstrapping and repeatable experiments.
- Status and usage live in the plan doc; upcoming surfaces include REST/CLI endpoints to generate/apply artifacts. The hydrology system builds atop terrain for dynamic water flows (see below).

### Hydrology snapshot (experimental)
- Runtime hydrology state models per-cell flow direction, accumulation, basins, and a water depth field. Use `GET /api/hydrology` for a snapshot with:
  - `width`, `height`, `total_water_depth`, `mean_water_depth`, flooded cell counts with thresholds
  - Arrays: `water_depth`, `flow_directions` (N/S/E/W/-), `basin_ids`, `accumulation`, `spill_elevation`
- CLI: `scriptbots-control hydrology` prints a summary (ratios, thresholds, array sizes). Hydrology integrates with terrain and is deterministic per seed.

### Combat & mortality analytics
- **Spikes**: damage scales with requested spike length and agent speed; collision resolution is staged for determinism.
- **Carcass sharing**: meat distribution honors age scaling and diet tendencies; events persisted for analysis.
- **Analytics**: attacker/victim flags (carnivore/herbivore), births/deaths, hybrid markers, age/boost tracking, and per-tick summaries feed FrankenSQLite plus the immutable HUD analytics snapshot.

## Rendering & UX
- GPUI window, HUD, and canvas renderer for food tiles and agents (circles/spikes). The dual-window HUD and simulation canvas launch transactionally under `QuitMode::LastWindowClosed`; closing either paired window ends the session, so a launch/close failure cannot leave a degraded or orphaned UI. The world window is read-only while the HUD temporarily remains the sole driver pending `HostCore`.
- Camera controls: pan/zoom; keyboard bindings for pause, draw toggle, speed ±.
- Overlays: selection highlights, diagnostics panel; charts and advanced overlays are staged in the plan.
- Functional search: the settings panel includes a live search bar that filters parameters across all categories via a centralized filter, making it fast to find and tweak knobs.
- Inspector: per-agent stats and genome/brain views (scoped to plan milestones); mutation-rate adjusters (±) for primary/secondary let you tweak an agent’s evolution parameters live.
- Optional audio via `kira` (feature `audio`).

### GPUI performance & diagnostics
- Adaptive GPU adapter selection; viewport culling for terrain; chart decimation; batched path rendering to reduce draw calls.
- Troubleshooting flags: `--renderer-safe` (conservative paint path) and `--debug-watermark` (tiny on-canvas badge) help isolate rendering issues.

### Accessibility & input
- **Colorblind-safe palettes** (deuteranopia/protanopia/tritanopia) and a high-contrast mode; UI elements and overlays respect palette transforms.
- **Keyboard remapping** with conflict resolution and capture mode; discoverable bindings in the HUD.
- **Narration hooks** prepared for future screen-reader integration; toggles surfaced in the inspector.

### Renderer abstraction
- The app selects GPUI, Bevy, or terminal renderers via `--mode {auto|gui|bevy|terminal}` (subject to build features). Every frontend consumes the same world and immutable analytics snapshots.

### Keyboard shortcuts (GUI)
- Playback: `space` pause/resume, `+`/`-` speed up/down, `s` single-step
- Views: `d` toggle drawing, `f` toggle food overlay, `Ctrl+Shift+O` toggle agent outlines
- Spawning: `a` add crossover agents, `q`/`h` spawn carnivore/herbivore
- Persistence safety: a spawn shortcut is refused after the current tick is admitted under the selected storage guarantee; advance to the next open tick boundary before spawning. An unresolved admission is reported separately and must be resolved by retrying the exact retained batch before either the world or an external arrival can advance. Both typed reasons are logged without consuming simulation RNG.
- World: `c` toggle closed environment, `o` follow oldest, `s` follow selected
- Accessibility: `p` cycle color palettes (with keyboard rebinding support)

### Audio system
- Optional `kira`-backed mixer (feature `audio`) with event-driven cues (births, deaths, spikes) and accessibility toggles.
- Channels planned for ambience/effects; platform caveats apply on Linux/WSL2. Audio is disabled in wasm; use Web Audio API from JS if needed.

### Terminal mode
The implemented Ratatui frontend provides an emoji-rich dashboard and headless snapshot mode. Automatic selection chooses it only before launch; an explicit GPUI/Bevy request or a native window launch failure is surfaced rather than silently changing products.

#### Terminal-only mode
- Force the emoji TUI renderer (useful on headless machines):
  ```bash
  SCRIPTBOTS_MODE=terminal cargo run -p scriptbots-app
  ```
- Auto selection: `SCRIPTBOTS_MODE=auto` (default) chooses a compiled native renderer only for a real native graphical session and otherwise starts the terminal (including macOS SSH sessions).
- Auto-mode policy overrides:
  - `SCRIPTBOTS_FORCE_TERMINAL=1` → choose terminal in Auto mode even when a display server is present.
  - `SCRIPTBOTS_FORCE_GUI=1` → require compiled GPUI in Auto mode even if no display variables are set; failures remain visible.
- CI/headless smoke runs can bypass raw TTY requirements by setting `SCRIPTBOTS_TERMINAL_HEADLESS=1`, which drives the renderer against an in-memory buffer for a few frames.

- Emoji mode (terminal renderer):
  - Defaults ON when a modern UTF‑8 terminal is detected; press `e` to toggle at runtime.
  - Force enable via env: `SCRIPTBOTS_TERMINAL_EMOJI=1|true|yes|on`; force disable with `0|false|off|no`.
  - Heuristic: enabled if `TERM` is not `dumb/linux/vt100`, locale contains `utf-8|utf8`, and `CI` is unset.
  - Emoji mappings: terrain `🌊/💧/🏜/🌿/🌺/🪨` (lush swaps: `🐟`, `🌴`, `🌾`, barren `🥀`); agents single `🐇/🦝/🦊`, small groups `🐑/🐻/🐺`, large cluster `👥`, boosted `🚀`, spike peak `⚔` (underline). Heading arrows remain for single agents when available.
  - If emojis render as tofu/misaligned, install an emoji-capable font (e.g., Noto Color Emoji) or toggle off with `e`.
- Narrow symbols mode: press `n` to switch to width-1 friendly symbols while keeping emoji colors off-background; helpful for strict terminals/alignment.

Keybinds: space (pause), +/- (speed), s (single-step), b (toggle metrics baseline), S (save ASCII screenshot), e (emoji), n (narrow symbols), x (expanded panels), ?/h (help), q/Esc (quit). The terminal HUD shows tick/agents/births/deaths/energy, Insights (rolling metrics), Mortality panel, Brains leaderboard, recent events log, and an emoji world mini-map. The layout is responsive and auto-expands panels on wider terminals; press `x` to toggle. Screenshots saved via `S` are written under `screenshots/frame_<tick>.txt`.

## Storage & analytics

- **One embedded engine:** ScriptBots uses the public `fsqlite` facade from FrankenSQLite with `version = "=0.1.16"`, pinned to immutable revision `e536d7f8ca102b3eb5236bef48514582379f9346` at `https://github.com/Dicklesworthstone/frankensqlite`. The current dependency enables `native` with default features disabled and provides create-free existing-file open plus expected-identity verification before recovery can read or mutate database bytes.
- **Two storage targets:** `--storage file` exclusively reserves `SCRIPTBOTS_STORAGE_PATH` or a generated `runs/scriptbots-<unix-ms>-<pid>.sqlite` and prints the selected run database; it refuses reuse or stale sidecars. `--storage memory` opens volatile `:memory:` through the same implementation.
- **Explicit interrupted-run recovery:** `--recover-storage FILE` (or `SCRIPTBOTS_RECOVER_STORAGE`) is the only application path that opens an existing run database for mutation. It holds the OS writer lease, binds recovery to the identity of the already-open VFS handle, verifies the exact structural schema fingerprint plus the supported migration sequence and persistence invariants, and refuses missing, replaced, unrelated, symlink, and multiply-linked files before replaying admitted-but-unapplied outbox rows and finalizing applied rows. It prints the resulting watermarks and exits. This is persistence repair, not world resume.
- **Thread-confined, single-writer connection:** `fsqlite::Connection` is deliberately `!Send + !Sync`. The storage worker creates, uses, explicitly closes, and drops its connection on that worker thread. File writers hold a nonblocking OS advisory lease on a persistent companion lock file; process-local path/inode tracking is only defense in depth. No connection-owning value is shared through `Arc<Mutex<_>>`.
- **Bounded admission, distinct proof levels:** persistence batches enter a bounded queue. Configurable `StorageDeadlines` bound startup, admission-gate, command-enqueue, receipt, flush, and shutdown acknowledgement waits, but cannot cancel a database call already executing on the owner thread or bound the supervised reaper. Validation, closed-gate, queue-send, and rolled-back outbox failures are definitely `NotAdmitted`; the external admission session retains the exact completed batch and acknowledgement fault while the world's payload-free `Pending` marker blocks later science ticks until explicit retry succeeds. A lost or timed-out acknowledgement remains typed as `Indeterminate` at the world boundary, but retrying the unchanged canonical payload is idempotent and reuses its stable batch ID; a conflicting payload is rejected by its BLAKE3 identity. Timed-out shutdown retains the exact pending receipt and worker handle for retry; dropping the controller hands both to an independent supervised reaper rather than abandoning connection ownership. `submit_with_receipt` returns the batch ID after the exact payload enters the worker outbox and reports `Durable` for a file database or `CommittedVolatile` for memory. That receipt proves admission, not scientific-table application. Same-thread `Storage::persist` now follows the identical outbox identity and watermark protocol; exact duplicates remain idempotent, conflicting payloads are typed refusals, raw insert SQL is confined to a non-production conformance test, and receipt plus shutdown/join retain the same typed terminal cause (`bd-2z0.8.9.4.4`).
- **Durable recovery and watermarks:** each file-backed batch advances three monotonic, separately queryable prefixes: `admitted` after the outbox transaction, `applied` in the same transaction as all scientific-table rows, and `durable` in a later marker transaction that permits outbox-payload compaction. Startup replays admitted-but-unapplied payloads in order and finalizes applied-but-not-durable batches without duplicating rows. Exact duplicate retries reuse the original batch identity; a different payload for an already admitted tick is rejected. Flush and shutdown receipts include all three watermarks.
- **V8 scientific evidence (centralized DSR pending):** every durable scientific sequence has an archive-bound projection batch, including honest zero-row boundaries. Exact births, exact deaths, and nonzero aggregate-combat counters retain deterministic local order. Bounded typed evidence/pages reject sequence gaps, duplicate or corrupt ordinals, fabricated cursors, and an expected-but-empty scenario with typed `NoEvidence`; pairwise attacker/victim edges remain `bd-2z0.5.9`.
- **V9 command evidence (centralized DSR pending):** normalized records retain the exact command envelope, source/client identity, optional admission order, all three revision guards, terminal boundary, and canonical archive digest. Runtime `admitted`/`applied`/`rejected`/`failed` transitions remain distinct from storage `committed_volatile`/`durable` transitions. Finished files expose bounded `StorageReader::command_journal_evidence`, `command_journal_record`, and `command_journal_page`; a nonempty pre-V9 archive is refused before migration because archive v1 cannot supply missing lifecycle evidence.
- **Complete typed ancestry origins:** every agent insertion emits exactly one immutable `born`, `seeded`, or `injected` origin row under a globally unique stable agent UID. Only `born` rows have a global demographic `AgentIdentity::birth_ordinal` or contribute to demographic birth totals; ancestry and offline rebuilds consume all three origins plus exact death causes. Natural-offspring RNG instead uses the primary parent's local `birth` continuation in `AgentRngCountersV1`, which is directional lineage state and may share the same numeric value in different families without collision because the parent UID also participates. Completing any scientific tick while persistence is disabled creates a history gap, even when that tick has no lifecycle or replay rows, because its summary, metrics, and snapshot were not admitted. That world refuses to re-enable persistence; start a new world and storage identity instead of creating a run database with a hidden interval.
- **Lock-free frontend reads:** the worker atomically publishes immutable `Arc<AnalyticsSnapshot>` latest-value state. GUI, TUI, and API consumers load it without a mutex and may skip stale snapshots; they never run SQL while rendering.

### Tables and query examples

The V6 base schema run-scopes scientific and operational tables for matched-seed experiments. V7 adds the canonical host archive/ledger, V8 adds per-scientific-boundary domain evidence, and V9 adds normalized command records with separate application/storage transitions. Unsupported pre-release data is refused rather than guessed or destructively rewritten. Use the bounded typed command/domain readers above for journal evidence; the SQL below illustrates run-scoped scientific reporting only.

Representative SQLite-compatible query shapes include:

```sql
-- Latest metrics snapshot
select name, value
from metrics
where run_id = ?1
  and tick = (select max(tick) from metrics where run_id = ?1)
order by name;

-- Top predators by average energy
select agent_id,
       avg(energy) as avg_energy,
       max(spike_length) as max_spike_length,
       max(tick) as last_tick
from agents
where run_id = ?1
group by agent_id
order by avg_energy desc
limit 10;

-- Event totals for a run report
select kind, sum(count) as total
from events
where run_id = ?1
group by kind
order by kind;
```

JSON replay payloads are validated by ScriptBots and stored as ordinary `TEXT`. Integer flags are decoded at the storage boundary into Rust booleans.

### Pipeline and maintenance

- A persistence transaction either commits the entire accepted batch or rolls it back; a failed statement may not leave the connection in an active transaction.
- The published health snapshot exposes revision, last committed tick, committed agent count, admitted/applied/durable batch watermarks, structured last failure, and stopped state. GUI/TUI code derives lag from live and committed ticks without issuing SQL. Queue depth is not published yet.
- The same-thread `Storage` boundary provides explicit flush, close/checkpoint, and `VACUUM` operations. The asynchronous pipeline currently exposes flush and shutdown barriers; maintenance must stay on the connection-owning worker and never run on a GUI/TUI paint path.
- The exact-revision file-backed conformance test closes and reopens the database and verifies committed data and integrity. Dedicated durability and concurrent-reader/writer gates must use independent connections rather than sharing one connection across threads.

## Development workflow
- **Coding standards**: See `RUST_SYSTEM_PROGRAMMING_BEST_PRACTICES.md`. Embrace `Result`-based errors, clear traits, and avoid `unsafe`.
- **Linting**: `cargo clippy --workspace --all-targets --all-features -W clippy::all -W clippy::pedantic -W clippy::nursery`
- **Formatting**: `cargo fmt --all`
- **Tests**: `cargo test --workspace` (simulation and GPUI tests will be added as systems land)
- **Profiles**: Release uses LTO, single codegen unit, and abort-on-panic for optimal binaries.

### Authoritative tracker triage

`br sync --flush-only` exports the tracked source of truth to
`.beads/issues.jsonl`. Do not invoke BV directly in this repository: BV can
silently prefer the separate `.beads/beads.jsonl` snapshot. Run every
data-bearing robot command through the fail-closed integration instead:

```bash
scripts/bv_authoritative.sh --robot-triage
scripts/bv_authoritative.sh --robot-plan  # fails closed if BV and BR claim sets diverge
scripts/bv_authoritative.sh --robot-insights | jq '.status'
scripts/test_bv_authoritative.sh
```

The wrapper creates a unique external read-only view, forces JSON robot mode,
and emits a result only after BR's ID/status/dependency-count projection agrees with the
tracked export, BV's issue/status/blocking-edge counts and `data_hash` agree,
BV's native actionable count agrees with its complete plan, and that plan
covers every BR-ready issue. Because BV 0.16.0 includes in-progress work and
uses different hierarchy semantics, its native actionable set need not equal
`br ready`. Every BV next result and triage top pick must be BR-ready; an
unscoped plan must equal the BR-ready set and a scoped plan must be its subset.
Unsafe claim-oriented output fails closed. Graph-ranked recommendations may be
blocked and never authorize a claim; `br ready` remains the sole actionability
and claim authority. The wrapper refuses
caller-selected databases/workspaces, missing or empty exports, non-JSON graph
output, every unrecognized or mutating BV option, and a source export that
changes during analysis. Historical `--as-of` sources are refused. Neither
repository snapshot is overwritten or deleted.

BV history and diff are also refused until their handlers honor the explicitly
isolated tracker source; otherwise a stale checkout snapshot can contaminate the
result.

## Testing & verification
- **Core tests**: unit and property tests for reproduction math, spike damage, food sharing/consumption; determinism tests run seeded scenarios and assert stable summaries.
- **Render tests**: GPUI compile-time view tests; terminal HUD headless smoke tests (`SCRIPTBOTS_TERMINAL_HEADLESS=1`).
- **Benchmarks**: `criterion` harness for ticks/sec at various agent counts.
- **Batch verification**: the pinned Doodlestein Self-Releaser (`dsr`) profile is the only build, lint, test, WASM, and release evidence lane. Hosted workflow results are not used.

## Performance & profiling
- CPU profiling (Linux/macOS): run with `RUSTFLAGS='-g'` and use `perf record`/`perf report` or `dtrace`/Instruments; annotate hot paths in sense/actuation.
- Tracy (optional): integrate client in dev builds to visualize frame times and background worker activity.
- Threading: tune `RAYON_NUM_THREADS` to match physical cores; verify determinism with seeded runs.
- Rendering: measure HUD/canvas frame times; avoid per-frame allocations; prefer batched path building.
 - Built-in tools:
   - `--profile-steps N` and `--profile-storage-steps N` to run headless micro-benchmarks
   - `--profile-sweep N` and `--auto-tune N` to explore and auto-pick thread/flush settings
   - `--renderer-safe` for a conservative paint path; `--debug-watermark` overlays a diagnostics badge
 - Low-power mode: in addition to capping threads (`--low-power`), the app lowers OS process priority (Unix niceness +10; Windows BELOW_NORMAL) to be a better background citizen.

## Tracing & logging
- Logging uses `tracing` with `RUST_LOG` filters (e.g., `RUST_LOG=info,scriptbots_core=debug`).
- Categories of interest:
  - `scriptbots_core::world` — tick summaries, seeding, closed/open flips
  - `scriptbots_storage` — transaction attempts, structured worker errors (including commit state), committed-tick publication, and flush/shutdown receipts. The GUI/TUI derive live-versus-committed lag from the published snapshot; no queue-depth metric is exposed yet.
  - `scriptbots_app::servers` — REST and MCP server lifecycle, tool invocations
  - `scriptbots_render` — window lifecycle, input bindings
- Prefer structured fields (e.g., `tick = summary.tick.0`) for machine-readable logs. Avoid panics in production; release profile uses `panic = abort`.

## Runtime control surfaces

For an interactive run, ScriptBots parses the control environment and transactionally reserves every enabled REST/MCP socket before writing config output, tuning, changing process priority, constructing the world, or reserving FrankenSQLite storage. The runtime adopts those exact listeners. REST and MCP are supervised independently; an unexpected exit from either stops the other, publishes failed health, and causes the active TUI, GPUI, or Bevy frontend to return the preserved root error.

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
  - `POST /api/selection` → queue a selection update (modes: set/add/clear; optional state: none|hovered|selected)
  - `GET /api/presets` → list scenario presets
  - `POST /api/presets/apply` → apply preset by name
  - `GET /api/config/audit` → recent config patches (audit ring buffer)
  - `GET /api/scenario` → the launch scenario identity bound to this run
  - `GET /api/ws/stream` → WebSocket stream of tick summaries
  - `GET /api/narrative/search` → search the recent birth/death/combat narrative
  - Playback (fire-and-forget acknowledgement): `POST /api/pause`, `POST /api/resume`,
    `POST /api/step`, `POST /api/speed`, and `GET /api/status`
  - Playback (two-axis command status): `POST /api/control/pause`, `POST /api/control/resume`,
    `POST /api/control/step`, `POST /api/control/speed`, `POST /api/control/shutdown`,
    and `GET /api/control/status/{command_id}`

Every route above is registered in the published OpenAPI document, so `/docs` and
`/api-docs/openapi.json` are the authoritative, machine-readable version of this list.

Examples:
```bash
# Filter debug table by selected agents only, limit 20, sort by age
curl -s 'http://127.0.0.1:8088/api/agents/debug?selection=selected&limit=20&sort=age' | jq .

# Select a cohort (replace selection with ids [1,2,3])
curl -s -X POST http://127.0.0.1:8088/api/selection \
  -H 'content-type: application/json' \
  -d '{"mode":"set","agent_ids":[1,2,3]}'

# Add agents to current selection and mark them highlighted
curl -s -X POST http://127.0.0.1:8088/api/selection \
  -H 'content-type: application/json' \
  -d '{"mode":"add","agent_ids":[4,5],"state":"highlighted"}'
```

More examples:
```bash
# List and apply a preset via REST
curl -s http://127.0.0.1:8088/api/presets | jq
curl -s -X POST http://127.0.0.1:8088/api/presets/apply -H 'content-type: application/json' -d '{"name":"arctic"}' | jq .

# Recent events (tail)
curl -s 'http://127.0.0.1:8088/api/events/tail?limit=10' | jq .

# Scoreboard (top predators, oldest agents)
curl -s 'http://127.0.0.1:8088/api/scoreboard?limit=10' | jq .

# SSE tick stream (press Ctrl+C to stop)
curl -N -H 'Accept: text/event-stream' http://127.0.0.1:8088/api/ticks/stream | sed -n '1,10p'
```

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
# `--db` is required: paste the exact path printed as `Run database: ...` by the file-mode app.
# `--out` must name a fresh path; export refuses every existing file, symlink, or database sidecar.
RUN_DB='runs/scriptbots-1773264512345-4242.sqlite' # example shape; replace with the printed path
cargo run -p scriptbots-app --bin control_cli -- export metrics --db "$RUN_DB" --last 1000 --out latest_metrics.csv
# New commands:
cargo run -p scriptbots-app --bin control_cli -- presets
cargo run -p scriptbots-app --bin control_cli -- apply-preset arctic
cargo run -p scriptbots-app --bin control_cli -- screenshot --out screenshots/frame_0001.txt
cargo run -p scriptbots-app --bin control_cli -- screenshot --png --out screenshots/frame_0001.png
cargo run -p scriptbots-app --bin control_cli -- hydrology
``` 

Interactive dashboard:
```bash
# Live TUI dashboard of knobs and their current values; press 'q' to quit, 'r' to refresh
cargo run -p scriptbots-app --bin control_cli -- watch --interval-ms 500
```

### Scenario layering & replay CLI scaffold
- **Layered configs**: pass one or more `--config path/to/file.toml` (or `.ron`) flags—or set `SCRIPTBOTS_CONFIG` with semicolon-separated paths—to build scenarios from reusable fragments (e.g., `base.toml → arctic_biome.toml → evolution_study.toml`). Layers merge in order before env overrides, unlocking repeatable experiments without editing code.
- **Config inspection**: add `--print-config` to dump the merged configuration (default JSON) or `--write-config output.toml` to persist it; choose `--config-format json|toml|ron` and combine with `--config-only` for a dry run in CI/tooling workflows.
- **Replay status**: `--replay-db` and `--compare-db` load, re-simulate, and diff encoded event streams, but production does not yet emit a meaningful nonempty stream. Nonzero empty-vs-empty runs now fail closed instead of claiming verification; track `bd-2z0.8.9.8` for complete instrumentation and mock-free proof.
- **Storage helpers**: FrankenSQLite-backed accessors (`max_tick`, `load_replay_events`, `replay_event_counts`) underpin the CLI so analytics pipelines or external tools can reuse the same deterministic data.

### MCP HTTP server (Model Context Protocol)
- Default: `127.0.0.1:8090` over HTTP; disable with `SCRIPTBOTS_CONTROL_MCP=disabled`.
- Override bind address: `SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR=127.0.0.1:9090`.
- Implemented with `fastmcp-rust`; the server shares the same `ControlHandle` as REST.
- Tools exposed (13):
  - Configuration: `list_knobs` → array of knob entries; `get_config` → full config snapshot;
    `apply_updates` → accepts `{ updates: [{ path, value }, ...] }`; `apply_patch` → accepts `{ patch: { ... } }`
  - Scenarios: `list_presets` → available scenario presets; `apply_preset` → accepts `{ name }`
  - Playback: `pause`, `resume`, `step` (accepts `{ count }`), `set_speed`, `shutdown`
  - Observation: `get_status` → current simulation status; `get_command_status` → two-axis status by command ID
Notes: Only HTTP transport is supported here; stdio/SSE are not used.

MCP quickstart:
- Start the app; verify MCP binds on `127.0.0.1:8090` (override via `SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR`).
- Connect an MCP HTTP client to the endpoint and call `tools/list` for the authoritative roster
  and JSON schemas; the list above is the same set the server registers at startup.
- Each tool returns structured JSON; use your MCP-compatible agent to orchestrate parameter sweeps and log findings to FrankenSQLite.

## Configuration files & scenarios
- **Current state**: repeatable scenarios can layer TOML or RON files with repeated `--config` flags or the semicolon-separated `SCRIPTBOTS_CONFIG` variable; later layers override earlier ones before environment overrides.
  ```text
  base.toml → arctic_biome.toml → evolution_study.toml
  ```
- **Inspection/export**: use `--print-config`, `--write-config`, `--config-format`, and `--config-only` to inspect or persist the composed configuration without launching a renderer.
- **Value**: researchers gain repeatable, composable scenarios without hand-editing large monolithic files, making it easy to swap biomes, mutation parameters, or analytics presets on demand.

## Deterministic replay roadmap
- **Already implemented**
  - ✅ Event-log schema (`replay_events` table) and typed event encoding.
  - ✅ Type-safe encoding for replay artifacts shared by storage and analytics crates.
  - ✅ Storage plumbing to persist replay batches alongside standard metrics.
  - ✅ Headless runner and CLI comparison scaffold that refuses vacuous nonzero runs.
  - ✅ Persistence-disabled core science checkpoint with strict restore and next-transition proof; application discovery/resume remain planned.
- **Planned**
  - ❌ Production instrumentation for nonempty RNG/brain/action event streams.
  - ❌ Branch/diff workflows comparing Rust vs. Rust PR builds vs. the legacy C++ baseline.
  - ❌ Mock-free positive and perturbed-candidate E2E proofs.
  - ❌ FrankenSQLite-backed analysis views for quick triage of regressions and experiment outcomes.
- **Use cases**
  - Parity testing between the Rust port and the original C++ implementation.
  - Regression prevention by replaying critical seeds in CI or pre-merge checks.
  - Debugging elusive bugs by reproducing exact agent decisions tick-by-tick.
  - Long-running research experiments that demand bitwise-stable replays.

## Security & operations
- REST and MCP servers bind to loopback by default and are plaintext HTTP. If you expose them externally, terminate TLS in a separate trusted proxy and configure CORS appropriately; `https://` control settings fail closed rather than pretending the embedded listener implements TLS. The WASM path requires COOP/COEP headers only when enabling multithreading; single-thread builds avoid this.

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
- The live app still owns a bounded MPMC `CommandBus`; REST, MCP, CLI, and interim frontend drivers enqueue legacy `ControlCommand`s and drain them at simulation boundaries.
- The new `scriptbots-runtime` crate defines the replacement protocol surface: stable command identity, one admission order, optional typed control/scientific/config revision guards checked at the ordered application boundary, independent application/journal status axes, immutable snapshots, and independent event cursors.
- `HostCore` is now the implemented command/science authority, and the optional native runner supplies fixed-deadline wake, cancellation, bounded catch-up, and durability-gated shutdown around that exact same host. This does not claim the live app has migrated: later adapter beads move REST/MCP, GPUI, Bevy, TUI, headless, and WASM callers away from the legacy bus.

## Contributing
- Keep changes scoped to the relevant crate; prefer improving existing files over adding new ones unless functionality is genuinely new.
- Update docs where it helps future maintainers understand decisions and invariants.
- For larger tasks, update `PLAN_TO_REARCHITECT_AND_REVIVE_RUST_SCRIPTBOTS.md` inline to mark progress.

## WebAssembly (sibling crate plan)
We maintain a sibling browser-targeted crate, `scriptbots-web`, that reuses core crates without invasive changes. See `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` and `docs/wasm/` (ADRs, audits, capability matrix). Initial MVP runs single-threaded by disabling `scriptbots-core`’s `parallel` feature on wasm; WebGPU vs Canvas2D rendering is under evaluation.

> Quick peek: `crates/scriptbots-web/web/` ships a Canvas demo harness that consumes the wasm snapshots, surfaces live metrics, and can be served locally via `python -m http.server`. Binary snapshots (`snapshot_format: "binary"`) and custom seeding strategies are already wired in for experimentation.

Helpful docs:
- `docs/wasm/adrs/ADR-001-wasm-rendering.md` — rendering stack decision record
- `docs/wasm/adrs/ADR-002-browser-persistence.md` — browser persistence approach
- `docs/wasm/adrs/ADR-004-component-model.md` — component model/WASI Preview assessment
- `docs/wasm/browser_matrix.csv` — browser capabilities (WebGPU, SAB, SIMD)
- `docs/franken_integration.md` — franken-library integration program: verdicts, constraints, boundary rules (bd-2js6)
- `docs/licenses.md` — dependency license audit incl. the franken MIT+rider analysis

### WASM snapshot format & APIs
- `snapshot_format`: `json` (default) or `binary` (Postcard `Uint8Array`).
- APIs: `default_init_options()`, `init_sim(options)`, `tick(steps)`, `snapshot()`, `reset(seed?)`, `registerBrain("wander"|"mlp"|"none")`.
- Determinism: wasm-vs-native parity tests compare snapshots for fixed seeds; single-thread fallback is default (Rayon disabled).

### WASM hosting guide (COOP/COEP)
- For multithreading (future), browsers require SharedArrayBuffer with headers:
  - `Cross-Origin-Opener-Policy: same-origin`
  - `Cross-Origin-Embedder-Policy: require-corp`
- Local dev: serve with a static server that sets these headers (or use a service worker). For now, single-thread builds avoid the requirement.
- DSR: the pinned `rust_scriptbots` profile builds the WASM package and runs its browser parity checks; hosted workflow runs are not accepted as evidence.

## Licensing

ScriptBots is licensed under **`LicenseRef-MIT-OpenAI-Anthropic-Rider`** — MIT with an additional rider (see `LICENSE`): **no rights are granted to OpenAI, Anthropic, their affiliates, or parties acting for them**, and any distribution of the software or derivative works must include the rider unmodified. This is the same license the embedded FrankenSQLite/asupersync components carry, so the whole product ships under one uniform license (owner relicensing decision, 2026-07-13; previously the first-party code was declared `MIT OR Apache-2.0`). Release archives bundle `THIRD-PARTY-LICENSES.md` for embedded-component notices; full audit: `docs/licenses.md`.

## Credits
- Original ScriptBots by Andrej Karpathy (reference snapshot included under `original_scriptbots_code_for_reference/`).
- This Rust port is an independent, from-scratch implementation guided by parity goals and modern Rust/GPUI best practices.

## FAQ
- **What platforms are supported?** Linux, macOS, and Windows 11 are targeted. Windows is supported natively (MSVC toolchain) and via WSL2. Early UI milestones may see platform-specific polish arriving at different times.
- **Where do I start hacking?** `scriptbots-core` for the world model; `scriptbots-runtime` for host/client protocol work; `scriptbots-render` for the GPUI view; `scriptbots-brain` for brain interfaces; `scriptbots-storage` for persistence.

## Troubleshooting
- **Fresh clone fails inside the `fsqlite` git dependency** (`failed to update submodule legacy_sqlite_code/sqlite: object not found - no match for id 450a9009...`): the pinned frankensqlite revision records a sqlite submodule commit that is not reachable from any branch/tag on `github.com/sqlite/sqlite`. The checked-in `.cargo/config.toml` sets `[net] git-fetch-with-cli = true`, which routes submodule fetches through the git CLI (it retries by exact OID, and GitHub honors fetch-by-SHA), so current checkouts build on a fresh cache. If you build with a cargo older than 1.46 or deliberately bypass the repo config, pre-populate the submodule once in the failed checkout (`git init && git remote add origin https://github.com/sqlite/sqlite && git fetch --depth 1 origin 450a90097fabf4eac7568a9688b34278b4d72122 && git checkout FETCH_HEAD`, then set `submodule.legacy_sqlite_code/sqlite.update=none` in the checkout's `.git/config`).
- **MSVC/SDK link errors on Windows**: Ensure VS Build Tools "Desktop development with C++" and Windows 11 SDK are installed. Then run `rustup default stable-x86_64-pc-windows-msvc`.
- **Blank or crashing window**: Update GPU drivers. On WSL2, update the WSL kernel and try again. Verify that your system supports D3D12 (Windows) or Vulkan/Metal (Linux/macOS).
- **Storage admission or commit failure**: Compare the live tick with the published committed tick and compare the admitted, applied, and durable batch watermarks. Inspect the structured last error, operation, attempt, and commit state. A definite synchronous `NotAdmitted` fault pauses later science ticks and retains the exact batch for explicit retry. A file admission receipt proves that the exact payload can be recovered; application and terminal durability remain separately visible until the later watermarks advance.
- **Determinism regressions**: Ensure you haven't introduced unordered parallel reductions; stage results and apply in a stable commit phase.

## Releases
- Releases are built, verified, and published only through Doodlestein Self-Releaser (`dsr`) using the pinned `rust_scriptbots` repository profile. Use `dsr build --tool rust_scriptbots ...` for artifacts and `dsr release --tool rust_scriptbots ...` for publication; hosted workflow execution is never release evidence.
- The DSR release profile is responsible for producing and retaining the platform archives and their verification evidence. Inspect the DSR result bundle before publication, and publish that exact blessed bundle rather than rebuilding it through another system.
- **macOS codesigning:** configure the signing identity and certificate through DSR's protected release environment. Never put signing credentials in the repository or substitute a hosted workflow's secret store for the pinned DSR release path.
- DSR imports signing material into an isolated temporary keychain, signs binaries and `.app` bundles, repacks the archives, and retains the resulting checksums in the release evidence. Release operators should verify that exact bundle (`codesign --verify --deep` on macOS and `shasum -a 256` on every platform) before publication.

## Roadmap (condensed)
1. Core data structures and config (done); expand parity (metabolism, locomotion, food math, carcass sharing).
2. World mechanics and determinism under parallelism; spatial index tuning.
3. Brains: MLP shipped; DWRAON + Assembly (feature-gated) and NeuroFlow optional.
4. Storage: durable outbox, exact identity/schema recovery, direct-write/root-cause unification, and the V6/V7 base/archive lineage are integrated. V8/V9 command/domain projections are code-first and centralized-DSR-pending; remaining work includes strict-run host policy, pairwise interactions, product checkpoint/replay integration, and run bundles.
5. Runtime: renderer-neutral protocol, sole-owner `HostCore`, and fixed-deadline native lifecycle landed; next publish canonical multi-subscriber projections and migrate every frontend and transport adapter off the legacy app-owned drivers.
6. Rendering: HUD/overlays/inspector polish; performance diagnostics.
7. DSR packaging and verification: release builds, binaries, and WASM/browser evidence from the pinned repository profile.

### Mixed brain families (default)
- The app now registers multiple brain families by default (MLP, DWRAON, Assembly experimental, NeuroFlow) and seeds mixed populations automatically. Random spawns are bound to a sampled brain family.
- NeuroFlow is enabled in the default config; edit at runtime via REST/CLI or env (e.g., SCRIPTBOTS_NEUROFLOW_*).
- Sexual reproduction only occurs within the same brain kind (species barrier). Cross-kind parents fall back to random spawns. This allows fair A/B comparisons between families.
- To force a single brain for new agents, bind a chosen `brain_key` to agents via `WorldState::bind_agent_brain` or modify the seeding function to always pick that key.
