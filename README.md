## Rust ScriptBots

ScriptBots is a modern Rust reimagining of Andrej Karpathy’s classic agent-based evolution simulator. Our goal is a faithful, deterministic port with a GPU-accelerated UI, pluggable brain implementations, and first-class analytics. This is a multi-crate Cargo workspace separating simulation core, brains, storage, rendering, and the application shell.

For design intent and the living roadmap, see `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md` (the project “bible”). A sibling WebAssembly plan lives in `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md`.

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
│   └── scriptbots-app        # Binary crate wiring everything together
└── original_scriptbots_code_for_reference/  # Upstream C++ snapshot for parity
```

### Crate roles
- **`scriptbots-core`**: Simulation core with `WorldState`, `AgentState`, deterministic staged tick pipeline, config, sensor/actuation scaffolding, and brain registry bindings.
- **`scriptbots-brain`**: `Brain` trait + baseline implementations and adapters; experimental `assembly` behind a feature.
- **`scriptbots-brain-ml`**: Optional ML backends (Candle, Tract, tch) for alternative/accelerated inference (feature-gated).
- **`scriptbots-brain-neuro`**: Optional NeuroFlow-based brain; controllable at runtime via config/env (see below).
- **`scriptbots-index`**: Spatial indexing implementations; default uniform grid, optional `rstar` (R-tree) and `kd` (kiddo).
- **`scriptbots-storage`**: DuckDB persistence with buffered writes (`ticks`, `metrics`, `events`, `agents`) plus analytics helpers (e.g., `top_predators`, `latest_metrics`).
- **`scriptbots-render`**: GPUI UI layer with a window shell, HUD, canvas renderer for agents/food, selection highlights, and diagnostics overlay.
- **`scriptbots-app`**: Binary shell. Wires tracing/logging, config/env, storage pipeline, installs brains, seeds agents, and launches the GPUI shell.

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

### Feature flags & variants
- **`scriptbots-app` features**:
  - `ml` → enable `scriptbots-brain-ml`
  - `neuro` → enable `scriptbots-brain-neuro`
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

## Rendering & UX
- GPUI window, HUD, and canvas renderer for food tiles and agents (circles/spikes).
- Camera controls: pan/zoom; keyboard bindings for pause, draw toggle, speed ±.
- Overlays: selection highlights, diagnostics panel; charts and advanced overlays are staged in the plan.
- Inspector: per-agent stats and genome/brain views (scoped to plan milestones).
- Optional audio via `kira` (feature `audio`).

## Storage & analytics
- DuckDB schema (`ticks`, `metrics`, `events`, `agents`) with buffered writes and maintenance (`optimize`, `VACUUM`).
- Analytics helpers: `latest_metrics`, `top_predators`.
- Deterministic replay tooling is planned in the roadmap.

## Development workflow
- **Coding standards**: See `RUST_SYSTEM_PROGRAMMING_BEST_PRACTICES.md`. Embrace `Result`-based errors, clear traits, and avoid `unsafe`.
- **Linting**: `cargo clippy --workspace --all-targets --all-features -W clippy::all -W clippy::pedantic -W clippy::nursery`
- **Formatting**: `cargo fmt --all`
- **Tests**: `cargo test --workspace` (simulation and GPUI tests will be added as systems land)
- **Profiles**: Release uses LTO, single codegen unit, and abort-on-panic for optimal binaries.

## Contributing
- Keep changes scoped to the relevant crate; prefer improving existing files over adding new ones unless functionality is genuinely new.
- Update docs where it helps future maintainers understand decisions and invariants.
- For larger tasks, update `PLAN_TO_PORT_SCRIPTBOTS_TO_MODERN_IDIOMATIC_RUST_USING_GPUI.md` inline to mark progress.

## WebAssembly (sibling crate plan)
We maintain a separate plan for a browser-targeted sibling app (`scriptbots-web`) that reuses core crates without invasive changes. See `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` and the docs under `docs/wasm/` (ADRs, audits, and capability matrix). Initial MVP will run single-threaded with feature-gated dependencies; WebGPU vs Canvas2D rendering is under evaluation.

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
