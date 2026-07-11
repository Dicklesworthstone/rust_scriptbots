# Changelog

All notable changes to **Rust ScriptBots** are documented in this file.

This project has no tagged releases. History is organized into development
phases derived from the commit record and grouped by capability area within
each phase. Every entry links to the actual commit on GitHub.

Repository: <https://github.com/Dicklesworthstone/rust_scriptbots>

---

## Development Phases

| Phase | Dates | Theme |
|-------|-------|-------|
| [1. Foundation](#1-foundation--core-simulation-2025-10-21) | 2025-10-21 | Workspace scaffold, SoA core, full tick pipeline, spatial indexing |
| [2. Brains & GUI Shell](#2-brains--gui-shell-2025-10-21) | 2025-10-21 | MLP/DWRAON/Assembly/NeuroFlow brains, GPUI window, canvas renderer |
| [3. Persistence, Parity & Accessibility](#3-persistence-parity--accessibility-2025-10-22) | 2025-10-22 | DuckDB storage, CI/CD, combat/food parity, colorblind palettes, WASM research |
| [4. Control Plane & Terminal TUI](#4-control-plane--terminal-tui-2025-10-22--23) | 2025-10-22 -- 23 | REST/MCP servers, CLI tool, terminal renderer v2, replay events, WFC, hydrology |
| [5. Performance & SIMD](#5-performance--simd-2025-10-23--24) | 2025-10-23 -- 24 | Profiling, batched rendering, mimalloc, crossbeam, SIMD vectorization, benchmarks |
| [6. wgpu World Renderer](#6-wgpu-world-renderer-2025-10-24--26) | 2025-10-24 -- 26 | GPU terrain/agent pipelines, post-FX, wgpu 27 upgrade, CPU rasterizer fallback |
| [7. Rendering Parity & Camera](#7-rendering-parity--camera-overhaul-2025-10-29--30) | 2025-10-29 -- 30 | C++ geometry parity, camera extraction, HUD theming, accessibility modes |
| [8. Bevy 3D Renderer](#8-bevy-3d-renderer-2025-10-30--31) | 2025-10-30 -- 31 | Bevy 0.17 integration, 3D terrain heightfields, agent avatars, tonemapping |
| [9. Maintenance & Licensing](#9-maintenance--licensing-2025-11--2026-03) | 2025-11 -- 2026-03 | CI hardening, WASM compat, MIT + AI rider license, documentation |
| [10. Recovery Architecture](#10-recovery-architecture-2026-07) | 2026-07 | Evidence-led rearchitecture, FrankenSQLite migration, truthful GUI/TUI persistence |

---

## 10. Recovery Architecture (2026-07)

The recovery program replaced the former DuckDB stack with exact-revision FrankenSQLite and made persistence a bounded, observable subsystem instead of renderer-owned shared state.

### FrankenSQLite persistence replacement

- Pinned `fsqlite` 0.1.16 to immutable revision `cd9990bb16291d8c7c247b75b47faae8d7701adb` and qualified the real ScriptBots schema/query/transaction workload.
- Ported the seven-table run schema, explicit values, migrations, lifecycle rows, replay events, analytics queries, and CSV exports.
- Moved the deliberately non-`Send`/non-`Sync` connection entirely inside one bounded worker thread with startup, flush, durability, failure, and shutdown acknowledgements.
- Replaced GUI/TUI SQL and mutex access with revisioned immutable analytics snapshots; frontends now expose commit lag and storage health.
- Added a read-only `StorageReader` boundary so the control CLI and E2E tests never import the database engine or mutate a database during export.
- Changed application targets to `--storage {file|memory}`, fresh run paths to `.sqlite`, and CI replay artifacts to unique runner-temporary paths without destructive cleanup.
- Removed the DuckDB, bundled C++ database, Arrow, and Parquet dependency closure from manifests and the lockfile.

### Browser storage decision

FrankenSQLite remains the only SQL engine in browsers. The current WASM build is honestly documented as memory-only; durable browser storage requires a dedicated Worker plus SQLite-image checkpoints and an idempotent IndexedDB journal until a native FrankenSQLite OPFS/IndexedDB VFS passes real browser recovery tests.

---

## 1. Foundation & Core Simulation (2025-10-21)

Bootstrapped the entire Cargo workspace and implemented the deterministic
simulation core in a single day -- from project scaffolding to a fully
functional evolution loop with spatial indexing, brain genomes, sensing,
actuation, combat, death, reproduction, and persistence hooks.

### Project Scaffolding

Created the multi-crate Cargo workspace with shared lints, profiles, and
dependency management. Established the `scriptbots-core`, `scriptbots-brain`,
`scriptbots-brain-ml`, `scriptbots-index`, `scriptbots-storage`, and
`scriptbots-render` crate boundaries.

- [`faa7c10`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/faa7c10bcb7c458a1dea6661a55b5e4d691e3bdc) -- Initial commit (repo creation)
- [`6c12c02`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6c12c025e7f842b9ea735c024087d0c8d66ae806) -- ScriptBots Rust port project structure (multi-crate workspace)
- [`767d2db`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/767d2db3d45e0646360bf8eecd6f6deda06579f7) -- Add `scriptbots-index` crate and workspace dependencies
- [`5384988`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/53849884343e133a1cbe17890bc2f48d6a5d22ab) -- Add spatial indexing dependencies (rstar, kiddo)
- [`5cb51f2`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5cb51f23e51667f2edcd4ab57b363f41c45c3b14) -- Expand README with Windows support, feature flags, troubleshooting
- [`a004ce3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a004ce3f430c5f646f66620f3080f3000046a2be) -- Update workspace manifests, index/brain-ml scaffolding

### Simulation Engine (Tick Pipeline)

Implemented the entire deterministic tick pipeline: SoA agent columns with
generational IDs (slotmap), brain genome system, spatial indexing with
read-only snapshots, brain execution via registry, actuation with double
buffering, food system, combat with staged death, and reproduction with
mutation/crossover. Added Rayon-based parallelism for sensing.

- [`ca7f79e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ca7f79e2b8b22d4cc606eddf097511efaf56badd) -- Core simulation structures with SoA layout and generational IDs
- [`8fc07bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8fc07bc0fadfe56c0b329ec2fc8fcfd7807d1bfa) -- Brain genome system and simulation time-step pipeline
- [`c79e959`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c79e9590fbeba7d583a4ce8d29c335040f040e7b) -- Spatial indexing and agent sensing pipeline
- [`65befb4`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/65befb41b1a52512de6e2d58b60fa822e6911439) -- Brain execution pipeline with registry-backed runner system
- [`da16f74`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/da16f74db337666e4678fc19b2fc6368e3aec31c) -- Actuation pipeline and food system completing the action loop
- [`81312c3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/81312c3210be89fe6c165f2272d0bad2070fb6dc) -- Combat system and death mechanics completing survival dynamics
- [`12946f9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/12946f97722abf115dc5b23c89e075061c974c9a) -- Reproduction system and persistence layer completing the evolution loop
- [`bb919dd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bb919dd73e20b278af5c917d1d41843112cd645c) -- Refactor brain system, add history retention, parallelize sensing with Rayon

---

## 2. Brains & GUI Shell (2025-10-21)

With the tick pipeline complete, this phase delivered all baseline brain
implementations and the first GPU-accelerated rendering shell using GPUI.

### GPUI Rendering Shell

Integrated GPUI for a declarative, GPU-accelerated desktop window. Implemented
the Render trait, canvas renderer for agents and food, and camera controls
(pan/zoom).

- [`95b3187`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/95b31871352799640cf420dbffb230876f138caa) -- Complete pipeline parallelization and implement GPUI rendering shell
- [`c6d8ac9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c6d8ac99dd9432f9dd10093037ab181990747bea) -- Update GPUI integration to modern Render trait and element system
- [`c82e3e1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c82e3e1de08963e71cdd126ebac29c2d020f3b29) -- Integrate GPUI window into the application shell

### Brain Implementations

Delivered four brain families: MLP (production default), DWRAON, experimental
Assembly, and optional NeuroFlow. All support mutation, crossover, and
deterministic seeding. The brain registry enables mixed-species populations
and runtime selection.

- [`9a9bb1b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9a9bb1b0450844eb407f591f41c128053ce2b821) -- Baseline MLP brain with mutation and crossover operators
- [`c78f4da`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c78f4da2479246acc80853c05abbc98b945b1b94) -- Fix `as_any` methods in brain-ml for downcasting
- [`df8b1ed`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/df8b1edbca0eb7c2d1e236fd74d921de2f4c2edd) -- Mark MLP brain milestone as completed in roadmap
- [`b331ddd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b331ddda902ecbed494caf09a20336a3fbd77e14) -- Update rendering layer status with progress notes
- [`dcbde99`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dcbde99c20eb428b136272284fe6959ab148caef) -- Add DWRAON/Assembly brains, buffered storage analytics, HUD dashboard
- [`c5b010c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c5b010c6687df9f4879f7eb0d615a2e4255cda98) -- Add type aliases and cleanup redundant Default implementations
- [`e1a467d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e1a467d9f0b0e1922b277c5087dab916e418caa9) -- Remove unused Linear activation from NeuroFlow brain

### NeuroFlow Configuration & Determinism

Made NeuroFlow runtime-configurable via environment variables (hidden layer
sizes, activation function) and verified deterministic output with seeded runs.

- [`535df83`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/535df83e005b0240aa27ab03739de013bfa59f76) -- Deterministic RNG seeding and runtime NeuroFlow configuration
- [`5b022e1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5b022e1016acd448dff14edff8bc2856453ce902) -- Comprehensive NeuroFlow configuration with environment variable support
- [`fe9e93a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fe9e93a7a0397a494a2e4431efc0e759e0d0e20d) -- Apply rustfmt formatting and derive Default macro
- [`6af6fde`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6af6fde231ac194c5b6fbf16dda890ff64feba72) -- Fix missing imports for NeuroFlow configuration types
- [`ff69818`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ff69818650302dfec86250573c520a300f1c3f62) -- Deterministic NeuroFlow seeding, camera controls, canvas renderer
- [`5ce7754`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5ce77549a0cae084852ea385a87544751bb6ceba) -- Apply rustfmt formatting to event handlers and test functions

---

## 3. Persistence, Parity & Accessibility (2025-10-22)

Built out DuckDB persistence with async buffered writes, completed combat and
food consumption parity with the original C++ simulator, added a comprehensive
accessibility system, established CI/CD infrastructure, and began the
WebAssembly research initiative.

### DuckDB Storage & CI/CD Pipeline

Implemented the async storage worker with buffered writes to DuckDB tables
(ticks, metrics, events, agents), analytics helpers, and a CI/CD release
pipeline using `cargo dist`.

- [`004eacb`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/004eacb49a6815926b686dd504f01233545a7ef8) -- Async storage worker, comprehensive test suite, CI/CD release pipeline
- [`52d5351`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/52d53511ff158cd1b863c6e1ac2d8a0c38d1f66a) -- Brain registry factory pattern, food dynamics, inspector panel

### Simulation Parity with C++ Original

Achieved faithful reproduction of the original ScriptBots combat system,
sensory pipeline, reproduction mechanics, aging, food consumption (speed-based
intake, reproduction bonuses), and carcass sharing.

- [`a17483b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a17483b7a02e500e23f80c49b5ce2d4883ea2b29) -- Complete combat system, full sensory pipeline, enhanced reproduction, playback controls
- [`696224b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/696224b56e10be3e8c60f3f0ba036dc39f0b9431) -- Aging mechanics with health/energy decay and parallel feature flag
- [`5bf1966`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5bf19665415c514062f3bcf2767ff5934b19029c) -- Carcass sharing configuration scaffolding
- [`70dbd91`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/70dbd9164aeef84af79cdfa2cf2dac6c2d3a1fd5) -- Refine persistence HUD controls, sync WASM planning docs
- [`29e67bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/29e67bc00c0db006b3278d17fcc4dcd0b060cd4d) -- Legacy food consumption mechanics with speed-based intake and reproduction bonuses
- [`eca2aad`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/eca2aad15b68b18274c0f84e6c7796786d8da971) -- Apply rustfmt formatting to food mechanics code
- [`e75bc9d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e75bc9d63cdff365685c5fe5062a13af4a19e539) -- Improve regression test assertions with diagnostic messages
- [`b26a29f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b26a29f3ed646e4e5095fe1c7cfa8aac284d5eb0) -- Enable `getrandom` wasm_js feature and correct toroidal distance calculations

### Accessibility System

Delivered colorblind-safe palettes (deuteranopia, protanopia, tritanopia),
high-contrast mode, keyboard remapping with conflict resolution, and narration
hooks for future screen-reader integration.

- [`bb19cab`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bb19cab6ac11d61aefe398876cccec1c6178793f) -- Comprehensive accessibility system with colorblind palettes, keyboard rebinding, narration

### WebAssembly Research (Phase 1)

Conducted dependency audit, dry-run compilation, browser compatibility matrix,
security baseline, and authored ADRs for rendering pipeline, persistence, and
multithreading strategies.

- [`58268ca`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/58268ca876d988386c49abd439f10845dadcaf8f) -- Scaffold WebAssembly deployment initiative with Phase 1 research tasks
- [`a623537`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a6235379c24c4a2141ef5ccd6b9829703d24a18d) -- Phase 1 deliverables: dry run findings and dependency audit
- [`9d1997d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9d1997d47172ad43554d1e40e834e7a1aeabe097) -- Phase 1 completion: browser matrix, security baseline, ADR-001, spike plan
- [`fff47c1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fff47c1be435f538e6aacad72263407a205cd3cb) -- Tile-based terrain system, audio integration, browser persistence ADR
- [`1420971`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1420971137b063106ca5166231cbb15236112c75) -- Complete Phase 1 ADR suite and enhance rendering pipeline
- [`323515e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/323515e162e5f55b07949594e88c92fa387e7276) -- Add CPU RUSTFLAGS suppression task to Phase 1 TODO
- [`d14cce5`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d14cce5dbf64e00bdc69110a6b18ac988fd8b395) -- Phase 1 follow-up tasks and rendering pipeline refactor

### WASM Sibling Crate & Cross-Compilation

Created the `scriptbots-web` crate with wasm-bindgen bindings, settings UI,
WASM CI workflow, binary snapshot format (Postcard), and determinism validation
between native and WASM builds.

- [`8bee457`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8bee457bca28fa564a433053b0cd7ce52ab021e8) -- Topography, selection system, population management, release infrastructure
- [`1217ce8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1217ce8dcc07bda268fc129cd56b58b9a5702e7f) -- Phase 2 WASM sibling crate, settings UI, CI workflow, cross-compilation infrastructure
- [`bbc92fe`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bbc92fe2b398f4ce6162f0ac117d7c25746a2a13) -- Control API scaffolding, WASM determinism validation, web demo harness, lifecycle analytics
- [`7c950ba`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7c950ba827606ff34eabe1806d832e0bcbad0318) -- Binary snapshot format (Postcard), seed strategy control, WASM API documentation

### REST/MCP Servers & Analytics HUD

Wired the axum-based REST API with Swagger UI, MCP HTTP server, CLI tool with
TUI dashboard, and rich HUD panels showing mortality, generation tracking,
temperature discomfort, and hybrid birth analytics.

- [`55fc6bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/55fc6bce91d49a789e29035b161621ffb975698d) -- REST/MCP servers, CLI tool with TUI dashboard, advanced analytics, fertility-integrated food
- [`81ae14b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/81ae14b8bd6d91ccb65f488483155e804b48c3df) -- WASM CI workflow, analytics HUD display, food respawn capacity tests
- [`e96579e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e96579e29ba4eb342b877c3e1a393f598d537c77) -- Renderer abstraction, analytics HUD panels, comprehensive documentation
- [`9503657`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9503657f75661ce21698da296d7ab98a867a6bd6) -- Comprehensive mortality analytics, age/boost tracking, enhanced HUD panels
- [`f0c12c7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f0c12c7015683968326bc30583148180a2d32b60) -- Brain registration API, fix config parameter counts, update milestone status
- [`2ebcb29`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2ebcb299ebc152e6b3027aafe796c7f654ad9949) -- Generation tracking, temperature discomfort metrics, hybrid birth analytics

### Terminal & Headless Rendering

Implemented the terminal renderer with headless auto-detection, async command
queue, WebGPU rendering path, and headless smoke test infrastructure.

- [`1f21dfe`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1f21dfec4586be06c7ee2c188ec2ea80148b0fbc) -- Terminal renderer, WebGPU rendering, headless auto-detection
- [`e89ada9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e89ada9ae1be35779ff6e58f556c12daca681cec) -- Async command queue, complete MCP HTTP server, binary snapshot decoder
- [`9a8f073`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9a8f0737ae489423e7b2e5342c9e1dff73fdf987) -- Integrate command bus into renderers, refactor MCP server, update dependencies
- [`f740ecf`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f740ecf1e6df6f5528f47cefe51fe3a7059b2a27) -- Headless terminal mode with smoke test and documentation

### Runtime Configuration & Settings UI

Added a data-driven settings panel with live search, exclusive keyboard
capture, and runtime config mutation via REST and in-app UI.

- [`69f0acc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/69f0acc3aef30ffad5782771dc2aed44c4975dea) -- Adopt let-chain patterns and add terminal renderer tests
- [`0da233f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0da233f8e7b6d76e51183348fbeac0890a58976e) -- Specify nightly toolchain in CI workflows
- [`f2686eb`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f2686eb67176f422458865c1068bd2211bf85de2) -- Adopt let-chains in energy penalty logic, optimize test imports
- [`260d89f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/260d89fc450e154eb0932374569ec933c04c725b) -- Unblock host builds and WASM automation in CI
- [`7ca1f3b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7ca1f3b7034e37489afdd70752b0e0cd29bc9244) -- Avoid let-chain requiring edition 2024
- [`89cfa4e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/89cfa4eeb1f8d5358a009abdfa41845ca4884ef8) -- Satisfy clippy collapsible-if lint
- [`253810d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/253810d43f89e1300693ef1180035969738552b9) -- Functional search in settings, enhance terminal quit, refactor SQL generation
- [`2af9bba`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2af9bba3d751fb9f987a9698525815ab412eb515) -- Format search panel changes
- [`faeb5b5`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/faeb5b519e9320e3ec56cd498a6647e392954212) -- Unify settings panel rendering with data-driven parameter lists
- [`d5b2bdf`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d5b2bdf7446d90418e7294552f9f4693c89bb934) -- Exclusive keyboard capture for settings panel, enhanced category UI
- [`bab98f8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bab98f88f038e954b1a09b53a24080ffc58264a1) -- Runtime config updates and modern Rust idioms
- [`56e7d1c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/56e7d1c69c6125d8e59932314b6d1b6322313765) -- Correct food growth normalization and strengthen test validation
- [`6a80ae3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6a80ae3c853dd7b0c4bdc49b5d946b9600940b12) -- Expand documentation and add headless report generation

---

## 4. Control Plane & Terminal TUI (2025-10-22 -- 23)

Built the full control surface: REST API with Swagger UI, MCP HTTP server,
CLI tool with TUI dashboard, replay event system, terminal renderer v2 with
terrain/emoji/sparkline visualization, Wave Function Collapse procedural map
generator, and the hydrology simulation layer.

### Replay Event System

Implemented event capture, serialization/deserialization, architecture
documentation, and a CLI replay verification tool for deterministic
reproduction of simulation runs.

- [`58757f8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/58757f8c32186c27c5966dde17e37c39b9649cc9) -- Replay event system, comprehensive testing, architecture docs
- [`6cf6179`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6cf6179ecfccd5d256ad2ef49290c4161217c879) -- Environment variables reference, fix replay events test
- [`2a079bd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2a079bd83422ca976ea46638cf37c610541e0a89) -- Replay event types and comprehensive architecture documentation
- [`4c291c0`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4c291c05c2468b106040eb96e6dcbc41dd04c7c0) -- Configuration scenarios, replay roadmap, security notes
- [`58f8d5e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/58f8d5e784289c9f6662dd0cec5ae53eaaf3a886) -- Complete replay event deserialization infrastructure
- [`9626f03`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9626f03132ecf5796affa14ecb8a05ffb2253324) -- CLI replay verification tool with config layering and test coverage

### Terminal Renderer v2

Complete rewrite of the terminal renderer with terrain visualization,
diet-aware agent glyphs (emoji mappings for herbivores/carnivores/groups),
sparkline telemetry, directional heading arrows, leaderboard, and responsive
layout with auto-expanding panels.

- [`4bf9d18`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4bf9d185b449f946551c789d1a0e7b00f1642e5b) -- Terminal renderer v2: terrain visualization, diet-aware glyphs, sparkline telemetry, leaderboard
- [`338e4ee`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/338e4ee06ccb0888ba375742076f4c0a0e111039) -- Deterministic replay CI job, terminal rendering fixes, WFC sandbox roadmap
- [`f343f04`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f343f04a8caf83742d88d22120a56501bec88341) -- Fix replay diff color formatting borrow issues
- [`62fb01b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/62fb01be24fa5910443d5c9f326216076719a13c) -- Directional agent glyphs, diet percentages, enhanced trends
- [`2370516`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/23705164c471a48395f36589b0a210c1dc8e62c7) -- Normalize colorized delta formatting

### Control API & Auto-Pause

Added tick summary streaming endpoints, auto-pause infrastructure
(population thresholds, age limits, spike events), config inspection, CSV
export, and audit trail for configuration changes.

- [`1e40550`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1e40550880f1c5297c2981255b89dff71a74c776) -- Optimize release profile for faster incremental builds, update roadmap status
- [`e5c31b3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e5c31b3da813721d8a7c3d0a8c06705ffdbae230) -- Tick summary streaming endpoints and auto-pause infrastructure
- [`7640f46`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7640f4663942ee344b96f59ee02e6c8736703e1a) -- Mold linker for faster CI builds, fix wasm-opt flags, modernize Rust idioms
- [`a192e3f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a192e3f8320fd8f081541fc33479b0f3e3bf53a6) -- Config inspection, CSV export, auto-pause enforcement
- [`75ffe95`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/75ffe95cbf4154fe1f95fd52d38519b54ea217ac) -- Config audit trail, baseline metrics comparison, heading vector averaging

### Scenario Presets & MCP Tools

Introduced a scenario preset system accessible via REST API, CLI, GPUI UI,
and MCP tools. Presets allow one-click experiment configuration.

- [`669b00a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/669b00a5852e7af6c8d76faf0a5070ca4e597c97) -- Scenario preset system with REST API and GPUI baseline toggle button
- [`e0bee45`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e0bee451df1c02b0ca1c6f75900ad1ca40224c9b) -- Fix RON syntax in config test, remove baseline toggle from GPUI controls
- [`33c9bfd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/33c9bfdde9ee74b0eb95095488110ff5b5b855e3) -- MCP preset tools, GPUI preset UI, enhanced auto-pause, TickSummary fields
- [`9b4f2dc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9b4f2dcd83197597d40716804407857c70f5f0d7) -- Inline auto-pause logic and add TickSummary serialization support
- [`4cb5f2c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4cb5f2cdb586b5dd3009a80c74de7b3810aafbd8) -- Centralize PresetKind enum in core, eliminate duplication

### Wave Function Collapse Map Generator

Implemented a rule-based WFC generator producing deterministic terrain
layouts from tileset specs, with comprehensive test coverage and integration
into the terrain system.

- [`870cf3f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/870cf3f86e4c6e4d6870c3804513465ca4340ab2) -- Expand terrain system for WFC integration and restore auto-pause method
- [`f6a396e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f6a396e284e92a7cb71e77c75c68db892abdaead) -- Wave Function Collapse map generator with comprehensive test coverage

### Hydrology System

Built a per-cell water simulation layer on top of terrain: flow direction
computation, accumulation, basin identification, water depth fields, and a
hydrologic flow solver. Exposed via REST endpoint and CLI command.

- [`492fd31`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/492fd3183d055bcfcd5058ec72e7655fe60ba174) -- ControlRuntime test helper and hydrology system blueprint
- [`c196e44`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c196e44970e988e5ef74f794474e7117021877aa) -- Hydrology tile layer infrastructure for water simulation foundation
- [`e24363d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e24363dea58b0dbfb0b25dd2b332458d5ddeb5f6) -- Events API, scoreboard system, and hydrologic flow solver
- [`d60ff51`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d60ff51e6eaa0347ff8b2a469fe40d567c7d7748) -- Events/scoreboard REST endpoints, dual HUD leaderboards, runtime hydrology state
- [`46433ab`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/46433abbd40e5078a728f48c400cae13916bf407) -- Hydrology state REST endpoint with comprehensive flow data
- [`dc253d8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dc253d8f6a850b62629d204ea0b3edea4148802e) -- Hydrology CLI command and determinism self-check with parallel validation

### Screenshot & PNG Rendering

Added screenshot REST endpoints, NDJSON streaming, and headless PNG rendering
with configurable resolution.

- [`8a444ce`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8a444ce087a7fa889a94619b10907b90edb45dab) -- Screenshot endpoints with PNG placeholder and NDJSON stream
- [`8b15912`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8b15912cf091f6349589b3291b16eacdccd91609) -- Headless PNG rendering, test serialization, bug fixes

### Tick Cadence & Agent Debug APIs

Implemented a tick cadence scheduler for periodic events, added DietClass
enum and agent selection/debug query APIs exposed via REST.

- [`0d196c1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0d196c19743edd3d69c6101fc3184fa4fbd12d7b) -- GPUI system deps in CI, fix WASM RNG, apply clippy suggestions
- [`89e3660`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/89e3660adb9d27cdb066581c1e811ae65d24baf8) -- Apply cargo fmt to fix CI format check
- [`0017c0a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0017c0ad4162e6746377e3624ca2cc857b8dc021) -- Tick cadence scheduler for periodic simulation events
- [`2c4f75b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2c4f75be5550001e307303ab7962e366b2100fdc) -- Comprehensive tests for tick cadence scheduler
- [`fad315d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fad315d482f96066eadabcd54627cf802a09f827) -- Prevent integer overflow in coordinate noise hash function
- [`f1c4a93`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f1c4a933e469baa5bf7da09f24cdcb1db5b3d7bd) -- Assertions and debug output for cadence tests
- [`9e79889`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9e798894e0e5bdda662a8effed2f709b131cd705) -- Runtime trace for history sampling in tick cadence
- [`f70291f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f70291faf74875f89328341746e1b9780257ae74) -- Verbose traces for TickCadence decision methods
- [`1c33825`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1c338253e665496b8b94647bf331dffe752f1164) -- Detailed trace for history sampling decision in finalize_tick
- [`ae59c1e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ae59c1e0efaced4e0325b5bfe5c09e7c3f282351) -- Remove history sampling cadence, fix hydrology recursion, clean debug output
- [`b842948`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b842948e446cfd2efb9165ee59718378b3249235) -- DietClass enum and slotmap Key exports for debug APIs
- [`c2c99b7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c2c99b762ddf25a30c7305b6900280e837451e37) -- Agent selection and debug query APIs
- [`45f644f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/45f644f5a4088049d6155b3389cfc8eb3b8f9721) -- Agent debug and selection REST endpoints
- [`91839d6`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/91839d6d1cce59687238c2a58c27b640b49fa174) -- Comprehensive test coverage for agent selection and debug APIs

---

## 5. Performance & SIMD (2025-10-23 -- 24)

Intensive optimization pass targeting every layer of the stack: profiling
infrastructure, batched rendering with massive draw call reduction,
mimalloc allocator, crossbeam-channel for storage, SIMD vectorization of
eye sensors/combat/food regrowth using the `wide` crate, criterion
benchmarks, and rich GUI/terminal visualization features.

### Resource Control & Application Diagnostics

Added OS-level process priority management, diagnostic safe mode, debug
watermark overlay, dual-window GPUI layout, in-memory storage option, and
built-in profiling commands (`--profile-steps`, `--profile-sweep`,
`--auto-tune`).

- [`9369ede`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9369edea84eb3dd1d654fe286b9dc6ae257a1626) -- Resource control and headless PNG rendering capabilities
- [`8f440a4`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8f440a4c6ce52f7b34a884cc58173fbc19f49c74) -- Diagnostic safe mode and debug watermark for troubleshooting
- [`b3d9eba`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b3d9eba3c803289307382e2aa4a201f83b80fdb4) -- OS-level process priority management for low-power mode
- [`6ea289b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6ea289bf115dd3666279b247a656233dc3311409) -- Dual-window mode and in-memory storage option
- [`1fbb821`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1fbb821e1dd7f8af58c357b6e759f4b78d3707b8) -- Refactor canvas rendering and header methods
- [`f75822b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f75822bd3827e0b0c34331e4ff5eda1635754ffc) -- Fix storage variable for HUD window rendering
- [`559964d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/559964d2441ff9f99a56e371beca2005909e2e2c) -- Performance profiling infrastructure and optimization tools
- [`98838f2`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/98838f2b3fe85df38cdbc108e99717b2f735f7ae) -- Optional GUI feature for headless builds, enhanced stability

### Rendering Performance

Achieved major speedups through batched path rendering, viewport culling,
chart decimation, GPU adapter selection, coordinate calculation hoisting,
and elimination of per-frame allocations.

- [`67abc5c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/67abc5c0268da6cf54156105a9b93cdcdd6500de) -- Adaptive spatial indexing and intelligent frame scheduling
- [`b4abac0`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b4abac0155595ca7b04bfd034e7daf978a4e47c7) -- Auto-tune, optimize control API, enhance terminal rendering
- [`fd1dd20`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fd1dd20a5dc770edf12d5becd3fba4182af073b0) -- Batched path rendering for massive draw call reduction
- [`caa0b18`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/caa0b180879b4f219ab778bfdb68070a956fc925) -- Eliminate per-frame allocations, optimize hot path calculations
- [`98d8562`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/98d856219f9731faee3ffcfefaa23e2a2715381c) -- Aggressive compiler optimization, API caching, adaptive terrain rendering
- [`d7eba6a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d7eba6ac476f7ba87a30bad66570d2e08f02d354) -- Optimize food rendering with inverse division, hoist view bounds
- [`fa10b12`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fa10b12ba3a56403d9d30a31f11f248bcdc74af4) -- GPU adapter selection, chart decimation, coordinate calculation hoisting
- [`68ac5db`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/68ac5db428554faf0507e72fba345f879ed9f1b2) -- Viewport culling for terrain tiles, optimize pixel conversions
- [`141107f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/141107f0b4a1cf5ab93b5fbe786122606078029a) -- Hoist view bounds calculation and pass as parameters to terrain layer

### Allocator & Channel Upgrades

Replaced the standard allocator with mimalloc (feature-gated `fast-alloc`,
enabled by default) and swapped `std::sync::mpsc` for crossbeam-channel in
the storage pipeline for better throughput. Tuned DuckDB threading.

- [`b560566`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b560566f99327b2649c1954fa04235a702e4c958) -- mimalloc allocator option, intelligent emoji auto-detection, DuckDB tuning
- [`4489013`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4489013497868777e9d9af7b3c5b35d72a59f679) -- Replace mpsc with crossbeam-channel, enable fast-alloc by default, optimize DuckDB threading
- [`e93c368`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e93c3682c1fd5626a5c2f452318e2e4aca2871a6) -- Performance tuning guide, improve thread scaling defaults
- [`a3d75d6`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a3d75d63086bf0178ad4590b21223422e531aeef) -- Simplify heading char selection, remove unnecessary preallocation

### Terminal Enhancements & Cross-Platform Scripts

Added emoji mode toggle, narrow symbols mode, real-time analytics panels
(Insights, Brains leaderboard, mortality), responsive layout, and
cross-platform launcher scripts for Linux, macOS, and Windows.

- [`2ee3116`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2ee31168d13e7a2478fc2be74b1fd9101a37b1b7) -- Emoji mode toggle, native CPU optimizations
- [`6f0ce48`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6f0ce4814a0f742ff28905bd72ce92d1b5dd3dd9) -- Consolidate duplicate [build] sections in cargo config
- [`e3db696`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e3db696ee919eeb7379a3ff80425476e3d175617) -- Enable mixed brain families by default, eliminate unsafe code, deterministic API output
- [`7530c9f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7530c9f3ce22d06bd5f3595834c81ed7fd19f0fe) -- Narrow symbols mode, deterministic agent debug sorting, force-disable option
- [`df7b90c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/df7b90cf0090204ec6701ef2e7c1eefc0a3dd9d8) -- Optimize parallel execution with minimum chunk size, improve narrow mode UX
- [`e376731`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e37673136952d585e4e6a98379a80697008eab67) -- Cross-platform launcher scripts for terminal and GUI modes
- [`037f12c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/037f12c952dc8b5c92fac1eaed3297d6cd81657a) -- Enhanced help system with persistent hint, expanded legend, 'h' hotkey
- [`f36e6ac`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f36e6ac72d515f9521f910b920cdc41daa2a348a) -- Real-time analytics panels: Insights and Brains leaderboard
- [`46be120`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/46be1201477100d52aef338f8de822095d09b5b1) -- Enhanced Windows launchers with MSVC targeting, parallel builds, isolated artifacts
- [`cf79c46`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/cf79c462abdba591a909d90ccab2b705aa02b1e5) -- Document new APIs, responsive layout with mortality panel

### GUI Inspector Visualizations

Added brain output sparklines, radial radar chart, diet bars, temperature
comfort gauge, sensor heat ring, and per-agent diet energy gauges to the
GPUI inspector panel.

- [`0d856eb`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0d856eb14a70dcbd3d4849b35e70a9a1acca3bf4) -- macOS launchers, expanded analytics, brain bars in GUI inspector
- [`2aa37dd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2aa37ddd0f9c2d2ae1ca7f0d99df1b4a4e0ae21b) -- Brain output sparkline visualizations, temperature analytics
- [`ffe987d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ffe987dd43fd216fe0b1c779a00ede08cc62da90) -- Replace brain gauge with radial radar chart in inspector
- [`8e1c7ad`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8e1c7ad3fbceca333d730a8f574fba059f5730e9) -- Document WFC procedural map generator and hydrology snapshot API
- [`9774b72`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9774b727befec97cbf61ef78aa6f57cecd3a529f) -- Diet bars, temperature comfort gauge, sensor heat ring, diet energy gauges
- [`fbc358f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fbc358f1bb320c8741fe5cd96bb8dcdd856d22d2) -- Document storage backend flag, storage path env var, settings search feature

### Brain Activation Visualization

Added infrastructure to snapshot neural network activations at runtime and
render them in both GUI (neural edge rendering) and terminal (multi-row,
multi-layer display with emoji mode). Includes focus-lock modes for automatic
brain tracking.

- [`0da620f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0da620fc02df05e7fc73169c024165ba199087f8) -- Remove unused imports, silence dead code warnings
- [`08d2cf7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/08d2cf797c3af9ac21e10efe9540aba1e30d2ed4) -- Brain activation snapshot infrastructure for visualization
- [`d28995f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d28995f8c744383793d669ebd506c265e1cc96f4) -- Terminal brain visualization, NeuroFlow activation snapshots, neural edge rendering, camera offscreen guard
- [`1bef227`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1bef227d41b28d25797229466af598bacc22ad97) -- Multi-row brain visualization with emoji mode, interactive navigation, auto-follow
- [`2f8d831`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2f8d83169c911b6283295cd18cd0d1ae41d76ee3) -- Move `snapshot_activations` to Brain trait, multi-layer terminal support, type safety
- [`d4f3776`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d4f3776d009bb5edc0bd0df7f31b36840c47936c) -- Focus lock modes for automatic brain tracking, layer name display
- [`573891a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/573891a1177e415f891c2a35b58b4125baf98874) -- Temperature preference analytics display

### Camera & Rendering Fixes

Addressed edge cases in paint_frame logic for extreme panning/zooming,
offscreen recenter behavior, and first-frame blank canvas on Windows.

- [`a8cd394`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a8cd3945e15956d3dd0b66dc005ccabca3af084e) -- Re-evaluate render rectangle after camera adjustments
- [`463b693`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/463b693974960b9436739e9a0f75432fe42c89a4) -- Add `run_linux_with_gui.sh` script, optimize terminal settings
- [`5b48e6e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5b48e6e20addb25cc24cccec2f03684ed57eab98) -- Streamline offscreen recenter logic to prevent blank views
- [`d3e9663`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d3e9663cb011480dad9a2b92e2e359ff9a14f0cd) -- Remove unnecessary mutability in paint_frame camera logic
- [`530d5e0`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/530d5e0f689f196348de6cfa4280af062506b07c) -- Enhance storage pipeline error handling and fallback mechanism
- [`9f8782d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9f8782deac0f04b1b6ed5169718f97527cc2a8c3) -- Final offscreen guard for first-frame blank canvas on Windows

### SIMD Vectorization

Implemented 4-wide SIMD vectorization using the `wide` crate for eye sensor
computation, combat calculations, and food regrowth. Eliminated OrderedFloat
overhead, added criterion benchmarks, and introduced tunable parallelism
thresholds.

- [`8d91e19`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8d91e19574efe39a5bb81c04aa1c1f3b1358f8de) -- Let-chains, working buffers to eliminate per-tick allocations, parallelize food grid updates
- [`6dfe478`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6dfe478c837222a3f8f2aa1e7efaf9895ee3de94) -- SIMD eye sensor computation with `wide` crate, enabled by default
- [`d2874f7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d2874f7c76987ee509b5a1b14952e432a6d3d6a9) -- Eliminate OrderedFloat overhead, SIMD combat, criterion benchmarks, tunable parallelism thresholds

---

## 6. wgpu World Renderer (2025-10-24 -- 26)

Added the `scriptbots-world-gfx` crate -- a dedicated GPU world renderer
built on raw wgpu. Features terrain atlas with biome variation, water
caustics/shimmer, agent sprites with soft edges and rim highlights, a
full post-processing pipeline (ACES tonemapping, vignette, bloom,
height-fog), wgpu 0.20-to-27 upgrade, frame capture debugging tools, and
a CPU rasterizer fallback for systems without GPU support.

### Terrain & Agent GPU Pipelines

Built the terrain rendering pipeline with atlas UV mapping, water shimmer
effects, slope accents, biome variation, and CPU frustum culling. Agent
rendering uses premium sprite rendering with soft edges and rim highlights.

- [`566959a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/566959a42d4a54599e824f70845feb98ad1468c8) -- wgpu world renderer foundation, enhanced benchmarks, simplified SIMD eye sensors, offscreen architecture docs
- [`7b81848`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7b8184825b3fba8f7b02e9048be4f083f52e883c) -- Precompute sensor calculations, terrain/agent pipelines, SIMD Phase 2 roadmap
- [`bc04ec4`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bc04ec40f09d19c337b31d2dcc8bf4a959a6f9ee) -- ViewUniforms infrastructure, premium agent sprite rendering
- [`ec98937`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ec98937cbd5857af5416d5587f065a92683f2d97) -- Integrate ViewUniforms into rendering pipelines, fix wgpu API compatibility
- [`8e57233`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8e57233b4bf5d2e83e0c1868e1c412ca228eab3c) -- Remove texture cloning from RenderFrame, simplify readback API
- [`d8e7c1f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d8e7c1f53167a793ee5ecb000ccf0b91d483a6e7) -- Terrain atlas UV mapping, water shimmer, slope accents, CPU frustum culling, compositor stub

### SIMD Phase 2 Completion

Finished the remaining SIMD optimizations: food regrowth with 4-wide
vectorization, eyes dot-product, actuation batching, and SoA neighbor iterator.

- [`11d5c78`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/11d5c78d35ed6084f7d781d8f7fa6d9d1974dbd1) -- SIMD food regrowth with 4-wide vectorization
- [`2afd3f5`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2afd3f54cf80c48e97b2035a77c6349364a6d446) -- Complete Phase 2 SIMD (eyes dot-product, actuation batching), integrate wgpu compositor, SoA neighbor iterator

### Post-Processing Pipeline

Implemented a multi-pass post-FX pipeline: ACES tonemapping, vignette (Phase 1),
then bloom with Gaussian blur and height-fog (Phase 2). Added water caustics,
biome color variation, and boost visuals.

- [`b79575c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b79575c81af140f7dae8faa910cb0aed355a725d) -- Replace decimated blit with persistent image present, water caustics, biome variation, boost visuals, post-FX roadmap
- [`3e5184d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/3e5184d33d1b720c670263cb7fc19dc47a7b82b9) -- Post-FX Phase 1 MVP: ACES tonemapping and vignette

### wgpu 0.20 to 27 Upgrade

Major wgpu version upgrade resolving all API breaking changes across the
Instance, adapter request, bind group, and texture APIs.

- [`2ddbeee`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2ddbeeec9b0d916629180362632abbeec01aff99) -- Upgrade wgpu 0.20 to 27, Post-FX Phase 2 (bloom + height-fog), API compatibility fixes
- [`56c7e0a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/56c7e0abbc09aa7329d65cde33b84c8c52919459) -- Complete wgpu 27 API migration (remaining breaking changes)
- [`09f8d0e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/09f8d0ee6052d7a2333568dcd49a2174ad011ffb) -- Fix wgpu 27 Instance API and `request_adapter` Result type
- [`fc44cd9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fc44cd95801f8c197bdcec25ecd4580649393810) -- Validation fixes, bloom format, correct fog direction, cleanup warnings
- [`399e254`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/399e254710e4ea752140e3f4cc0dd147f525ab5f) -- Enable premium post-FX by default, env cache, sensible FPS/present defaults
- [`84cd2af`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/84cd2afecaf5f393b84c8c9303fb957481d293c2) -- Restore params_layout and blur_params_layout fields for bind group recreation

### Diagnostics, Frame Capture & Debugging

Built comprehensive debugging tools: frame capture to PNG, zero-sized viewport
guards, headless wgpu dump, auto-diagnostic capture, culling debug controls,
first-frame spin-wait, and auto-center camera on launch.

- [`dfb2f35`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dfb2f3549cb9bc42d05c0a0e2c78f2132833cd43) -- Optimize wgpu backend selection for reduced binary size
- [`687303c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/687303c40b299a732d6b006927618ffd27e991b3) -- Zero-sized viewport guards to prevent wgpu validation errors
- [`9ec8413`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9ec84139081ee716fcdd61bb9042de9c8f7dde70) -- Frame capture debugging tool, camera offset fix, shader optimization, bloom linear format
- [`216ed74`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/216ed74135d8bad2546d3e1c993ca966bbc1e58d) -- Integration test for wgpu compositor frame capture
- [`3f59954`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/3f59954755cf50b9ec5e69033b6d59846d3dcca7) -- Switch readback to blocking poll, default full present mode
- [`90e3c64`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/90e3c647e6bf7dc01822a7294f7ef3f679194f28) -- Set SB_WGPU_PRESENT_MODE=full in all platform launch scripts
- [`e067fe3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e067fe365943f10c07e10077b5681f22cf20f3d6) -- Agent rendering integration test, camera scaling fix for reduced resolution
- [`7247f46`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7247f460b49ee745d169a7cc84ec16476a0ef4c9) -- Standardize wgpu environment defaults across platforms with override support
- [`b6d04c7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b6d04c7cc0a408b4bc7ed5f8972e093ee4848311) -- Revert present mode default from diff to full
- [`7ac05c1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7ac05c18e8a4ad6529b51e55bb50f8b68d60fda7) -- Headless wgpu PNG dump, auto-diagnostic capture, culling debug controls
- [`0219570`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0219570bbb6ae9783486f37d29fc369357a1a6ef) -- First-frame spin-wait, auto-center camera on launch, visualization debug logging

### CPU Rasterizer Fallback

Added a software rasterizer fallback that activates automatically when wgpu
compositor initialization fails, ensuring the simulation can always render.

- [`0d86db8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0d86db87c0b9a4f55c4bd0eaed94cc849e474c1f) -- CPU rasterizer fallback when wgpu compositor fails to initialize

### winit 0.30 Migration & Smoke Tests

Migrated all GPU tests and examples to the winit 0.30 ApplicationHandler API
for consistency and forward compatibility.

- [`7860ee3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7860ee30aa750d823dc3fe1a30a648a75a0f5473) -- Minimal wgpu triangle smoke test for driver validation
- [`8a44936`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8a44936c41405f878d235da093bebbd6273024ac) -- Migrate smoke tests to winit 0.30, modernize event loop architecture
- [`1adeaad`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1adeaadb8624c25e96038cf51e90d790526b2812) -- Migrate wgpu_triangle to ApplicationHandler for winit 0.30 consistency
- [`96b05f1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/96b05f1b74f87986b086071b05f0e96c73eb65f5) -- Extract env_flag helper for consistent environment variable parsing

### wgpu Diagnostics (Late October)

Instrumented the readback pipeline, camera mapping, and renderer path
selection with comprehensive logging for production debugging.

- [`57471fc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/57471fc9bb6d1fcfea40e59ce5bc537fc612c619) -- Log selection and entry into wgpu paint path
- [`fb3a892`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fb3a89266cdcd1d269c4d50b78340a1a2d252253) -- Log camera mapping and visible instance counts; first-frame capture dir
- [`afcd025`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/afcd02521578fd8b56036cb7469277e8a34c01af) -- Fix SIGILL crashes, restore rustfmt compliance, document current work
- [`1916a1e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1916a1ec8ad89593dc9263b7108a038c965f8aba) -- Instrument readback pipeline with comprehensive metadata capture
- [`11df58f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/11df58fe4be689df0e24254c1afce538f71e45a6) -- Mark fixes in plan doc
- [`cf03181`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/cf03181bbdd1acb931c421759a21ccfd1815f40a) -- SIMD compound assignment operators; throttled readback logging
- [`513e631`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/513e6313dd4f5c1b56a62c0781dbc27078669fcf) -- SIMD compound assignment operators applied to third vision accumulation site

---

## 7. Rendering Parity & Camera Overhaul (2025-10-29 -- 30)

Restored geometric parity with the original C++ ScriptBots implementation,
extracted the camera module into a clean API, implemented snapshot regression
testing, and completed a visual polish pass with HUD theming and accessibility
modes.

### Code Quality & CI Hardening

Applied clippy suggestions across all crates, modernized Rust idioms, fixed
iterator destructuring, and eliminated dead variables.

- [`3418209`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/341820993a5a43496c36c72e544cb5484e9c44bc) -- Clippy suggestions and modernized Rust idioms across core crates
- [`395337f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/395337f77c0a3202e611c03701aff237a991421f) -- Fix iterator destructuring, eliminate dead variables in render viewport logic
- [`7a7f537`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7a7f537d475d400df8dd6e1ff039d375cee1c824) -- Enhance Linux GUI support and CI improvements
- [`97989ef`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/97989ef08df9cf66d575135327c6b97ff48a4432) -- Code quality, CI robustness, adapter diagnostics
- [`743bebb`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/743bebbfd79b69bb5d17d1ed4f7b19e70d20015e) -- Type safety, encapsulation, clippy compliance

### C++ Geometry Parity & Camera Extraction

Restored exact rendering geometry from the original C++ ScriptBots (coordinate
mapping, viewport layout, agent sizing). Extracted the camera module as a
standalone unit with a reduced API surface, consolidated viewport computation,
and established snapshot regression testing infrastructure.

- [`95bc206`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/95bc2061f2b35af6b84365335fe14a0094def2db) -- Restore legacy geometry parity with original C++ implementation
- [`15698ff`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/15698ffae826ae4c44376bfd55ff79536617eb69) -- GPUI camera alignment, Windows batch script for CPU rendering
- [`65c62c5`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/65c62c52ef01105f35a11bb8e590eb593b674508) -- Rendering parity infrastructure and snapshot regression testing
- [`0d56a8b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0d56a8b27987eaf72460da9aa7733c4989261150) -- Reduce CameraState API surface, improve script robustness
- [`d235e06`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d235e0648b52d907bb809175b33df9ff90a9e4ce) -- Camera module extraction and CI regression testing (Stage 1)
- [`f8802e9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f8802e92b699ef515ee61045c7947db7b8c6b5a0) -- Consolidate viewport layout computation (Stage 2)
- [`2cc8f3c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2cc8f3c9812e4ad5722f665d6621142ee078f6f9) -- Complete Stage 2 camera wiring, begin legacy palette migration

### Visual Polish & Accessibility Modes

Completed the PLAN document sections 1-3: enhanced agent visibility with
dynamic shading, HUD theming, terminal palette support, camera fit controls,
coordinate inspection overlay, and pathological zoom prevention.

- [`4ce38a1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4ce38a12321f6da27549b82ad81c3ccec4750744) -- Enhanced agent visibility, dynamic shading
- [`009d365`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/009d3659ecba8e38984bc9fb0a1a3936a7797db3) -- Visual polish pass: HUD theming, accessibility modes (PLAN section 3)
- [`df6ed29`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/df6ed29f23accaf4b83cf5f740a74734e2800d27) -- PLAN sections 1-3: terminal palettes, camera fit controls, coordinate inspection
- [`ac33ded`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ac33ded0b68a55ba25e9fb31400902a305a8fad3) -- Prevent pathological zoom states, improve fit-to-selection UX

---

## 8. Bevy 3D Renderer (2025-10-30 -- 31)

Added the `scriptbots-bevy` crate -- a Bevy 0.17-based 3D renderer with WFC
terrain heightfield meshes, 3D agent avatars, tonemapping controls, HDR
camera with auto-exposure, playback controls (speed/pause/step), and
cross-platform launch scripts.

### Phase 0: Workspace Scaffolding

Integrated Bevy into the workspace with a stub renderer, CLI integration,
and asset system wiring.

- [`dd37514`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dd375147d35abd94d0a43b4f2ed021502986febf) -- wgpu blank-frame detection and Bevy integration plan
- [`d1c8158`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d1c8158d359ef9d0605645ecaa453982da2d3e95) -- Mark Phase 0 scaffolding as in progress
- [`13971f4`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/13971f4d88ff9ad49a3d3dfe080c281b879e7997) -- Log Bevy Phase 0 start in coordination docs
- [`f084281`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f084281099484e9106bb01574b024dc6803b7a0c) -- Phase 0 scaffolding with workspace integration and stub renderer
- [`fc43d7e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fc43d7e88e7174e2e90d7ef910434d52c85b25c9) -- CLI integration and asset system fixes for Phase 0
- [`a4e29c2`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a4e29c2bcf8450ef92f44d7001d47f900f76ec8e) -- Improve conditional compilation pattern in renderer selection
- [`81a21dd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/81a21dd10d3f2727d48c5bf6cb29ee6ac0d94ce8) -- Simplify stub renderer, update to Bevy 0.14 API conventions
- [`8a493c0`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8a493c05d22e290f6abbdbce8003c726a3f28a2b) -- Remove invalid workspace.features section from root Cargo.toml
- [`9712e07`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9712e07abad70e695ade988568f2e5846cd28ae6) -- Add scriptbots-core dependency for Phase 1 preparation

### Phases 1-2: World Rendering & Camera

Completed full world rendering with camera controls and HUD feature parity
with the GPUI renderer.

- [`b429044`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b42904496eb0998d1c8862adf1a52026079612de) -- Complete Phases 0-2: world rendering, camera controls, HUD parity

### Phases 3-4: Terrain, Interactivity & HUD

Implemented WFC terrain heightfield rendering with chunked meshes, agent
elevation sampling, interactive HUD action row with follow buttons,
keyboard shortcuts, and command bridge wiring.

- [`f669817`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f66981727a47106043bcd49a4485b7703a0fb530) -- Begin Phase 4 interactivity, enhance Phase 1 terrain planning
- [`7fe4cd0`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7fe4cd06b8c6890a736a3a19c55d0603c147f5b7) -- Phase 4 command bridge wiring, add slotmap dependency
- [`fef6d52`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fef6d52bf703cd3bb31064b54132d1c8414d59c6) -- WFC terrain heightfield rendering with chunked meshes, agent elevation sampling
- [`058986b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/058986b6902d7ca1ee91f2fc346a8350d0da571b) -- Phase 4 HUD action row with interactive follow buttons and clear selection
- [`a5e1aa6`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a5e1aa645d5afae3ce86a83cba14c9e29b765414) -- Phase 4 working TODO checklist and consolidated progress notes
- [`299642e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/299642e80ba01aad8e6847c8ad4b42b3a3a01e27) -- Button icons, keyboard shortcuts, debug logging, color management refactor
- [`5d6bbcf`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5d6bbcf2c15d7c5f6f7087691bc40899db582b3c) -- Mark Phase 4 complete, terrain materials with height-based reflectance

### Simulation Playback Controls

Implemented a complete playback control system with speed adjustment,
pause/resume, single-step, and UI controls -- all wired through the
SimulationCommand channel.

- [`e559845`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e559845fe09fe4758e1fd6117332d5bc69335823) -- Complete playback control system with speed/pause/step UI and threading
- [`60a5933`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/60a59337c24753604571622eecbc9245b6ba59c8) -- Wire playback controls to SimulationCommand channel; fix GPUI collapsed viewport edge case

### Bevy 0.17 Migration & Rich Agent Avatars

Upgraded to Bevy 0.17 (then 0.17.2), migrated all text/query APIs, and
implemented trait-driven rich agent avatar rendering in both GPUI and wgpu
renderers. The wgpu path uses procedural WGSL shaders with a 40-float per-agent
GPU payload.

- [`6b12f2b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6b12f2bfa9ed22b157612fadf0b81bb0248a0b11) -- Phase 4 simulation command pipeline, Bevy 0.17 upgrade, cross-platform launch scripts, rich avatar rendering start
- [`31b6a70`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/31b6a70bd745ade7dc7d7f9b75aee84a6b0f5e0b) -- Extract agent avatar rendering into dedicated paint_agent_avatar helper (GPUI)
- [`2c0a2ba`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2c0a2baaa86b7e3d2ff835907822d039f47d61a6) -- Complete 0.17 API migration + rich ScriptBot avatar rendering with trait-driven visual features (GPUI)
- [`a2e2ea7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a2e2ea7542541921f23703abd854b4a94c1e8f42) -- Complete 0.17 text/query API migration + Phase 5 QA & performance checklist
- [`1942f02`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1942f02dff0aa5e91a2e8ab37e8052f970bd65bf) -- Expand AgentInstanceGpu to 40-float payload for rich avatar rendering (wgpu)
- [`ec4e518`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ec4e518e6db87fceddfa686c74eaa523d30c1ef5) -- Rich avatar rendering with procedural WGSL shaders + Bevy 0.17.2 upgrade + Phase 5 benchmarking infra
- [`ba0e1f6`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ba0e1f6f62098a87283336fbafa9e23ae4a0df3f) -- Replace TransformBundle with explicit Transform+GlobalTransform; remove legacy viewport fallback

### QA, Benchmarking & Lighting

Added turnkey performance benchmarking scripts, reflection probe lighting
for terrain chunks, and addressed benchmarking blockers.

- [`cd952be`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/cd952be101cf141b77a3d3734e467290eb4efe78) -- Exclude runtime-generated logs directory from git
- [`6c195ef`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6c195efd64688dd232fe942d398a3c1a72683661) -- Turnkey performance benchmarking script + snapshot refresh checklist
- [`a1d6175`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a1d6175db088e71840262012485553572ba674d1) -- Reflection probe lighting for terrain chunks; guard eye radius clamp against panic
- [`69cb037`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/69cb0378dcaa18df6cc0023d7ecaf1d1b5607901) -- Bevy 3D agent avatars roadmap, checklist indentation fix, shader eye placement fix

### Tonemapping & HDR Camera

Implemented HDR camera with auto-exposure infrastructure, tonemapping
controls (ACES, Reinhard, etc.), and accessibility-aware color palettes
for the 3D renderer.

- [`afd4fda`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/afd4fda345749ec2729e065803f0f29f10e8f885) -- Tonemapping controls + HDR camera setup + auto-exposure infrastructure
- [`d850baa`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d850baa336a54376a7933aafebdcd7cb0181ab3d) -- Complete 3D agent avatars + wire tonemapping systems + accessibility palettes
- [`dfc4867`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dfc486743dce1cb70df19c1bafb9291db83ba7aa) -- Use TonemappingState speed fields instead of hardcoded auto-exposure values

---

## 9. Maintenance & Licensing (2025-11 -- 2026-03)

Post-feature-development period focused on tonemapping polish, CI hardening,
WASM compatibility fixes, documentation, repository hygiene, and licensing
under MIT with an OpenAI/Anthropic rider.

### Tonemapping Configuration (2025-11-07)

Surfaced tonemapping parameters as runtime configuration with GPUI offscreen
parity and environment variable overrides.

- [`3399176`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/3399176cbc79dd6eb0c0d3395e1b51b5f09bac50) -- Tonemapping config surface + GPUI offscreen parity + env overrides

### Repository Hygiene (2026-01)

Updated gitignore patterns for beads metadata, daemon logs, ephemeral files,
and beads viewer local config.

- [`bbd443f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bbd443f20d19e1a6bb51142799ca591d38ca4830) -- Add daemon log pattern to .beads/.gitignore, create .ubsignore
- [`66b78bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/66b78bc14695185906728637ee7c76d1f170134a) -- Update beads config (config.json to metadata.json)
- [`aeaa49c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/aeaa49cd50ae1fc1ebd63076a9a9cb3c106c51d7) -- Exclude beads viewer local config and caches from git
- [`2d92148`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2d92148acd79f8bcee9ccf3a65be2610281ed1d1) -- Add ephemeral beads file patterns to gitignore

### CI & WASM Compatibility (2026-01)

Hardened GitHub Actions workflows, fixed WASM compatibility issues with
`configure_parallelism`, simplified core library public API, and addressed
security audit warnings.

- [`6dc273d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6dc273df787f83f7f5f2e6b9af134074aeba5dc1) -- Update AGENTS.md and documentation
- [`c82b528`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c82b5285548483f55c438b4152656b9b9643b692) -- Improve GitHub Actions workflows with best practices
- [`d273927`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d273927cbaedb37bc68e62705d5181c4d2b9232e) -- Update AGENTS.md with project context
- [`83b0710`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/83b07107eb9ac09629c69fe7129906e25ca63670) -- Ignore unmaintained package advisories in security audit
- [`2eabab5`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2eabab5ec52c6025de3150bfdb1276e512099b75) -- WASM compatibility for `configure_parallelism`
- [`be03536`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/be035363cd244b76084043de075a3e662fbf3856) -- Simplify core library and improve web module initialization
- [`9057cd3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9057cd367aeb66b987946245e51bfb2f0296db54) -- Improve GitHub Actions workflow configuration

### Licensing (2026-01 -- 02)

Added MIT license, then updated it with an OpenAI/Anthropic rider for
AI-assisted development attribution.

- [`ea8fadc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ea8fadc5f7b5e1643fd3ad818cefe563037367e6) -- Add MIT License
- [`993b7bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/993b7bcd9036b804e5d4b17262e934bd697433cd) -- Update license to MIT with OpenAI/Anthropic Rider

### Documentation (2026-02 -- 03)

Updated multi-agent conventions and project documentation.

- [`15d0591`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/15d05913c3bda82e669c08a46aec21eba1d4fc8a) -- Update AGENTS.md with latest multi-agent conventions

---

## Project Statistics

| Metric | Value |
|--------|-------|
| Total commits (excl. stash) | 274 |
| First commit | 2025-10-21 |
| Latest commit | 2026-03-21 |
| Tagged releases | 0 |
| Crates in workspace | 12 |
| Active feature development | 11 days (2025-10-21 -- 2025-10-31) |
| Maintenance window | 2025-11 -- 2026-03 |

### Workspace Crates

| Crate | Role |
|-------|------|
| `scriptbots-core` | Simulation engine: WorldState, tick pipeline, config, terrain, hydrology |
| `scriptbots-brain` | Brain trait + MLP, DWRAON, Assembly implementations |
| `scriptbots-brain-ml` | Optional ML backends (Candle, Tract, tch), feature-gated |
| `scriptbots-brain-neuro` | Optional NeuroFlow brain, feature-gated |
| `scriptbots-index` | Pluggable spatial indices (grid, rstar, kd-tree) |
| `scriptbots-storage` | FrankenSQLite persistence, bounded worker, replay, read-only analytics |
| `scriptbots-render` | GPUI UI layer: window shell, HUD, canvas renderer, inspector |
| `scriptbots-app` | Binary orchestrator: CLI, REST/MCP servers, renderer selection |
| `scriptbots-web` | WebAssembly harness (wasm-bindgen bindings) |
| `scriptbots-world-gfx` | Raw wgpu world renderer with post-FX pipeline |
| `scriptbots-bevy` | Bevy 0.17 3D renderer with terrain heightfields and agent avatars |

### Key Capabilities Timeline

| Capability | Phase | First Commit |
|------------|-------|-------------|
| SoA agent layout + generational IDs | 1 | `ca7f79e` (2025-10-21) |
| Deterministic tick pipeline | 1 | `da16f74` (2025-10-21) |
| Brain registry (MLP/DWRAON/Assembly) | 2 | `dcbde99` (2025-10-21) |
| GPUI desktop window | 2 | `95b3187` (2025-10-21) |
| NeuroFlow brain | 2 | `535df83` (2025-10-21) |
| DuckDB persistence | 3 | `004eacb` (2025-10-22) |
| Accessibility (colorblind palettes) | 3 | `bb19cab` (2025-10-22) |
| WASM sibling crate | 3 | `1217ce8` (2025-10-22) |
| REST API + Swagger UI | 3 | `55fc6bc` (2025-10-22) |
| MCP HTTP server | 3 | `e89ada9` (2025-10-22) |
| Terminal renderer v2 (emoji TUI) | 4 | `4bf9d18` (2025-10-23) |
| Replay event system | 4 | `58757f8` (2025-10-22) |
| WFC map generator | 4 | `f6a396e` (2025-10-23) |
| Hydrology simulation | 4 | `c196e44` (2025-10-23) |
| Scenario presets | 4 | `669b00a` (2025-10-23) |
| SIMD eye sensors (wide crate) | 5 | `6dfe478` (2025-10-24) |
| mimalloc allocator | 5 | `b560566` (2025-10-24) |
| Criterion benchmarks | 5 | `d2874f7` (2025-10-24) |
| Brain activation visualization | 5 | `08d2cf7` (2025-10-24) |
| wgpu world renderer | 6 | `566959a` (2025-10-24) |
| Post-FX pipeline (bloom/fog/ACES) | 6 | `3e5184d` (2025-10-25) |
| CPU rasterizer fallback | 6 | `0d86db8` (2025-10-25) |
| C++ geometry parity | 7 | `95bc206` (2025-10-30) |
| Camera module extraction | 7 | `d235e06` (2025-10-30) |
| Bevy 3D renderer | 8 | `f084281` (2025-10-30) |
| WFC terrain heightfields (3D) | 8 | `fef6d52` (2025-10-31) |
| HDR camera + tonemapping | 8 | `afd4fda` (2025-10-31) |
| Rich 3D agent avatars | 8 | `d850baa` (2025-10-31) |
| MIT + AI rider license | 9 | `ea8fadc` (2026-01-21) |
