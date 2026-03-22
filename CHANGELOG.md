# Changelog

All notable changes to **Rust ScriptBots** are documented in this file.

This project has no tagged releases yet. History is organized into development
phases derived from the commit record. Each section links to the actual commit
on GitHub so agents and humans can trace every change to source.

Repository: <https://github.com/Dicklesworthstone/rust_scriptbots>

---

## Development Phases At A Glance

| Phase | Dates | Focus |
|-------|-------|-------|
| [Foundation](#phase-1--foundation--core-simulation-2025-10-21) | 2025-10-21 | Workspace scaffold, SoA core, tick pipeline, brain registry, spatial index |
| [Brains & Rendering Shell](#phase-2--brains--rendering-shell-2025-10-21) | 2025-10-21 | MLP/DWRAON/Assembly/NeuroFlow brains, GPUI window, canvas renderer |
| [Storage, CI & Accessibility](#phase-3--storage-ci--accessibility-2025-10-22) | 2025-10-22 | DuckDB persistence, CI/CD pipeline, combat/reproduction parity, accessibility, WASM research |
| [Control Plane & Terminal TUI](#phase-4--control-plane--terminal-tui-2025-10-22--23) | 2025-10-22 -- 23 | REST/MCP servers, CLI tool, terminal renderer v2, replay events, WFC map gen, hydrology |
| [Performance & SIMD](#phase-5--performance--simd-2025-10-23--24) | 2025-10-23 -- 24 | Profiling infra, batched rendering, mimalloc, crossbeam, SIMD eye/combat/food, criterion benchmarks |
| [wgpu World Renderer](#phase-6--wgpu-world-renderer-2025-10-24--26) | 2025-10-24 -- 26 | Terrain/agent GPU pipelines, post-FX (bloom, fog, tonemapping), wgpu 27 upgrade, frame capture |
| [Rendering Parity & Camera](#phase-7--rendering-parity--camera-overhaul-2025-10-29--30) | 2025-10-29 -- 30 | Legacy C++ geometry parity, camera module extraction, HUD theming, accessibility modes |
| [Bevy 3D Renderer](#phase-8--bevy-3d-renderer-2025-10-30--31) | 2025-10-30 -- 31 | Bevy 0.17 integration, 3D terrain heightfields, agent avatars, tonemapping, playback controls |
| [Maintenance & Licensing](#phase-9--maintenance--licensing-2025-11--2026-02) | 2025-11 -- 2026-02 | Tonemapping config, CI hardening, WASM compat, MIT license with AI rider |

---

## Phase 1 -- Foundation & Core Simulation (2025-10-21)

Bootstrapped the Cargo workspace and implemented the deterministic simulation
core in a single day. All commits build the tick pipeline from first principles:
SoA agent layout, spatial indexing, brain genomes, sensing, actuation, combat,
death, reproduction, and persistence hooks.

### Workspace & Scaffolding
- [`faa7c10`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/faa7c10bcb7c458a1dea6661a55b5e4d691e3bdc) -- Initial commit
- [`6c12c02`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6c12c025e7f842b9ea735c024087d0c8d66ae806) -- ScriptBots Rust port project structure (multi-crate workspace)
- [`767d2db`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/767d2db3d45e0646360bf8eecd6f6deda06579f7) -- Add `scriptbots-index` crate and workspace dependencies
- [`5384988`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/53849884343e133a1cbe17890bc2f48d6a5d22ab) -- Add spatial indexing dependencies (rstar, kiddo)
- [`5cb51f2`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5cb51f23e51667f2edcd4ab57b363f41c45c3b14) -- Expand README with Windows support, features, troubleshooting
- [`a004ce3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a004ce3f430c5f646f66620f3080f3000046a2be) -- Update workspace manifests and index/brain-ml scaffolding

### Core Simulation Engine
- [`ca7f79e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ca7f79e2b8b22d4cc606eddf097511efaf56badd) -- Implement core simulation structures with SoA layout and generational IDs
- [`8fc07bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8fc07bc0fadfe56c0b329ec2fc8fcfd7807d1bfa) -- Implement brain genome system and simulation time-step pipeline
- [`c79e959`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c79e9590fbeba7d583a4ce8d29c335040f040e7b) -- Implement spatial indexing and agent sensing pipeline
- [`65befb4`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/65befb41b1a52512de6e2d58b60fa822e6911439) -- Implement brain execution pipeline with registry-backed runner system
- [`da16f74`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/da16f74db337666e4678fc19b2fc6368e3aec31c) -- Implement actuation pipeline and food system completing the action loop
- [`81312c3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/81312c3210be89fe6c165f2272d0bad2070fb6dc) -- Implement combat system and death mechanics
- [`12946f9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/12946f97722abf115dc5b23c89e075061c974c9a) -- Implement reproduction system and persistence layer completing evolution loop
- [`bb919dd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bb919dd73e20b278af5c917d1d41843112cd645c) -- Refactor brain system, add history retention, and parallelize sensing (Rayon)

---

## Phase 2 -- Brains & Rendering Shell (2025-10-21)

With the tick pipeline complete, this phase added the first GPUI rendering
shell and all baseline brain implementations (MLP, DWRAON, Assembly, NeuroFlow).

### GPUI Rendering
- [`95b3187`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/95b31871352799640cf420dbffb230876f138caa) -- Complete pipeline parallelization and implement GPUI rendering shell
- [`c6d8ac9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c6d8ac99dd9432f9dd10093037ab181990747bea) -- Fix GPUI integration to modern Render trait and element system
- [`c82e3e1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c82e3e1de08963e71cdd126ebac29c2d020f3b29) -- Integrate GPUI window

### Brain Implementations
- [`9a9bb1b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9a9bb1b0450844eb407f591f41c128053ce2b821) -- Implement baseline MLP brain with mutation and crossover operators
- [`c78f4da`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c78f4da2479246acc80853c05abbc98b945b1b94) -- Fix `as_any` methods in brain-ml
- [`dcbde99`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dcbde99c20eb428b136272284fe6959ab148caef) -- Add DWRAON/Assembly brains, buffered storage analytics, and HUD dashboard
- [`535df83`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/535df83e005b0240aa27ab03739de013bfa59f76) -- Add deterministic RNG seeding and runtime NeuroFlow configuration
- [`5b022e1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5b022e1016acd448dff14edff8bc2856453ce902) -- Add comprehensive NeuroFlow configuration with environment variable support
- [`ff69818`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ff69818650302dfec86250573c520a300f1c3f62) -- Add deterministic NeuroFlow seeding, camera controls, and canvas renderer

---

## Phase 3 -- Storage, CI & Accessibility (2025-10-22)

Built out DuckDB persistence, the CI/CD release pipeline, full combat/
reproduction parity with the original C++ simulator, accessibility features,
and began WebAssembly research.

### Storage & CI
- [`004eacb`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/004eacb49a6815926b686dd504f01233545a7ef8) -- Add async storage worker, comprehensive test suite, and CI/CD release pipeline
- [`52d5351`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/52d53511ff158cd1b863c6e1ac2d8a0c38d1f66a) -- Refactor brain registry to factory pattern, food dynamics, and inspector panel

### Simulation Parity
- [`a17483b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a17483b7a02e500e23f80c49b5ce2d4883ea2b29) -- Complete combat system, full sensory pipeline, enhanced reproduction, and playback controls
- [`696224b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/696224b56e10be3e8c60f3f0ba036dc39f0b9431) -- Implement aging mechanics with health/energy decay and parallel feature flag
- [`5bf1966`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5bf19665415c514062f3bcf2767ff5934b19029c) -- Add carcass sharing configuration scaffolding
- [`29e67bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/29e67bc00c0db006b3278d17fcc4dcd0b060cd4d) -- Implement legacy food consumption mechanics with speed-based intake and reproduction bonuses
- [`b26a29f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b26a29f3ed646e4e5095fe1c7cfa8aac284d5eb0) -- Fix: enable `getrandom` wasm_js feature and correct toroidal distance calculations

### Accessibility
- [`bb19cab`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bb19cab6ac11d61aefe398876cccec1c6178793f) -- Comprehensive accessibility system with colorblind palettes, keyboard rebinding, and narration

### WebAssembly Research (Phase 1)
- [`58268ca`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/58268ca876d988386c49abd439f10845dadcaf8f) -- Scaffold WebAssembly deployment initiative with Phase 1 research tasks
- [`a623537`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a6235379c24c4a2141ef5ccd6b9829703d24a18d) -- Phase 1 deliverables: dry run findings and dependency audit
- [`9d1997d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9d1997d47172ad43554d1e40e834e7a1aeabe097) -- Phase 1 completion: browser matrix, security baseline, ADR-001, spike plan
- [`fff47c1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fff47c1be435f538e6aacad72263407a205cd3cb) -- Tile-based terrain system, audio integration, and browser persistence ADR

### WASM Sibling Crate & Rendering Infrastructure
- [`8bee457`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8bee457bca28fa564a433053b0cd7ce52ab021e8) -- Implement topography, selection system, population management, and release infra
- [`1217ce8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1217ce8dcc07bda268fc129cd56b58b9a5702e7f) -- Implement Phase 2 WASM sibling crate, settings UI, CI workflow, and cross-compilation
- [`bbc92fe`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bbc92fe2b398f4ce6162f0ac117d7c25746a2a13) -- Implement control API scaffolding, WASM determinism validation, web demo harness, lifecycle analytics
- [`55fc6bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/55fc6bce91d49a789e29035b161621ffb975698d) -- Implement REST/MCP servers, CLI tool with TUI dashboard, advanced analytics, fertility-integrated food
- [`7c950ba`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7c950ba827606ff34eabe1806d832e0bcbad0318) -- Add binary snapshot format (Postcard), seed strategy control, WASM API docs

### Analytics & HUD
- [`9503657`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9503657f75661ce21698da296d7ab98a867a6bd6) -- Add comprehensive mortality analytics, age/boost tracking, enhanced HUD panels
- [`2ebcb29`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2ebcb299ebc152e6b3027aafe796c7f654ad9949) -- Add generation tracking, temperature discomfort metrics, hybrid birth analytics
- [`e96579e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e96579e29ba4eb342b877c3e1a393f598d537c77) -- Add renderer abstraction, analytics HUD panels, comprehensive documentation

### Runtime Config & Settings UI
- [`faeb5b5`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/faeb5b519e9320e3ec56cd498a6647e392954212) -- Unify settings panel rendering with data-driven parameter lists
- [`d5b2bdf`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d5b2bdf7446d90418e7294552f9f4693c89bb934) -- Add exclusive keyboard capture for settings panel and enhance category UI
- [`bab98f8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bab98f88f038e954b1a09b53a24080ffc58264a1) -- Implement runtime config updates and adopt modern Rust idioms
- [`253810d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/253810d43f89e1300693ef1180035969738552b9) -- Implement functional search, enhance terminal quit, refactor SQL generation

### Terminal & Headless Modes
- [`1f21dfe`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1f21dfec4586be06c7ee2c188ec2ea80148b0fbc) -- Implement terminal renderer, WebGPU rendering, headless auto-detection
- [`e89ada9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e89ada9ae1be35779ff6e58f556c12daca681cec) -- Implement async command queue, complete MCP HTTP server, binary snapshot decoder
- [`f740ecf`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f740ecf1e6df6f5528f47cefe51fe3a7059b2a27) -- Add headless terminal mode with smoke test

---

## Phase 4 -- Control Plane & Terminal TUI (2025-10-22 -- 23)

Built the full control surface: REST API (axum + Swagger), MCP HTTP server,
CLI tool with TUI dashboard, replay event system, terminal renderer v2 with
terrain/emoji/sparklines, Wave Function Collapse map generator, and hydrology
simulation layer.

### Replay Event System
- [`58757f8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/58757f8c32186c27c5966dde17e37c39b9649cc9) -- Add replay event system, comprehensive testing, architecture docs
- [`2a079bd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2a079bd83422ca976ea46638cf37c610541e0a89) -- Add replay event types and comprehensive architecture documentation
- [`58f8d5e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/58f8d5e784289c9f6662dd0cec5ae53eaaf3a886) -- Complete replay event deserialization infrastructure
- [`9626f03`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9626f03132ecf5796affa14ecb8a05ffb2253324) -- Add CLI replay verification tool with config layering and test coverage

### Terminal Renderer v2
- [`4bf9d18`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4bf9d185b449f946551c789d1a0e7b00f1642e5b) -- Complete terminal renderer v2: terrain visualization, diet-aware agent glyphs, sparkline telemetry, leaderboard
- [`338e4ee`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/338e4ee06ccb0888ba375742076f4c0a0e111039) -- Add deterministic replay CI job, terminal rendering fixes, WFC sandbox roadmap
- [`62fb01b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/62fb01be24fa5910443d5c9f326216076719a13c) -- Add directional agent glyphs, diet percentages, enhanced trends

### Control API & Auto-Pause
- [`e5c31b3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e5c31b3da813721d8a7c3d0a8c06705ffdbae230) -- Add tick summary streaming endpoints and auto-pause infrastructure
- [`a192e3f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a192e3f8320fd8f081541fc33479b0f3e3bf53a6) -- Add config inspection, CSV export, and auto-pause enforcement
- [`75ffe95`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/75ffe95cbf4154fe1f95fd52d38519b54ea217ac) -- Add config audit trail, baseline metrics comparison, heading vector averaging

### Scenario Presets & MCP Tools
- [`669b00a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/669b00a5852e7af6c8d76faf0a5070ca4e597c97) -- Add scenario preset system with REST API and GPUI baseline toggle button
- [`33c9bfd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/33c9bfdde9ee74b0eb95095488110ff5b5b855e3) -- Add MCP preset tools, GPUI preset UI, enhanced auto-pause, TickSummary fields
- [`4cb5f2c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4cb5f2cdb586b5dd3009a80c74de7b3810aafbd8) -- Centralize PresetKind enum in core and eliminate duplication

### Wave Function Collapse Map Generator
- [`870cf3f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/870cf3f86e4c6e4d6870c3804513465ca4340ab2) -- Expand terrain system for WFC integration and restore auto-pause method
- [`f6a396e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f6a396e284e92a7cb71e77c75c68db892abdaead) -- Implement Wave Function Collapse map generator with comprehensive test coverage

### Hydrology System
- [`492fd31`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/492fd3183d055bcfcd5058ec72e7655fe60ba174) -- Add ControlRuntime test helper and hydrology system blueprint
- [`c196e44`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c196e44970e988e5ef74f794474e7117021877aa) -- Implement hydrology tile layer infrastructure for water simulation foundation
- [`e24363d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e24363dea58b0dbfb0b25dd2b332458d5ddeb5f6) -- Add events API, scoreboard system, and hydrologic flow solver
- [`d60ff51`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d60ff51e6eaa0347ff8b2a469fe40d567c7d7748) -- Expose events/scoreboard REST endpoints, dual HUD leaderboards, runtime hydrology state
- [`46433ab`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/46433abbd40e5078a728f48c400cae13916bf407) -- Expose hydrology state via REST endpoint with comprehensive flow data
- [`dc253d8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dc253d8f6a850b62629d204ea0b3edea4148802e) -- Add hydrology CLI command and determinism self-check with parallel validation

### Screenshot & PNG Rendering
- [`8a444ce`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8a444ce087a7fa889a94619b10907b90edb45dab) -- Add screenshot endpoints with PNG placeholder and NDJSON stream
- [`8b15912`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8b15912cf091f6349589b3291b16eacdccd91609) -- Implement headless PNG rendering, test serialization, bug fixes

### Tick Cadence & Agent Debug APIs
- [`0017c0a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0017c0ad4162e6746377e3624ca2cc857b8dc021) -- Implement tick cadence scheduler for periodic simulation events
- [`fad315d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fad315d482f96066eadabcd54627cf802a09f827) -- Fix: prevent integer overflow in coordinate noise hash function
- [`ae59c1e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ae59c1e0efaced4e0325b5bfe5c09e7c3f282351) -- Remove history sampling cadence, fix hydrology recursion, clean debug output
- [`b842948`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b842948e446cfd2efb9165ee59718378b3249235) -- Add DietClass enum and slotmap Key exports for debug APIs
- [`c2c99b7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c2c99b762ddf25a30c7305b6900280e837451e37) -- Implement agent selection and debug query APIs
- [`45f644f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/45f644f5a4088049d6155b3389cfc8eb3b8f9721) -- Expose agent debug and selection REST endpoints

---

## Phase 5 -- Performance & SIMD (2025-10-23 -- 24)

Intensive optimization pass. Added profiling infrastructure, batched rendering,
alternative allocators (mimalloc), replaced mpsc with crossbeam-channel,
implemented SIMD vectorization for eye sensors, combat, and food regrowth,
and added criterion benchmarks.

### Resource Control & Diagnostics
- [`9369ede`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9369edea84eb3dd1d654fe286b9dc6ae257a1626) -- Add resource control and headless PNG rendering capabilities
- [`8f440a4`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8f440a4c6ce52f7b34a884cc58173fbc19f49c74) -- Add diagnostic safe mode and debug watermark for troubleshooting
- [`b3d9eba`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b3d9eba3c803289307382e2aa4a201f83b80fdb4) -- Add OS-level process priority management for low-power mode
- [`6ea289b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6ea289bf115dd3666279b247a656233dc3311409) -- Add dual-window mode and in-memory storage option
- [`559964d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/559964d2441ff9f99a56e371beca2005909e2e2c) -- Add performance profiling infrastructure and optimization tools

### Rendering Performance
- [`67abc5c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/67abc5c0268da6cf54156105a9b93cdcdd6500de) -- Adaptive spatial indexing and intelligent frame scheduling
- [`b4abac0`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b4abac0155595ca7b04bfd034e7daf978a4e47c7) -- Auto-tune, optimize control API, enhance terminal rendering
- [`fd1dd20`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fd1dd20a5dc770edf12d5becd3fba4182af073b0) -- Implement batched path rendering for massive draw call reduction
- [`caa0b18`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/caa0b180879b4f219ab778bfdb68070a956fc925) -- Eliminate per-frame allocations and optimize hot path calculations
- [`98d8562`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/98d856219f9731faee3ffcfefaa23e2a2715381c) -- Aggressive compiler optimization, API caching, adaptive terrain rendering
- [`d7eba6a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d7eba6ac476f7ba87a30bad66570d2e08f02d354) -- Optimize food rendering with inverse division and hoist view bounds
- [`fa10b12`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fa10b12ba3a56403d9d30a31f11f248bcdc74af4) -- GPU adapter selection, chart decimation, coordinate calculation hoisting
- [`68ac5db`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/68ac5db428554faf0507e72fba345f879ed9f1b2) -- Implement viewport culling for terrain tiles and optimize pixel conversions

### Allocator & Channel Upgrades
- [`b560566`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b560566f99327b2649c1954fa04235a702e4c958) -- Add mimalloc allocator option, intelligent emoji auto-detection, DuckDB tuning
- [`4489013`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4489013497868777e9d9af7b3c5b35d72a59f679) -- Replace mpsc with crossbeam-channel, enable fast-alloc by default, optimize DuckDB threading

### Terminal Enhancements
- [`2ee3116`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2ee31168d13e7a2478fc2be74b1fd9101a37b1b7) -- Add emoji mode toggle and enable native CPU optimizations
- [`e3db696`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e3db696ee919eeb7379a3ff80425476e3d175617) -- Enable mixed brain families by default, eliminate unsafe code, deterministic API output
- [`7530c9f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7530c9f3ce22d06bd5f3595834c81ed7fd19f0fe) -- Add narrow symbols mode for emoji alignment, deterministic agent debug sorting
- [`e376731`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e37673136952d585e4e6a98379a80697008eab67) -- Add cross-platform launcher scripts for terminal and GUI modes
- [`037f12c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/037f12c952dc8b5c92fac1eaed3297d6cd81657a) -- Enhance help system with persistent hint, expanded legend, 'h' hotkey
- [`f36e6ac`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f36e6ac72d515f9521f910b920cdc41daa2a348a) -- Add real-time analytics panels: Insights and Brains leaderboard
- [`46be120`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/46be1201477100d52aef338f8de822095d09b5b1) -- Enhance Windows launchers with MSVC targeting, parallel builds, isolated artifacts

### GUI Inspector Enhancements
- [`0d856eb`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0d856eb14a70dcbd3d4849b35e70a9a1acca3bf4) -- Add macOS launchers, expand analytics, brain bars in GUI inspector
- [`2aa37dd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2aa37ddd0f9c2d2ae1ca7f0d99df1b4a4e0ae21b) -- Add brain output sparkline visualizations to GUI inspector
- [`ffe987d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ffe987dd43fd216fe0b1c779a00ede08cc62da90) -- Replace brain gauge with radial radar chart in inspector
- [`9774b72`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9774b727befec97cbf61ef78aa6f57cecd3a529f) -- Add diet bars, temperature comfort gauge, sensor heat ring, diet energy gauges

### Brain Visualization
- [`08d2cf7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/08d2cf797c3af9ac21e10efe9540aba1e30d2ed4) -- Add brain activation snapshot infrastructure for visualization
- [`d28995f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d28995f8c744383793d669ebd506c265e1cc96f4) -- Terminal brain visualization, NeuroFlow activation snapshots, neural edge rendering, camera offscreen guard
- [`1bef227`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1bef227d41b28d25797229466af598bacc22ad97) -- Multi-row brain visualization with emoji mode, interactive navigation, auto-follow
- [`2f8d831`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2f8d83169c911b6283295cd18cd0d1ae41d76ee3) -- Move `snapshot_activations` to Brain trait, multi-layer terminal support, type safety
- [`d4f3776`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d4f3776d009bb5edc0bd0df7f31b36840c47936c) -- Add focus lock modes for automatic brain tracking, layer name display

### SIMD Vectorization
- [`8d91e19`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8d91e19574efe39a5bb81c04aa1c1f3b1358f8de) -- Adopt let-chains, add working buffers to eliminate per-tick allocations, parallelize food grid updates
- [`6dfe478`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6dfe478c837222a3f8f2aa1e7efaf9895ee3de94) -- Implement SIMD eye sensor computation with `wide` crate, enable by default
- [`d2874f7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d2874f7c76987ee509b5a1b14952e432a6d3d6a9) -- Eliminate OrderedFloat overhead, add SIMD combat, implement criterion benchmarks, tunable parallelism thresholds

---

## Phase 6 -- wgpu World Renderer (2025-10-24 -- 26)

Added a dedicated GPU world renderer (`scriptbots-world-gfx`) using raw wgpu.
Terrain atlas, water effects, agent sprites, post-processing pipeline (ACES
tonemapping, vignette, bloom, height-fog), wgpu 0.20-to-27 upgrade, frame
capture debugging, and CPU rasterizer fallback.

### Foundation & Terrain Pipeline
- [`566959a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/566959a42d4a54599e824f70845feb98ad1468c8) -- Add wgpu world renderer foundation, enhance benchmarks, simplify SIMD eye sensors, document offscreen architecture
- [`7b81848`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7b8184825b3fba8f7b02e9048be4f083f52e883c) -- Precompute sensor calculations, implement terrain/agent pipelines, document SIMD Phase 2 roadmap
- [`bc04ec4`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bc04ec40f09d19c337b31d2dcc8bf4a959a6f9ee) -- Add ViewUniforms infrastructure and premium agent sprite rendering with soft edges and rim highlights
- [`d8e7c1f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d8e7c1f53167a793ee5ecb000ccf0b91d483a6e7) -- Implement terrain atlas UV mapping, water shimmer, slope accents, CPU frustum culling, compositor stub

### SIMD Phase 2 Completion
- [`11d5c78`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/11d5c78d35ed6084f7d781d8f7fa6d9d1974dbd1) -- Implement SIMD food regrowth with 4-wide vectorization, mark Phase 2 sensing optimizations complete
- [`2afd3f5`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2afd3f54cf80c48e97b2035a77c6349364a6d446) -- Complete Phase 2 SIMD optimizations (eyes dot-product, actuation batching), integrate wgpu compositor, SoA neighbor iterator

### Post-Processing Pipeline
- [`b79575c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b79575c81af140f7dae8faa910cb0aed355a725d) -- Replace decimated blit with persistent image present, add water caustics, biome variation, boost visuals, post-FX roadmap
- [`3e5184d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/3e5184d33d1b720c670263cb7fc19dc47a7b82b9) -- Implement Post-FX Phase 1 MVP with ACES tonemapping and vignette

### wgpu 27 Upgrade
- [`2ddbeee`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2ddbeeec9b0d916629180362632abbeec01aff99) -- Upgrade wgpu 0.20 to 27, implement Post-FX Phase 2 (bloom + height-fog), fix API compatibility
- [`56c7e0a`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/56c7e0abbc09aa7329d65cde33b84c8c52919459) -- Complete wgpu 27 API migration with remaining breaking changes
- [`09f8d0e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/09f8d0ee6052d7a2333568dcd49a2174ad011ffb) -- Fix wgpu 27 Instance API and `request_adapter` Result type in compositor
- [`fc44cd9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fc44cd95801f8c197bdcec25ecd4580649393810) -- Add validation, fix bloom format, correct fog direction, cleanup warnings
- [`399e254`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/399e254710e4ea752140e3f4cc0dd147f525ab5f) -- Enable premium post-FX by default, add env cache, set sensible FPS/present defaults

### Diagnostics & Frame Capture
- [`687303c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/687303c40b299a732d6b006927618ffd27e991b3) -- Add zero-sized viewport guards to prevent wgpu validation errors and crashes
- [`9ec8413`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9ec84139081ee716fcdd61bb9042de9c8f7dde70) -- Add frame capture debugging tool, fix camera offset, optimize shaders, ensure bloom linear format
- [`216ed74`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/216ed74135d8bad2546d3e1c993ca966bbc1e58d) -- Integration test for wgpu compositor frame capture
- [`7ac05c1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7ac05c18e8a4ad6529b51e55bb50f8b68d60fda7) -- Add headless wgpu PNG dump, auto-diagnostic capture, culling debug controls
- [`0219570`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0219570bbb6ae9783486f37d29fc369357a1a6ef) -- Add first-frame spin-wait, auto-center camera on launch, visualization debug logging

### CPU Fallback & Backend Selection
- [`dfb2f35`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dfb2f3549cb9bc42d05c0a0e2c78f2132833cd43) -- Optimize wgpu backend selection for reduced binary size and faster compilation
- [`0d86db8`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0d86db87c0b9a4f55c4bd0eaed94cc849e474c1f) -- Add CPU rasterizer fallback when wgpu compositor fails to initialize

### Smoke Tests & winit Migration
- [`7860ee3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7860ee30aa750d823dc3fe1a30a648a75a0f5473) -- Add minimal wgpu triangle smoke test for driver validation
- [`8a44936`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/8a44936c41405f878d235da093bebbd6273024ac) -- Migrate smoke tests to winit 0.30 and modernize event loop architecture
- [`1adeaad`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1adeaadb8624c25e96038cf51e90d790526b2812) -- Migrate `wgpu_triangle` to ApplicationHandler for winit 0.30 consistency

### wgpu Diagnostics (late October)
- [`57471fc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/57471fc9bb6d1fcfea40e59ce5bc537fc612c619) -- Log selection and entry into wgpu paint path
- [`fb3a892`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fb3a89266cdcd1d269c4d50b78340a1a2d252253) -- Always log camera mapping and visible instance counts; ensure first-frame capture dir
- [`1916a1e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1916a1ec8ad89593dc9263b7108a038c965f8aba) -- Instrument readback pipeline with comprehensive metadata capture
- [`cf03181`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/cf03181bbdd1acb931c421759a21ccfd1815f40a) -- SIMD compound assignment ops; add throttled readback logging

---

## Phase 7 -- Rendering Parity & Camera Overhaul (2025-10-29 -- 30)

Restored geometric parity with the original C++ ScriptBots implementation,
extracted the camera module, implemented regression snapshot testing, and
added HUD theming with accessibility modes.

### Code Quality & CI
- [`afcd025`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/afcd02521578fd8b56036cb7469277e8a34c01af) -- Fix SIGILL crashes, restore rustfmt compliance, document current work
- [`3418209`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/341820993a5a43496c36c72e544cb5484e9c44bc) -- Apply clippy suggestions and modernize Rust idioms across core crates
- [`395337f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/395337f77c0a3202e611c03701aff237a991421f) -- Fix iterator destructuring, eliminate dead variables in render viewport logic
- [`7a7f537`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7a7f537d475d400df8dd6e1ff039d375cee1c824) -- Enhance Linux GUI support and CI improvements
- [`97989ef`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/97989ef08df9cf66d575135327c6b97ff48a4432) -- Enhance code quality, CI robustness, adapter diagnostics
- [`743bebb`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/743bebbfd79b69bb5d17d1ed4f7b19e70d20015e) -- Improve type safety, encapsulation, clippy compliance

### Geometry Parity & Camera
- [`95bc206`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/95bc2061f2b35af6b84365335fe14a0094def2db) -- Restore legacy geometry parity with original C++ implementation
- [`15698ff`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/15698ffae826ae4c44376bfd55ff79536617eb69) -- Enhance GPUI camera alignment and introduce Windows batch script for CPU rendering
- [`65c62c5`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/65c62c52ef01105f35a11bb8e590eb593b674508) -- Establish rendering parity infrastructure and snapshot regression testing
- [`0d56a8b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/0d56a8b27987eaf72460da9aa7733c4989261150) -- Reduce CameraState API surface and improve script robustness
- [`d235e06`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d235e0648b52d907bb809175b33df9ff90a9e4ce) -- Camera module extraction and CI regression testing (Stage 1)
- [`f8802e9`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f8802e92b699ef515ee61045c7947db7b8c6b5a0) -- Consolidate viewport layout computation (Stage 2)
- [`2cc8f3c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2cc8f3c9812e4ad5722f665d6621142ee078f6f9) -- Complete Stage 2 camera wiring and begin legacy palette migration

### Visual Polish & Accessibility
- [`4ce38a1`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/4ce38a12321f6da27549b82ad81c3ccec4750744) -- Enhance agent visibility and add dynamic shading
- [`009d365`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/009d3659ecba8e38984bc9fb0a1a3936a7797db3) -- Complete visual polish pass with HUD theming and accessibility modes (PLAN section 3)
- [`df6ed29`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/df6ed29f23accaf4b83cf5f740a74734e2800d27) -- Complete PLAN sections 1-3 with terminal palettes, camera fit controls, coordinate inspection
- [`ac33ded`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ac33ded0b68a55ba25e9fb31400902a305a8fad3) -- Fix: prevent pathological zoom states and improve fit-to-selection UX

---

## Phase 8 -- Bevy 3D Renderer (2025-10-30 -- 31)

Added a Bevy-based 3D renderer (`scriptbots-bevy`) with WFC terrain
heightfields, chunked meshes, agent elevation sampling, playback controls,
rich 3D agent avatars, tonemapping, and HDR camera with auto-exposure.

### Phase 0: Scaffolding
- [`dd37514`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dd375147d35abd94d0a43b4f2ed021502986febf) -- Add wgpu blank-frame detection and Bevy integration plan
- [`f084281`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f084281099484e9106bb01574b024dc6803b7a0c) -- Complete Phase 0 scaffolding with workspace integration and stub renderer
- [`fc43d7e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fc43d7e88e7174e2e90d7ef910434d52c85b25c9) -- Complete CLI integration and asset system fixes for Phase 0
- [`81a21dd`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/81a21dd10d3f2727d48c5bf6cb29ee6ac0d94ce8) -- Simplify stub renderer and update to Bevy 0.14 API conventions
- [`9712e07`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9712e07abad70e695ade988568f2e5846cd28ae6) -- Add scriptbots-core dependency for Phase 1 preparation

### Phases 1-2: World Rendering & Camera
- [`b429044`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/b42904496eb0998d1c8862adf1a52026079612de) -- Complete Phases 0-2 with full world rendering, camera controls, and HUD parity

### Phase 3-4: Interactivity & Terrain
- [`f669817`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/f66981727a47106043bcd49a4485b7703a0fb530) -- Begin Phase 4 interactivity, enhance Phase 1 terrain planning
- [`7fe4cd0`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/7fe4cd06b8c6890a736a3a19c55d0603c147f5b7) -- Complete Phase 4 command bridge wiring, add slotmap dependency
- [`fef6d52`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/fef6d52bf703cd3bb31064b54132d1c8414d59c6) -- Implement WFC terrain heightfield rendering with chunked meshes and agent elevation sampling
- [`058986b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/058986b6902d7ca1ee91f2fc346a8350d0da571b) -- Complete Phase 4 HUD action row with interactive follow buttons and clear selection
- [`299642e`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/299642e80ba01aad8e6847c8ad4b42b3a3a01e27) -- Add button icons, keyboard shortcuts, debug logging, refactor color management
- [`5d6bbcf`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/5d6bbcf2c15d7c5f6f7087691bc40899db582b3c) -- Mark Phase 4 complete, polish terrain materials with height-based reflectance

### Playback Controls
- [`e559845`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/e559845fe09fe4758e1fd6117332d5bc69335823) -- Implement complete simulation playback control system with speed/pause/step UI and threading
- [`60a5933`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/60a59337c24753604571622eecbc9245b6ba59c8) -- Wire playback controls to submit SimulationCommand through control channel

### Bevy 0.17 Migration & Rich Avatars
- [`6b12f2b`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6b12f2bfa9ed22b157612fadf0b81bb0248a0b11) -- Complete Phase 4 simulation command pipeline, upgrade Bevy 0.17, add cross-platform launch scripts, start rich agent avatar rendering
- [`2c0a2ba`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2c0a2baaa86b7e3d2ff835907822d039f47d61a6) -- Complete 0.17 API migration + implement rich ScriptBot avatar rendering with trait-driven visual features (GPUI)
- [`a2e2ea7`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a2e2ea7542541921f23703abd854b4a94c1e8f42) -- Complete 0.17 text/query API migration + add Phase 5 QA & performance checklist
- [`1942f02`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/1942f02dff0aa5e91a2e8ab37e8052f970bd65bf) -- Expand AgentInstanceGpu to 40-float payload for rich avatar rendering (wgpu)

### wgpu Rich Avatar Shaders
- [`ec4e518`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ec4e518e6db87fceddfa686c74eaa523d30c1ef5) -- Complete rich avatar rendering with procedural WGSL shaders + upgrade Bevy 0.17.2 + Phase 5 benchmarking infra
- [`ba0e1f6`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ba0e1f6f62098a87283336fbafa9e23ae4a0df3f) -- Replace TransformBundle with explicit Transform+GlobalTransform, remove legacy viewport fallback

### QA & Benchmarking
- [`6c195ef`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6c195efd64688dd232fe942d398a3c1a72683661) -- Add turnkey performance benchmarking script + finalize snapshot refresh checklist
- [`a1d6175`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/a1d6175db088e71840262012485553572ba674d1) -- Add reflection probe lighting for terrain chunks + guard eye radius clamp against panic

### Tonemapping & HDR
- [`afd4fda`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/afd4fda345749ec2729e065803f0f29f10e8f885) -- Add tonemapping controls + HDR camera setup + auto-exposure infrastructure
- [`d850baa`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d850baa336a54376a7933aafebdcd7cb0181ab3d) -- Complete 3D agent avatars + wire tonemapping systems + accessibility palettes
- [`dfc4867`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/dfc486743dce1cb70df19c1bafb9291db83ba7aa) -- Use TonemappingState speed fields instead of hardcoded auto-exposure values

---

## Phase 9 -- Maintenance & Licensing (2025-11 -- 2026-02)

Post-feature-development period focused on tonemapping polish, CI hardening,
WASM compatibility fixes, documentation updates, and licensing.

### Tonemapping Config (2025-11-07)
- [`3399176`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/3399176cbc79dd6eb0c0d3395e1b51b5f09bac50) -- Add tonemapping config surface + GPUI offscreen parity + env overrides

### Repository Hygiene (2026-01)
- [`bbd443f`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/bbd443f20d19e1a6bb51142799ca591d38ca4830) -- Add daemon log pattern to .beads/.gitignore, create .ubsignore
- [`66b78bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/66b78bc14695185906728637ee7c76d1f170134a) -- Update beads config (config.json to metadata.json)
- [`aeaa49c`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/aeaa49cd50ae1fc1ebd63076a9a9cb3c106c51d7) -- Exclude beads viewer local config and caches from git
- [`2d92148`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2d92148acd79f8bcee9ccf3a65be2610281ed1d1) -- Add ephemeral beads file patterns to gitignore

### CI & WASM Fixes (2026-01)
- [`c82b528`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/c82b5285548483f55c438b4152656b9b9643b692) -- Improve GitHub Actions workflows with best practices
- [`83b0710`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/83b07107eb9ac09629c69fe7129906e25ca63670) -- Fix(ci): ignore unmaintained package advisories in security audit
- [`2eabab5`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/2eabab5ec52c6025de3150bfdb1276e512099b75) -- Fix: WASM compatibility for `configure_parallelism`
- [`be03536`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/be035363cd244b76084043de075a3e662fbf3856) -- Simplify core library and improve web module initialization
- [`9057cd3`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/9057cd367aeb66b987946245e51bfb2f0296db54) -- Improve GitHub Actions workflow configuration

### Licensing (2026-01 -- 02)
- [`ea8fadc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/ea8fadc5f7b5e1643fd3ad818cefe563037367e6) -- Add MIT License
- [`993b7bc`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/993b7bcd9036b804e5d4b17262e934bd697433cd) -- Update license to MIT with OpenAI/Anthropic Rider

### Documentation
- [`6dc273d`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/6dc273df787f83f7f5f2e6b9af134074aeba5dc1) -- Update AGENTS.md and documentation
- [`d273927`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/d273927cbaedb37bc68e62705d5181c4d2b9232e) -- Update AGENTS.md with project context
- [`15d0591`](https://github.com/Dicklesworthstone/rust_scriptbots/commit/15d05913c3bda82e669c08a46aec21eba1d4fc8a) -- Update AGENTS.md with latest multi-agent conventions

---

## Commit Statistics

| Metric | Value |
|--------|-------|
| Total commits | 279 |
| First commit | 2025-10-21 |
| Latest commit | 2026-02-21 |
| Tagged releases | 0 |
| Crates in workspace | 12 |
| Active development window | ~11 days (Oct 21 -- Oct 31, 2025) |

### Crates

| Crate | Role |
|-------|------|
| `scriptbots-core` | Simulation engine: WorldState, AgentState, tick pipeline, config, spatial indexing bindings |
| `scriptbots-brain` | Brain trait + MLP, DWRAON, Assembly implementations |
| `scriptbots-brain-ml` | Optional ML backends (Candle, Tract, tch), feature-gated |
| `scriptbots-brain-neuro` | Optional NeuroFlow brain, feature-gated |
| `scriptbots-index` | Pluggable spatial indices (grid, rstar, kd-tree) |
| `scriptbots-storage` | DuckDB persistence, buffered writes, analytics helpers |
| `scriptbots-render` | GPUI UI layer: window shell, HUD, canvas renderer, inspector |
| `scriptbots-app` | Binary orchestrator: CLI, REST/MCP servers, renderer selection |
| `scriptbots-web` | WebAssembly harness (wasm-bindgen bindings) |
| `scriptbots-world-gfx` | Raw wgpu world renderer with post-FX pipeline |
| `scriptbots-bevy` | Bevy 0.17 3D renderer with terrain heightfields and agent avatars |
