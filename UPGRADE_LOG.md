# Dependency Upgrade Log

This ledger records every dependency-source, version, feature, and toolchain
mutation in the ScriptBots workspace. Each entry is intentionally narrow: one
package or one inseparable family, one reviewed lockfile delta, and one explicit
rollback boundary. Research can happen in parallel; changes to manifests and
`Cargo.lock` cannot.

For each entry, record:

- the owning Bead;
- the old declaration and resolved version/source;
- the target declaration and resolved version/source;
- primary release, migration, security, MSRV, and licensing references;
- the exact allowed manifest and lockfile delta;
- focused and workspace verification evidence;
- build, binary-size, feature, and behavior impact where relevant;
- the result and a manual, path-scoped rollback procedure.

Do not combine repair work for an existing red baseline with a dependency
mutation. Stop an update when unrelated packages move, the pinned toolchain is
incompatible, scientific output changes without an approved version boundary,
licensing is unclear, or the migration crosses the review circuit breakers in
the rearchitecture plan.

## 2026-07-11 — Freeze the existing GPUI source resolution

- **Bead:** `bd-2z0.1.2`
- **Change class:** reproducibility source pin; not a GPUI version upgrade
- **Manifest:** root `Cargo.toml`
- **Before:** `gpui` used mutable Zed `branch = "main"`
- **Before resolution:** Zed commit
  `5f8a7413a31769e0882357f90dc424b3962ac72d`
- **Before lock SHA-256:**
  `789b53e73ba975a8482b14c60dc47f10648a3033c577de1d6fcf75b541ee5136`
- **Target:** the same Zed commit selected with exact
  `rev = "5f8a7413a31769e0882357f90dc424b3962ac72d"`
- **Allowed lock delta:** query normalization from `branch=main` to the exact
  `rev` for the 16 packages already resolved from that Zed commit; no registry
  version, checksum, or unrelated Git revision may change
- **Primary source:**
  <https://github.com/zed-industries/zed/commit/5f8a7413a31769e0882357f90dc424b3962ac72d>
- **Known-red before baseline:** `scriptbots-render` does not compile against
  this already-resolved commit because `flex_grow` now requires an `f32` and
  `Application::new` is absent. This pin preserves that red fingerprint; it does
  not repair or bless it.
- **Out of scope:** dated nightly selection, `rust-version`, GPUI API repair,
  removal of the unused `num-bigint-dig` patch, registry updates, and CI changes
- **Dry-run circuit breaker:** `cargo update --dry-run -p gpui --precise
  5f8a7413a31769e0882357f90dc424b3962ac72d` proposed the expected 16
  source-ID replacements but also tried to prune `windows 0.57.0` and its three
  companion registry packages. The write was stopped. Only the 16 source IDs
  were then normalized; the unrelated registry entries were preserved.
- **After lock SHA-256:**
  `c09eb8cb5b51d77ce1a5ae1e27c68ab31265d97cf43d04332d686cfbf4104337`
- **Verification:** two `cargo metadata --locked --no-deps --format-version 1`
  resolutions accepted the same lock/hash; all 16 Zed sources use the exact
  revision; no mutable Git query remains; `cargo fmt --all --check` passed; the
  default `scriptbots-app` host-target check passed with its pre-existing
  warning; the locked workspace/all-targets check reproduced the same known
  `scriptbots-render` GPUI/test-initializer failures.
- **Result:** accepted as a behavior-preserving source pin with no dependency
  version, checksum, feature, or registry-package delta
- **Rollback:** manually restore only the prior `gpui` declaration and the 16
  reviewed Zed source lines from this patch. Do not use checkout, reset, clean,
  or any operation that could overwrite unrelated work.

## 2026-07-11 — Pin the dated nightly and correct the MSRV claim

- **Bead:** `bd-2z0.1.8`
- **Change class:** toolchain reproducibility; no dependency update
- **Before toolchain:** floating `nightly`, observed as `rustc
  1.99.0-nightly (375b1431b 2026-07-10)` during the baseline
- **Target toolchain:** `nightly-2026-07-09`, installed locally as `rustc
  1.99.0-nightly (14cae6813 2026-07-08)` with `cargo 1.99.0-nightly
  (59800466c 2026-07-07)`
- **MSRV declaration:** correct the stale workspace claim from 1.85 to 1.88,
  the minimum required by already-locked production dependencies; nightly
  remains project policy
- **Allowed files:** `rust-toolchain.toml`, the workspace `rust-version`, CI and
  release toolchain inputs, and the two live documentation claims
- **Out of scope:** action-revision pinning, dependency changes, CI feature
  repair, renderer repair, and historical examples that are not project policy
- **Pre-edit proof:** locked metadata and formatting passed; a clean external
  target-dir check of default `scriptbots-app` passed with its pre-existing
  macOS helper warning; `terminal_smoke` passed 2/2; core tests reproduced the
  same 41-pass/1-stale-grid failure as the floating-nightly baseline
- **Post-edit proof:** the repository selects the target commit with bare
  `rustc -Vv`; locked metadata, formatting, and the default application check
  pass; `terminal_smoke` passes 2/2; the full locked workspace/all-targets check
  reaches `scriptbots-render` and reproduces the known GPUI/test-initializer
  errors without an MSRV or language failure; `Cargo.lock` remains unchanged.
- **Clippy baseline:** the dated toolchain deterministically stops first on the
  existing collapsible nested `if` in `scriptbots-index`. The floating nightly
  had instead stopped on newer `chunks_exact_to_as_chunks` findings in core,
  demonstrating why the lint toolchain must be pinned. Neither pre-existing lint
  is repaired in this mutation.
- **CI alignment:** all eight CI/release toolchain inputs name
  `nightly-2026-07-09`; their action-revision pins remain a separate CI bead.
- **Result:** accepted; toolchain, manifest MSRV, CI inputs, and live docs now
  state one reproducible contract
- **Rollback:** manually restore only the dated channel, `rust-version`, CI
  toolchain inputs, and live documentation lines from this reviewed patch.

## 2026-07-11 — Complete dependency disposition and feature-graph ledger

- **Bead:** `bd-2z0.8.1`
- **Change class:** read-only dependency research and disposition; no manifest
  or lock mutation
- **Inventory instant:** 2026-07-11, with the crates.io sparse index refreshed
  immediately before the final dry run
- **Primary registry source:** <https://crates.io/> and each version-specific
  crate link in the tables below
- **Repository/source evidence:** Cargo metadata, `Cargo.lock`, all twelve
  workspace manifests (the virtual root plus eleven packages), exact source
  searches, feature metadata, and target-specific Cargo trees

### Scope and graph facts

- Eleven workspace packages expose 140 package dependency edges: 118 external
  registry/Git edges and 22 local path edges.
- The external edges comprise 113 normal and five development edges. There are
  no build dependencies, thirteen optional external edges, and seven
  target-specific external edges.
- The virtual root adds 31 workspace dependency policies and one crates.io
  patch. Two policies (`derive_more` and `fastrand`) are not inherited
  anywhere; four ML policies duplicate literal declarations in
  `scriptbots-brain-ml`.
- The external edges expose 71 dependency keys because `rand08` renames a
  second `rand` line; they resolve 70 unique directly named crates. The lock contains
  1,184 packages with 1,033 unique names; 114 names have multiple versions.
  Eighteen packages are Git-sourced: sixteen from the exact Zed revision and
  two immutable transitive Zed revisions.
- A fully online `cargo update --dry-run --verbose` proposed zero compatible
  package updates and listed 47 packages behind their newest release. This is
  expected: each is held by a major/pre-1.0 boundary, a coupled exact-version
  family, or another graph constraint. For example, `csv 1.4.0` satisfies the
  workspace's range but cannot resolve while NeuroFlow requires `csv ~1.2`.
- The same dry run proves that the `num-bigint-dig` Git patch is unused. An
  offline dry run also proposes pruning stale `ordered-float 5.0.0`; neither
  dry run wrote the lock.
- Registry metadata reports no incompatible direct license. The accepted set is
  MIT, Apache-2.0, MIT/Apache dual licensing, Zlib, and Unlicense/MIT. A dash in
  the MSRV column means the release does not declare one, so source review and
  pinned-toolchain CI remain mandatory.

Disposition codes used below:

- **KEEP** — used, already at the newest compatible resolution; retain until
  its owning family bead.
- **REMOVE** — source/feature inspection proves this declaration contributes no
  behavior; remove under `bd-2z0.8.2`.
- **PARTIAL REMOVE** — remove only the named dead occurrence; other owners use
  the same crate.
- **HOLD** — deliberate compatibility or maintenance-risk hold.
- **MIGRATE** — latest stable crosses a breaking or coupled-family boundary and
  requires the named gate.
- **PIN** — immutable source is intentional.
- **RELOCATE** — keep the crate but correct target/feature scope.

### Exhaustive external dependency ledger

Every unique external crate is listed once. The owner column names every
declaring package when occurrences have different dispositions.

| Crate | Owners and resolved graph | Latest stable; license; MSRV | Disposition, owner bead, and update gate |
|---|---|---|---|
| [anyhow](https://crates.io/crates/anyhow/1.0.103) | app, Bevy, web; 1.0.103 | 1.0.103; MIT/Apache; 1.68 | KEEP; `bd-2z0.8.6`; raise the manifest floor only with CLI error-path tests. |
| [async-trait](https://crates.io/crates/async-trait/0.1.89) | app; 0.1.89 | 0.1.89; MIT/Apache; 1.56 | KEEP; `bd-2z0.8.7`; transport-stack proof. |
| [axum](https://crates.io/crates/axum/0.8.9) | app; 0.8.9 | 0.8.9; MIT; 1.80 | KEEP; `bd-2z0.8.7`; HTTP/API/MCP conformance. |
| [bevy](https://crates.io/crates/bevy/0.19.0) | Bevy adapter; 0.17.3 | 0.19.0; MIT/Apache; 1.95 | MIGRATE 0.17→0.18→0.19; `bd-2z0.8.11`; live render goldens, feature minimization, MSRV review. |
| [bevy_mesh](https://crates.io/crates/bevy_mesh/0.19.0) | Bevy adapter; 0.17.3 | 0.19.0; MIT/Apache; undeclared | MIGRATE only with Bevy; `bd-2z0.8.11`. |
| [bevy_post_process](https://crates.io/crates/bevy_post_process/0.19.0) | Bevy adapter; 0.17.3 | 0.19.0; MIT/Apache; undeclared | MIGRATE only with Bevy; `bd-2z0.8.11`. |
| [bytemuck](https://crates.io/crates/bytemuck/1.25.1) | world-gfx; 1.25.1 | 1.25.1; Zlib/Apache/MIT; undeclared | KEEP; `bd-2z0.8.11`; GPU buffer layout and readback tests. |
| [candle-core](https://crates.io/crates/candle-core/0.11.0) | optional brain-ml feature; 0.4.1; duplicate unused root policy | 0.11.0; MIT/Apache; undeclared | REMOVE root policy now; MIGRATE/reintroduce backend only when it performs inference; `bd-2z0.8.2` then `.8.10`. |
| [candle-nn](https://crates.io/crates/candle-nn/0.11.0) | optional brain-ml feature; 0.4.1; duplicate unused root policy | 0.11.0; MIT/Apache; undeclared | REMOVE root policy now; MIGRATE only with Candle core and real model tests; `bd-2z0.8.2` then `.8.10`. |
| [clap](https://crates.io/crates/clap/4.6.1) | app; 4.6.1 | 4.6.1; MIT/Apache; 1.85 | KEEP; `bd-2z0.8.6`; help and CLI contract goldens. |
| [criterion](https://crates.io/crates/criterion/0.8.2) | core dev; 0.7.0 | 0.8.2; Apache/MIT; 1.86 | MIGRATE; `bd-2z0.8.13`; isolated benchmark harness and baseline review. |
| [crossbeam-channel](https://crates.io/crates/crossbeam-channel/0.5.16) | storage; 0.5.16 | 0.5.16; MIT/Apache; 1.60 | KEEP until bounded-writer migration; `bd-2z0.8.9`. |
| [crossfire](https://crates.io/crates/crossfire/3.1.19) | app command bus; 2.1.10 | 3.1.19; Apache-2.0; 1.79 | MIGRATE or replace; `bd-2z0.8.7` after command semantics and Asupersync decision. |
| [crossterm](https://crates.io/crates/crossterm/0.29.0) | app direct 0.27.0; graph also 0.28.1 and 0.29.0 | 0.29.0; MIT; 1.63 | MIGRATE/converge through FrankenTUI; `bd-2z0.8.8`; PTY, input, paste, resize, and terminal-restore tests. |
| [csv](https://crates.io/crates/csv/1.4.0) | app via root; 1.2.2 | 1.4.0; Unlicense/MIT; 1.73 | HOLD: NeuroFlow pins `~1.2`; `bd-2z0.8.9` with `.8.10`; export corpus tests. |
| [direction](https://crates.io/crates/direction/0.19.1) | core map generation; 0.18.1 | 0.19.1; MIT; undeclared | HOLD with WFC/rand08; `bd-2z0.8.5`; map-generation conformance before coupled migration. |
| [duckdb](https://crates.io/crates/duckdb/1.10504.0) | app/control CLI and storage; 1.10504.0 bundled | 1.10504.0; MIT; 1.85.1 | KEEP current; `bd-2z0.8.9`; first make storage optional/bounded, then schema and native-build matrix. |
| [futures-intrusive](https://crates.io/crates/futures-intrusive/0.5.0) | world-gfx; 0.5.0 | 0.5.0; MIT/Apache; undeclared | REMOVE; no source reference; `bd-2z0.8.2`. |
| [futures-util](https://crates.io/crates/futures-util/0.3.32) | app; 0.3.32 | 0.3.32; MIT/Apache; 1.71 | KEEP; `bd-2z0.8.7`; streaming/API tests. |
| [getrandom](https://crates.io/crates/getrandom/0.4.3) | core activation-only 0.3.4 with `wasm_js`; web wasm32 activation-only 0.2.17 with `js`; graph also 0.4.3 | 0.4.3; MIT/Apache; 1.85 | HOLD both direct activation edges: they feature-unify the rand_core 0.9/0.6 getrandom lines despite having no API references. MIGRATE only after RandomStream and wasm32 graph proof in `.8.4/.8.13`. |
| [glam](https://crates.io/crates/glam/0.33.2) | world-gfx; 0.30.10 | 0.33.2; MIT/Apache; 1.68.2 | REMOVE; no source reference; `bd-2z0.8.2`. |
| [gpui](https://github.com/zed-industries/zed/commit/5f8a7413a31769e0882357f90dc424b3962ac72d) | render; Git package 0.2.2 at exact Zed revision | published 0.2.2; Apache-2.0; undeclared | PIN; `bd-2z0.8.12`; retain/retire only after direct-texture and live-window gate. |
| [image](https://crates.io/crates/image/0.25.10) | app, Bevy, render, world-gfx dev; 0.25.10 | 0.25.10; MIT/Apache; 1.88 | PARTIAL REMOVE app occurrence; keep three render/test uses; `bd-2z0.8.2` and `.8.11`. |
| [js-sys](https://crates.io/crates/js-sys/0.3.103) | web; 0.3.103 | 0.3.103; MIT/Apache; 1.77 | KEEP; `bd-2z0.8.13`; wasm-pack/browser proof. |
| [kiddo](https://crates.io/crates/kiddo/5.3.2) | optional index feature; 4.2.1 but no implementation reference | 5.3.2; MIT/Apache; 1.85 | HOLD/remove fictional feature or implement first; `bd-2z0.8.5`; neighbor-query conformance. |
| [kira](https://crates.io/crates/kira/0.12.1) | optional render audio; 0.10.8 | 0.12.1; MIT/Apache; undeclared | MIGRATE after renderer boundary; `bd-2z0.8.11`; cpal feature audit and audio smoke. |
| [libc](https://crates.io/crates/libc/0.2.186) | app Unix priority helper; 0.2.186 | 0.2.186; MIT/Apache; 1.65 | RELOCATE to Unix target; `bd-2z0.8.2/.8.13`; macOS/Linux startup tests. |
| [mcp-protocol-sdk](https://crates.io/crates/mcp-protocol-sdk/0.5.1) | app; 0.5.1 | 0.5.1; MIT; 1.85 | KEEP/PIN range; `bd-2z0.8.7`; protocol conformance and lifecycle tests before any source change. |
| [mimalloc](https://crates.io/crates/mimalloc/0.1.52) | app optional but default-enabled; 0.1.52 | 0.1.52; MIT; undeclared | KEEP optional, remove from semantic default until measured; `bd-2z0.8.6`; allocator/platform benchmark. |
| [naga](https://crates.io/crates/naga/30.0.0) | render activation-only direct 27.0.3 enables `termcolor` on WGPU's Naga; GPUI graph 26.0.0 | 30.0.0; MIT/Apache; 1.87 | HOLD: no API reference is expected for a feature-unification edge. Characterize shader diagnostics before removal; later migrate only with WGPU in `.8.11`. |
| [neuroflow](https://crates.io/crates/neuroflow/0.2.0) | brain-neuro; 0.2.0 | 0.2.0; MIT; undeclared | HOLD maintenance risk; `bd-2z0.8.10`; genome/evaluator-state and CSV constraint decision. |
| [num_cpus](https://crates.io/crates/num_cpus/1.17.0) | app, core, storage; 1.17.0 | 1.17.0; MIT/Apache; undeclared | KEEP pending centralized thread budget; `bd-2z0.8.5/.8.9`. |
| [ordered-float](https://crates.io/crates/ordered-float/5.3.0) | index uses 4.6.0; core occurrence dead; stale lock has 5.0.0 | 5.3.0; MIT; 1.63 | PARTIAL REMOVE core; prune stale lock; migrate live index only with spatial/digest tests; `bd-2z0.8.2/.8.5`. |
| [owo-colors](https://crates.io/crates/owo-colors/4.3.0) | app uses 4.3.0; Bevy occurrence dead | 4.3.0; MIT; 1.81 | PARTIAL REMOVE Bevy occurrence; keep app; `bd-2z0.8.2/.8.6`. |
| [pollster](https://crates.io/crates/pollster/1.0.1) | world-gfx and optional render 0.4.0; GPUI graph 0.2.5 | 1.0.1; Apache/MIT; 1.69 | MIGRATE only with renderer/WGPU; `bd-2z0.8.11`. |
| [postcard](https://crates.io/crates/postcard/1.1.3) | web; 1.1.3 | 1.1.3; MIT/Apache; undeclared | KEEP; `bd-2z0.8.3/.8.13`; serialization golden and browser round trip. |
| [rand](https://crates.io/crates/rand/0.10.2) | app, brain, brain-ml, brain-neuro, core, render, web use workspace 0.9.5; core also aliases 0.8.7; graph has 0.10.2. App's occurrence is integration-test-only. | 0.10.2; MIT/Apache; 1.85 | RELOCATE app occurrence to dev first in `.8.2`; MIGRATE the live family only behind named RandomStream and fixed-seed digests in `.8.4`; preserve rand08 until WFC moves. |
| [ratatui](https://crates.io/crates/ratatui/0.30.2) | app declared alpha range, resolves 0.30.2 | 0.30.2; MIT; 1.88 | Declare truth, then replace/converge through FrankenTUI; `bd-2z0.8.8`; golden/PTY matrix. |
| [rayon](https://crates.io/crates/rayon/1.12.0) | core optional uses 1.12.0; brain occurrence dead | 1.12.0; MIT/Apache; 1.80 | PARTIAL REMOVE brain; keep core behind explicit product feature/thread budget; `bd-2z0.8.2/.8.5`. |
| [reqwest](https://crates.io/crates/reqwest/0.13.4) | app control CLI; 0.12.28 | 0.13.4; MIT/Apache; 1.85 | MIGRATE; `bd-2z0.8.7`; explicit rustls/default-feature decision and real control E2E. |
| [ron](https://crates.io/crates/ron/0.12.2) | app direct 0.8.1; graph also 0.10.1 | 0.12.2; MIT/Apache; 1.64 | MIGRATE; `bd-2z0.8.3`; scenario/config corpus golden round trips. |
| [rstar](https://crates.io/crates/rstar/0.13.0) | optional index feature; 0.12.2 but no implementation reference | 0.13.0; MIT/Apache; 1.85 | HOLD/remove fictional feature or implement first; `bd-2z0.8.5`; neighbor-query conformance. |
| [serde](https://crates.io/crates/serde/1.0.228) | app, brain, brain-ml, brain-neuro, core, index, render, storage, web; 1.0.228; brain-ml and render occurrences dead | 1.0.228; MIT/Apache; 1.56 | PARTIAL REMOVE two dead occurrences and the redundant brain-neuro `derive` feature repeat; keep seven owners; `bd-2z0.8.2`, floor update in `.8.3`. |
| [serde-wasm-bindgen](https://crates.io/crates/serde-wasm-bindgen/0.6.5) | web; 0.6.5 | 0.6.5; MIT; undeclared | KEEP; `bd-2z0.8.13`; canonical browser serialization tests. |
| [serde_json](https://crates.io/crates/serde_json/1.0.150) | app, brain-neuro, core, storage; 1.0.150 | 1.0.150; MIT/Apache; 1.71 | KEEP; `bd-2z0.8.3`; manifest/replay/config goldens. |
| [serde_path_to_error](https://crates.io/crates/serde_path_to_error/0.1.20) | app; 0.1.20 | 0.1.20; MIT/Apache; 1.61 | KEEP; `bd-2z0.8.3/.8.7`; structured error tests. |
| [serial_test](https://crates.io/crates/serial_test/3.5.0) | app dev; 3.5.0 | 3.5.0; MIT; 1.68 | KEEP; `bd-2z0.8.13`; test-only. |
| [slotmap](https://crates.io/crates/slotmap/1.1.1) | app, Bevy, core, storage, web; 1.1.1 | 1.1.1; Zlib; 1.58 | KEEP; `bd-2z0.8.5`; stable-ID/checkpoint boundary tests. |
| [smallvec](https://crates.io/crates/smallvec/1.15.2) | app; 1.15.2 | 1.15.2; MIT/Apache; undeclared | KEEP; `bd-2z0.8.6`; command/snapshot behavior. |
| [supports-color](https://crates.io/crates/supports-color/3.0.2) | app TUI; 3.0.2 | 3.0.2; Apache-2.0; 1.70 | KEEP pending FrankenTUI integration; `bd-2z0.8.8`. |
| [tch](https://crates.io/crates/tch/0.24.0) | optional brain-ml feature 0.18.1; duplicate unused root policy | 0.24.0; MIT/Apache; undeclared | REMOVE root policy now; MIGRATE/reintroduce only with exact LibTorch/PyTorch 2.11 provisioning and real inference; `bd-2z0.8.2/.8.10`. |
| [tempfile](https://crates.io/crates/tempfile/3.27.0) | app dev; 3.27.0 | 3.27.0; MIT/Apache; 1.63 | KEEP; `bd-2z0.8.13`; test-only. |
| [thiserror](https://crates.io/crates/thiserror/2.0.18) | app, brain, brain-ml, core, index, storage at 2.0.18; brain and brain-ml dead; graph also 1.0.69 | 2.0.18; MIT/Apache; 1.68 | PARTIAL REMOVE two dead occurrences; keep four; `bd-2z0.8.2/.8.6`. |
| [tokio](https://crates.io/crates/tokio/1.52.3) | app; 1.52.3 | 1.52.3; MIT; 1.71 | KEEP until runtime boundary decision; `bd-2z0.8.7`; lifecycle/cancellation tests. |
| [tokio-stream](https://crates.io/crates/tokio-stream/0.1.18) | app; 0.1.18 | 0.1.18; MIT; 1.71 | KEEP; `bd-2z0.8.7`; streaming/MCP tests. |
| [toml](https://crates.io/crates/toml/1.1.2+spec-1.1.0) | app direct 0.8.23; graph also 1.1.2 | 1.1.2+spec-1.1.0; MIT/Apache; 1.85 | MIGRATE; `bd-2z0.8.3`; config corpus and formatted-output review. |
| [tracing](https://crates.io/crates/tracing/0.1.44) | app, Bevy, render, storage, world-gfx at 0.1.44; storage occurrence dead | 0.1.44; MIT; 1.65 | PARTIAL REMOVE storage occurrence; keep four; `bd-2z0.8.2/.8.6`. |
| [tracing-subscriber](https://crates.io/crates/tracing-subscriber/0.3.23) | app; 0.3.23 | 0.3.23; MIT; 1.65 | KEEP; `bd-2z0.8.6`; logging snapshot/CLI smoke. |
| [tract-onnx](https://crates.io/crates/tract-onnx/0.23.4) | optional brain-ml feature; 0.21.10 exact-family graph; duplicate unused root policy | 0.23.4; MIT/Apache; 1.91 | REMOVE root policy now; MIGRATE full Tract family only with real ONNX fixtures and MSRV gate; `bd-2z0.8.2/.8.10`. |
| [unicode-width](https://crates.io/crates/unicode-width/0.2.2) | dead app direct 0.1.14; graph also 0.2.2 | 0.2.2; MIT/Apache; 1.66 | REMOVE direct occurrence; terminal stack retains transitive 0.2; `bd-2z0.8.2/.8.8`. |
| [utoipa](https://crates.io/crates/utoipa/5.5.0) | app; 5.5.0 | 5.5.0; MIT/Apache; 1.75 | KEEP; `bd-2z0.8.7`; OpenAPI golden and server E2E. |
| [utoipa-swagger-ui](https://crates.io/crates/utoipa-swagger-ui/9.0.2) | app; 9.0.2 | 9.0.2; MIT/Apache; 1.75 | KEEP; `bd-2z0.8.7`; embedded asset/API smoke. |
| [wasm-bindgen](https://crates.io/crates/wasm-bindgen/0.2.126) | web; 0.2.126 with deprecated serde-serialize feature | 0.2.126; MIT/Apache; 1.77 | KEEP version, remove deprecated feature after serde-wasm-bindgen proof; `bd-2z0.8.13`. |
| [wasm-bindgen-test](https://crates.io/crates/wasm-bindgen-test/0.3.76) | web dev; 0.3.76 | 0.3.76; MIT/Apache; 1.77 | KEEP; `bd-2z0.8.13`; browser tests. |
| [wayland-client](https://crates.io/crates/wayland-client/0.31.14) | app Linux-only; 0.31.14 | 0.31.14; MIT; 1.71 | KEEP target-scoped; `bd-2z0.8.13`; Linux startup/backend-selection proof. |
| [wfc](https://crates.io/crates/wfc/0.10.7) | core; 0.10.7 | 0.10.7; MIT; undeclared | HOLD with direction/rand08; `bd-2z0.8.5`; map-generation conformance or replacement. |
| [wgpu](https://crates.io/crates/wgpu/30.0.0) | world-gfx/render direct 27.0.1; GPUI graph 26.0.1 | 30.0.0; MIT/Apache; 1.87 | MIGRATE 27→28→29→30 only after renderer ownership; `bd-2z0.8.11`; per-major GPU/live-capture matrix. |
| [wide](https://crates.io/crates/wide/1.5.0) | core optional; 0.8.3 | 1.5.0; Zlib/Apache/MIT; 1.89 | MIGRATE after eyesight/scalar oracle; `bd-2z0.8.5`; declared MSRV increase plus scalar/SIMD digest and benchmark gates. |
| [windows-sys](https://crates.io/crates/windows-sys/0.61.2) | app unconditional direct 0.59.0; graph has 0.45/0.48/0.52/0.59/0.61 | 0.61.2; MIT/Apache; 1.71 | RELOCATE direct dependency to Windows target, then migrate; `bd-2z0.8.2/.8.13`; Windows priority/startup proof. |
| [winit](https://crates.io/crates/winit/0.30.13) | world-gfx target-specific; 0.30.13 | 0.30.13; Apache-2.0; 1.70 | KEEP target-scoped; `bd-2z0.8.11`; native window/GPU matrix. |

### Root-only policies and patch

These declarations do not appear as additional package edges, so they are
called out separately instead of being hidden in the crate rows:

| Declaration | Evidence | Disposition |
|---|---|---|
| `derive_more = 1.0.0` | only occurrence is the virtual root; registry latest is [2.1.1](https://crates.io/crates/derive_more/2.1.1), MIT, MSRV 1.81 | REMOVE in `bd-2z0.8.2`. |
| `fastrand = 2.1.0` | no workspace package inherits it; registry latest is [2.4.1](https://crates.io/crates/fastrand/2.4.1), MIT/Apache, MSRV 1.63; MCP still brings a transitive copy | REMOVE root policy in `bd-2z0.8.2`; do not claim the transitive crate disappears. |
| root Candle/Tract/Tch entries | brain-ml repeats literal versions and never uses the root policies | REMOVE four root policies in `bd-2z0.8.2`; the optional backend surface is a separate `.8.10` decision. |
| `[patch.crates-io] num-bigint-dig` | Cargo records it under `[[patch.unused]]`; no resolved package uses the Git revision; registry latest is [0.9.1](https://crates.io/crates/num-bigint-dig/0.9.1), MIT/Apache, MSRV 1.65 | REMOVE patch and its unused lock record in `bd-2z0.8.2`. |

### Local path dependency ledger

All 22 path edges are immutable local workspace edges; none is a build
dependency. Their version/source risk is therefore the workspace commit rather
than crates.io.

| Consumer | Path dependencies | Disposition and owner |
|---|---|---|
| app | unconditional brain, core, storage; optional brain-ml, brain-neuro, render, Bevy | Keep the executable edges, but make products explicit. Remove placeholder ML/Neuro/allocator defaults under `.8.10`; storage/server/TUI separation belongs to runtime beads; GUI edges remain optional under `.8.11/.8.12`. |
| Bevy | core | Replace direct world ownership with canonical snapshot/control ports under `bd-2z0.7.3`. |
| brain | core | Keep; brain protocol/evaluator separation under `bd-2z0.3`. |
| brain-ml | brain, core | Keep only if real adapters survive `bd-2z0.8.10`; current placeholder must not imply inference. |
| brain-neuro | brain, core | Keep behind explicit product feature; prove genome/evaluator state in `bd-2z0.3/.8.10`. |
| core | index | Keep; spatial oracle and implementation contract under `bd-2z0.2`. |
| render | brain, core, storage, optional world-gfx | Move the integration-test-only brain edge to dev in `.8.2`. Rendering must consume snapshots instead of owning simulation/storage; migrate under `bd-2z0.7/.9`, retain optional world-gfx only through the GPUI decision. |
| storage | core | Replace broad world coupling with event/snapshot records under `bd-2z0.5`. |
| web | core without defaults; brain without defaults plus MLP | Keep as the browser product boundary; prove feature closure and deterministic serialization under `bd-2z0.12/.8.13`. |
| world-gfx | core | Narrow to renderer-neutral snapshot buffers under `bd-2z0.7/.9`. |

### Feature and target graph findings

| Surface | Observed graph | Decision/gate |
|---|---|---|
| app default | enables placeholder `ml`, NeuroFlow, and mimalloc; enables neither GUI | Replace with named product features. Placeholder ML cannot remain a default success path; `bd-2z0.8.10`. |
| app no-default | only five of 43 normal direct edges are optional, so 38 remain; the macOS normal closure contains 325 distinct package-version nodes (Linux 332). It still compiles brain defaults, core parallel/SIMD defaults, storage plus bundled DuckDB/Arrow, server/MCP/HTTP/TLS, Ratatui/Crossterm, dead image/unicode-width, and unconditional windows-sys. | `--no-default-features` is not a minimal/headless product. Remove dead edges in `.8.2`, then make storage/server/TUI/core modes explicit across `.8.7-.8.9`. |
| core default | `parallel + simd_wide` | Keep until scalar/serial oracle exists; then test four explicit product lanes in `.8.5`. |
| brain default | MLP + DWRAON + `experimental`, where experimental enables Assembly | Remove “experimental by default” ambiguity after brain-family conformance; `bd-2z0.3/.8.6`. |
| brain-ml | backend features only enable heavy crates and change a label; tick still copies sensors to outputs | Quarantine from defaults and rebuild honest adapters under `.8.10`; do not upgrade fake backends. |
| index | `rstar` and `kd` only enable dependencies; source implements only the uniform grid | Implement conformance-backed adapters or remove fictional features before updating; `.8.5`. |
| render | default enables custom world-wgpu; direct GPUI is unconditional. On macOS the GUI closure feature-unifies render's WGPU 27 Metal edge with world-gfx's unconditional WGPU 27 Vulkan edge, enabling both backends. GUI-only grows the normal closure from 325 to 551 nodes. | Make world-gfx WGPU features target-correct, keep rich renderers isolated from normal defaults, repair current compile truth, then choose one primary renderer under `.8.11/.8.12`. |
| Bevy | app's optional edge is cleanly gated, but the Bevy crate uses its full default feature set; Bevy-only reaches 531 normal package-version nodes and includes audio, GLTF, gamepad, web/Android, X11/Wayland, and broad GPU backends | Disable Bevy defaults and admit only features exercised by live renderer tests during `.8.11`. GUI+Bevy currently reaches 701 nodes and carries WGPU/Naga 26 and 27 together. |
| platform | app scopes Wayland to Linux, but libc and windows-sys are unconditional | Relocate libc/windows-sys first; verify macOS, Linux, and Windows product graphs in `.8.2/.8.13`. |
| web | its direct core edge disables defaults, but its brain edge enables brain defaults and brain in turn re-enables core defaults; parallel/Rayon and SIMD therefore leak back into the supposed browser subset. Its activation-only getrandom 0.2 edge is intentional; the deprecated wasm-bindgen serde feature is not. | Close the transitive default-feature leak, keep the getrandom feature-unification effect, remove the deprecated serde feature, and prove a real wasm32/browser lane in `.8.13`. |
| binaries | app-level reqwest/rand/direct DuckDB serve only the control CLI or integration tests, yet every binary inherits them; no binary declares `required-features` | Move test-only rand now, then establish explicit control-cli/storage features and binary gates under `.8.7/.8.9`. |

### Directly relevant duplicate-version register

The lock has 114 multiply resolved names. These are the direct families that
need an explicit reason or convergence gate:

| Name | Locked versions | Reason / convergence owner |
|---|---|---|
| crossterm | 0.27.0, 0.28.1, 0.29.0 | direct legacy app, GPUI-era stack, current Ratatui; FrankenTUI `.8.8`. |
| getrandom | 0.2.17, 0.3.4, 0.4.3 | activation edges for two rand_core generations plus transitive latest; retain until feature-unification replacements are proven in `.8.4/.8.13`. |
| naga | 26.0.0, 27.0.3 | GPUI/WGPU 26 plus WGPU 27 `termcolor` activation edge; characterize, then renderer gate `.8.11/.8.12`. |
| ordered-float | 4.6.0, 5.0.0 | live index 4 plus stale lock residue 5; `.8.2/.8.5`. |
| pollster | 0.2.5, 0.4.0 | GPUI plus custom renderer; `.8.11/.8.12`. |
| rand | 0.8.7, 0.9.5, 0.10.2 | WFC adapter, workspace API, transitive latest; RandomStream `.8.4`. |
| ron | 0.8.1, 0.10.1 | direct config plus transitive consumer; serialization `.8.3`. |
| thiserror | 1.0.69, 2.0.18 | transitive legacy plus direct workspace; keep until owning upstreams move. |
| toml | 0.8.23, 1.1.2 | direct config plus newer transitive consumer; serialization `.8.3`. |
| unicode-width | 0.1.14, 0.2.2 | dead direct app plus Ratatui stack; remove direct in `.8.2`. |
| wgpu | 26.0.1, 27.0.1 | exact GPUI graph plus custom/Bevy-era renderer; `.8.11/.8.12`. |
| windows-sys | 0.45, 0.48, 0.52, 0.59, 0.61 | ecosystem generations; only direct 0.59 is controllable, first target-scope it in `.8.2`. |

### Acceptance and next mutation

This ledger assigns every registry, Git, patch, optional, dev, target, and local
path declaration a disposition and owner. The next allowed mutation is
`bd-2z0.8.2`: remove only proven dead/redundant declarations in serialized
manifest slices. For every slice:

1. capture the pre-edit lock hash and targeted metadata/tree;
2. edit one owner manifest (or the inseparable root-policy group);
3. run a dry lock update and reject unrelated package motion;
4. run locked metadata, the owning crate's feature/target checks and tests,
   format, Clippy to the known baseline, and UBS on changed files;
5. record the exact lock subtree removed before starting the next slice.

No version migration in this ledger is authorized merely by being newer.

## 2026-07-11 — Remove unused root policies and crates.io patch

- **Bead:** `bd-2z0.8.2`
- **Change class:** root-manifest hygiene; no resolved dependency version change
- **Before manifest SHA-256:**
  `2c5b23b852ed6492d69b1df35f894bb57ac0f86dfe36241a79dde8f8072ded12`
- **After manifest SHA-256:**
  `594c59c2c4adb4033613d9539dd1b9f193f57c5273beb8b5e8d2615762d48fd2`
- **Removed unused policies:** `derive_more`, `fastrand`, and the root
  `candle-core`, `candle-nn`, `tract-onnx`, and `tch` entries. No
  workspace package inherited any of these six policies. Transitive
  `derive_more`/`fastrand` packages and the literal optional ML backend
  dependencies intentionally remain.
- **Removed patch:** exact-revision `num-bigint-dig 0.8.4`, which Cargo
  reported as unused and represented only as `[[patch.unused]]`.
- **Allowed lock delta:** delete exactly that five-line unused-patch record.
  Lock SHA-256 changed from
  `c09eb8cb5b51d77ce1a5ae1e27c68ab31265d97cf43d04332d686cfbf4104337`
  to
  `025f88732b8ea47df52a3b4ffcf65f1c063b0c7f0b51d9588976adba5a03cfad`.
- **Circuit breaker exercised:** a targeted Cargo cleanup also proposed
  collapsing WGPU/GPUI's `ordered-float 5.0.0` line onto 4.6.0. That
  semver-valid renderer-family change was outside this slice and was preserved;
  no ordered-float package or dependency reference changed.
- **Verification:** locked/offline metadata, a locked core tree, and a fresh
  host-target locked/offline `scriptbots-core` check pass; the post-edit
  offline dry run locks zero packages and proposes no write; `cargo fmt
  --all --check` and `git diff --check` pass.
- **Result:** accepted; root policy surface is truthful and the lock contains no
  unused patch record.
- **Rollback:** manually restore only the nine root-manifest lines and the
  five-line lock record shown by this commit. Do not restore any unrelated
  working-tree path.
