# scriptbots-web

WebAssembly harness for running ScriptBots simulation logic inside browsers. The crate exposes wasm-bindgen bindings that initialize a world, advance ticks, and surface simulation snapshots for rendering layers implemented in JS/WebGPU/Canvas.

## Building

```bash
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown -p scriptbots-web
```

The package producer uses pinned wasm-pack 0.15.0 and adds standard module,
declaration, and WASM exports to its generated package metadata. Run packaging
and browser checks inside the source-pinned DSR profile, with its provisioned tools:

```bash
bash scripts/package_wasm.sh /absolute/fresh/scriptbots-web-package
SCRIPTBOTS_WASM_PACKAGE_DIR=/absolute/fresh/scriptbots-web-package \
  SCRIPTBOTS_WASM_PROOF_DIR=/absolute/existing/dsr-proof \
  bash scripts/e2e_wasm_browser_stats_dom.sh
```

Both the output directory and its `.wasm-pack` sibling must be absent before a
build. The sibling retains the original generated metadata and bytes; the final
package preserves them and includes the project license notices. The demo's
existing `/pkg` import is served from the selected final package by the smoke.

The browser smoke uses already provisioned Bun and Playwright. It retains the
three injected DOM contract/scheduling controls, adds actual package loading,
and checks an isolated corrupted-WASM bootstrap refusal. Existing native/WASM
parity fixtures and tests remain separate and unchanged. No package-manager
installation or bundler is required. Original bd-azi3 acceptance remains open
until the complete source-bound DSR package/browser qualification; source wiring
and CI feedback are not release evidence. See `docs/wasm/` for the roadmap.

## Bindings overview

The crate exposes the following wasm-bindgen surface:

- `default_init_options() -> JsValue` — returns the JSON-serialisable `InitOptions` structure with defaults applied.
- `init_sim(options: JsValue) -> SimHandle` — constructs a simulation using the supplied options. Recognised fields include:
  - `population` (usize): initial spawn count (defaults to 64).
  - `seed` (u64): explicit RNG seed. It overrides `config.rng_seed`; when omitted,
    the nested config seed is preserved. If both are omitted, core seeds the world
    from entropy.
  - `world_width` / `world_height`: override the world dimensions (pixels).
  - `config`: optional full `ScriptBotsConfig` override (rarely needed; defaults are usually sufficient).
  - `snapshot_format`: `"json"` (default) or `"binary"` (Postcard-encoded `Uint8Array`).
  - `seed_strategy`: `"wander"` (default) to attach lightweight wander brains or `"none"` to start with passive agents for custom registry wiring.
  - `default_brain`: `"mlp"` to pre-bind agents to the baseline MLP implementation during seeding.
- `SimHandle::tick(steps: u32) -> JsValue` — advances the simulation and returns either JSON or a `Uint8Array` depending on `snapshot_format`.
- `SimHandle::drive(elapsedMilliseconds: number) -> JsValue` — drives the existing
  `HostCore` at monotonic elapsed time since initialization/reset. Returns its
  actual scientific-step, due and skipped-opportunity receipt. Each call retains
  the host's bounded automatic catch-up policy; invalid/backwards time or a
  nanosecond receipt beyond JavaScript's safe-integer range refuses before work.
  This is a scheduling entry point, never a render callback.
- `SimHandle::setSpeed(multiplier: number) -> JsValue` — applies a canonical host
  command in the range `0..=32` and returns its two-axis command status. Zero
  pauses automatic science. Explicit `tick(steps)` still completes exactly the
  requested manual steps without advancing the elapsed clock or changing speed.
- `SimHandle::snapshot() -> JsValue` — builds a snapshot without ticking (uses the same format toggle).
- `SimHandle::reset(seed?: number)` — rebuilds the world. An explicit seed
  overrides the nested config for that reset; omitting it falls back to the
  original `config.rng_seed` (or entropy when the nested seed is also absent).
- `SimHandle::registerBrain(kind: string)` — installs a brain preset for all agents (`"wander"`, `"mlp"`, or `"none"`).
- `decode_snapshot_binary(bytes: &[u8]) -> JsValue` — helper exposed for JS callers to convert binary snapshots back into structured data.

Snapshots are deterministic and use camelCase field names at the JavaScript boundary. They include:

- `tick`, `epoch`, and `world` metadata (dimensions, closed/open flag)
- Per-tick summary metrics (`agentCount`, `births`, `deaths`, `averageEnergy`, `averageHealth`, etc.)
- Per-agent state (`position`, `velocity`, `heading`, `energy`, `health`, `color`, `spikeLength`, `boost` flag)

The wasm crate depends on `scriptbots-core` with `default-features = false`, disabling Rayon on wasm targets. Native builds can re-enable parallelism via the `parallel` feature if needed.

The Canvas demo drives elapsed time from one independent timer. RAF only reads
the latest immutable snapshot. Its speed slider and visibility changes use host
commands; hidden pages pause and resume without catching up the hidden interval.
Browser timers and rendering still share the main JS thread, so suspended or busy
pages have no hard real-time guarantee. The receipt exposes skipped opportunities.
Full packaged-browser/DSR acceptance remains with `bd-2z0.12.3` and `bd-azi3`;
native unit tests alone do not establish browser scheduling or durability.
