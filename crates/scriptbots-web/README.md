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
  - `default_brain`: `"mlp"`, `"dwraon"`, or `"assembly"` to bind agents to a
    versioned, heritable brain family during seeding. Omitting both brain choices
    starts with MLP. Assembly is an experimental evolving program family.
  - `seed_strategy`: explicitly choose `"wander"` for the legacy movement demo or
    `"none"` for unbound agents when `default_brain` is absent. A selected
    `default_brain` takes precedence. Wander cannot export continuations.
- `SimHandle::tick(steps: u32) -> JsValue` — advances the simulation and returns either JSON or a `Uint8Array` depending on `snapshot_format`.
- `SimHandle::drive(elapsedMilliseconds: number) -> JsValue` — drives the existing
  `HostCore` at monotonic elapsed time since initialization/reset. Returns its
  actual scientific-step, due and skipped-opportunity receipt. Each call retains
  the host's bounded automatic catch-up policy; invalid/backwards time or a
  nanosecond receipt beyond JavaScript's safe-integer range refuses before work.
  This is a scheduling entry point, never a render callback.
- `SimHandle::setSpeed(multiplier: number) -> JsValue` — atomically sets speed
  and pause state through one canonical host command in `0..=32`, returning its
  two-axis status. Zero pauses; positive values resume a paused owner.
  Explicit `tick(steps)` still completes exactly the
  requested manual steps without advancing the elapsed clock or changing speed.
- `SimHandle::snapshot() -> JsValue` — builds a snapshot without ticking (uses the same format toggle).
- `SimHandle::frame() -> JsValue` — reads one immutable host publication with exact
  decimal-string identities, its dynamic snapshot, and a bounded terrain/food/water
  overview. The environment is cached by owner and layer revisions and frozen in
  JavaScript; callers cannot mutate the shared cells. This does not tick the world.
- `SimHandle::inspectGenome(uid: string, locusOffset: number, deltaOffset: number)` —
  reads a bounded page of canonical loci and retained primary-parent differences.
  Start both offsets at zero and follow the returned independent page cursors.
- `SimHandle::intervene(request)` — stages a source-bound intervention at an observed
  paused boundary. Keep the exact request for an explicit idempotent retry after an
  uncertain response; an applied receipt means queued for the next scientific tick.
- `SimHandle::interventionStatus(commandId: string, sessionId: string)` — reads the
  retained application and journal outcome. A null result means unknown.
- `SimHandle::interventionHistory(sessionId: string, epoch: string, afterSeq: string, limit: number)` —
  reads 1–64 global ecological effect records, including explicit retention gaps.
- `SimHandle::reset(seed?: number)` — rebuilds the world. An explicit seed
  overrides the nested config for that reset; omitting it falls back to the
  original `config.rng_seed` (or entropy when the nested seed is also absent).
- `SimHandle::registerBrain(kind: string)` — rebuilds the world with `"mlp"`,
  `"dwraon"`, `"assembly"`, `"wander"`, or `"none"`. Unknown or invalid presets
  leave the active owner intact.
- `SimHandle::checkpoint() -> Uint8Array` — settles the current volatile journal
  boundary and exports the existing full owner checkpoint: science, evaluators,
  command history and cadence credit. It does not advance elapsed time. Legacy
  wander evaluators are refused; MLP, DWRAON, Assembly and unbound worlds are
  supported, including empty populations.
- `restore_checkpoint(bytes: Uint8Array) -> SimHandle` — validates and restores a
  new owner, with a new command namespace and rebased elapsed clock. Failed
  imports preserve existing handles. Future science matches the saved state;
  offline wall time adds no ticks. Reset starts a new world using the loaded
  configuration, checkpoint population and saved brain family. Restoration uses
  that family's exact registry and evaluator identity; unknown or mismatched
  families are refused instead of being substituted.
- `checkpoint_byte_limit() -> number` — the canonical owner codec's file bound,
  also checked before the demo reads a selected file.
- `SimHandle::brainPreset() -> string` — the new-world brain preset for this handle.
- `decode_snapshot_binary(bytes: &[u8]) -> JsValue` — helper exposed for JS callers to convert binary snapshots back into structured data.

Snapshots are deterministic and use camelCase field names at the JavaScript boundary. They include:

- `tick`, `epoch`, and `world` metadata (dimensions, closed/open flag)
- Per-tick summary metrics (`agentCount`, `births`, `deaths`, `averageEnergy`, `averageHealth`, etc.)
- Per-agent state (`position`, `velocity`, `heading`, `energy`, `health`, `color`, `spikeLength`, `boost` flag)

The wasm crate depends on `scriptbots-core` with `default-features = false`, disabling Rayon on wasm targets. Native builds can re-enable parallelism via the `parallel` feature if needed.

The demo's **World layers** selector works in Canvas 2D and WebGPU, locally and
while connected to a native server. It offers combined terrain/food/water, terrain,
food availability, water depth and agents-only views. Environment observations
cover the exact world with at most 128 × 128 cells. Small grids preserve every
source cell; larger grids use a labelled center-sampled overview, not statistical
averages. Empty populations retain the world background. Missing environment
observations clear it, and owner/epoch/revision changes invalidate cached pixels.

Remote binary frames use `SBWS` version 2 and decode to
`scriptbots.remote-frame.v2`; the unchanged dynamic-world payload keeps its v2
codec. Only canvas frames carry the new environment field. JSON commands,
subscriptions and frame acknowledgments remain protocol version 1. Build the
viewer, WASM decoder and native server from matching sources; an old binary
stream is explicitly refused. The standalone `snapshot()` and `tick()` codecs
remain unchanged.

The Canvas demo drives elapsed time from one independent timer. RAF only reads
the latest immutable snapshot. Its speed slider and visibility changes use host
commands; hidden pages pause and resume without catching up the hidden interval.
Browser timers and rendering still share the main JS thread, so suspended or busy
pages have no hard real-time guarantee. The receipt exposes skipped opportunities.
Full packaged-browser/DSR acceptance remains with `bd-2z0.12.3` and `bd-azi3`;
native unit tests alone do not establish browser scheduling or durability.

The demo starts with **MLP**, with canonical inherited genomes and evaluator
state. **Brain for new worlds** also offers DWRAON, experimental Assembly,
Unbound and the legacy Wander movement demo; the choice takes effect on Reset.
All except Wander support saved continuations. Wander visibly disables both
save controls. **Load Continuation File** validates a saved owner before replacing
the current world, restores its saved family and pauses it. Local continuation
controls are disabled while displaying a remote server. A delayed file read
cannot replace a newer reset or remote session.

**Save in This Browser** stores one complete owner continuation in IndexedDB for
this site's origin and browser profile. The UI acknowledges the save only after
the transaction completes; quota or abort failures preserve the previous slot.
Reload discovers the saved tick and time. **Restore Saved World** explicitly
validates and installs that continuation, paused, including when a bad startup
permalink left no active owner. Saves are manual: later ticks are not autosaved.
Browser data clearing or eviction can remove the slot; download a continuation
file when a portable copy is needed. An in-flight restore cannot replace a reset,
new permalink, or remote session. Remote mode disables local save/restore.

The browser record holds bounded opaque checkpoint bytes as a Blob, allowing a
size check before materializing an ArrayBuffer. Corrupt or unsupported records,
and a stored tick that disagrees with the validated owner, leave the current
world intact. This saved continuation is separate from the owner's journal
receipt state. It does not write a SQLite image or ordered batch journal, or
promote volatile receipts to `Durable`; those original `bd-ywtv` requirements
remain open. Multiple tabs share the slot: the last completed manual save wins.

## Genome inspection and interventions

Select a creature and open **Inherited genome & parent differences** to inspect its
actual canonical genome. Loci and changed parent loci have separate page controls;
the summary covers the full compared genome. Inspection reads on selection, page
navigation and explicit refresh, without advancing the world or decoding a genome
on every frame. Pages contain at most 64 loci and 32 differences, with a 64 KiB
response ceiling and a pre-decode ceiling of 65,536 loci per genome. The parent must
still be living and retain material matching the child's recorded provenance.
Unavailable parent material is reported explicitly. Crossover comparisons describe
differences from the primary parent, without attributing all differences to mutation.

Open **Scientific intervention toolbox**, pause the world, and choose drought,
embargo, bloom, meteor, or closed/open-world policy. Ecological actions can cover
the whole world or a toroidal disc; the disc can be centered on the selected
creature. **Stage intervention at paused boundary** records a command. **Step once**
or **Resume** then advances science so the queued action can take effect. The
toolbox displays command application and journal durability separately from the
global observed-effects history. History rows have no per-command attribution.

If delivery is uncertain, **Retry original request with same ID** preserves the
original action and source guard. It does not construct a new experiment. Another
intervention remains disabled while the previous result is unknown or a queued
action has not reached a later tick. Session, epoch and control-revision guards
refuse stale commands, including concurrent Resume or Step races.

Switching to a remote server preserves the retained local world's request and
receipt for **Return to Local**. Reconnecting to the same remote endpoint also
retains its request until a fresh, painted owner frame confirms the source. A
rewound same-session source disables intervention actions while retaining the
original request. Resetting or replacing the local world explicitly starts a new
toolbox context. Only one local and one remote view are retained.

The same surfaces work against the connected native owner. Its HTTP endpoints are
`GET /api/agents/genome`, `POST /api/control/browser-intervention`,
`GET /api/control/browser-intervention/{command_id}` and
`GET /api/interventions/browser`. Genome queries carry `uid`, `session_id`, `epoch`,
`locus_offset` and `delta_offset`; history queries carry `session_id`, `epoch`,
`after_seq` and `limit`. Preserve decimal strings for 64-bit source identities and
counters, and the canonical 32-character hexadecimal command ID. Request and
response sizes, owner work, page size and browser read time are bounded.
