# ScriptBots architecture & contribution guide

This document is the map a new contributor needs before changing anything, and the invariant
list a reviewer checks a change against. It is organised by subsystem; each section states the
**design**, the **rule** that must not be broken, and the **source of truth** — the file or type
that actually defines the behaviour, so this document can be verified against the code rather than
trusted blind. When code and this document disagree, the code wins and this document is the bug.

The overriding principle everything below serves: **a run must be reproducible, and the record of
a run must be honest about what it does and does not prove.** Most of the rules here exist because
some path once quietly broke one of those two properties.

---

## 1. The pure core step

`scriptbots-core` owns the simulation and nothing else. `WorldState::step` advances the world by
one tick through a **fixed, ordered pipeline of stages**. The order is part of the science: it
determines what each stage sees and the order in which stochastic decisions are made.

The stages, in execution order (`WorldState::step_outcome`, `crates/scriptbots-core/src/lib.rs`):

1. `stage_food_respawn`
2. `stage_interventions` — queued external effects (drought, meteor, injection) apply at the TOP
   of the tick so they are visible to every agent's senses on the tick they land.
3. `stage_aging`
4. `stage_food_dynamics`
5. `stage_sense`
6. `stage_brains`
7. `stage_actuation`
8. `stage_temperature_discomfort`
9. `stage_food`
10. `stage_combat`
11. `stage_death_cleanup`
12. `stage_reproduction`
13. `stage_population`
14. `stage_spawn_commit`
15. `stage_accumulate_food_balance`
16. `stage_accumulate_tick_events`
17. `stage_record_history`
18. `stage_narrative`
19. `stage_reset_events`
20. `stage_projection`
21. `stage_completion`

**Rules.**

- **Core is pure with respect to the outside world.** No clock reads, no filesystem, no network,
  no `set_var`, no `Math.random`, no thread-local entropy inside a stage. Everything a stage needs
  is a function of `WorldState` and the domain RNG. A stage that reads ambient state makes the run
  irreproducible, and the failure shows up as "the digest moved" with no diff to point at — the
  single most expensive kind of bug in this project.
- **Order is not an implementation detail.** Reordering stages, or reordering iteration *within* a
  stage in a way that changes RNG-consumption or floating-point-reduction order, changes the
  science. If you must, it is a deliberate, reviewed re-baseline (see §6).
- **Feature parity across build configs.** Core's default features are `parallel` (rayon) +
  `simd_wide` (wide). WASM builds core with `default-features = false`, so the `cfg(not(simd_wide))`
  scalar paths genuinely ship and **must stay semantically identical** to the SIMD paths. A change
  to one lane that is not mirrored in the other silently forks the simulation between native and
  browser.

Source of truth: `crates/scriptbots-core/src/lib.rs` (`WorldState`, the `stage_*` methods).

---

## 2. HostCore ownership (`scriptbots-runtime`)

`scriptbots-runtime::HostCore` is the **sole owner** of a running `WorldState`. Frontends, the CLI,
the REST surface, and MCP do not touch `WorldState`; they hand `HostCore` a command and read back a
snapshot. This is what makes concurrent control surfaces safe and what keeps the simulation
single-threaded-authoritative while many things observe it.

The surface (`crates/scriptbots-runtime/src/lib.rs`):

- **Commands in.** `HostCommand` is the closed set of things you may ask the host to do. Every
  command is `validate()`d before it is accepted (`CommandValidationError`), so a malformed request
  is rejected at the door rather than corrupting a tick. Accepted commands carry a `CommandId` and
  travel as a `CommandEnvelope`.
- **Status and health out.** `HostLifecycle`, `HostHealth`, `HostFault`, `HostBlocker`, and
  `HostDriveInterest` describe what the host is doing and why it might be stalled. Status
  *combinations* are themselves validated (`StatusCombinationError`) — an impossible pair (e.g.
  "healthy" and "faulted") is a bug the type system refuses to represent.
- **Events out.** `HostEventKind` is an append-only event log. Observers `poll(cursor, limit)` with
  an `EventCursor` and receive an `EventPoll`; nobody blocks the simulation to read it.

**Rules.**

- **Ownership is exclusive.** If you find yourself reaching for a `&mut WorldState` outside
  `HostCore`, stop — you are re-introducing the multi-writer races HostCore exists to prevent.
- **Commands are validated, not trusted.** New control paths add a `HostCommand` variant and its
  validation; they do not bypass the envelope.

Source of truth: `crates/scriptbots-runtime/src/lib.rs` (`HostCore`, `HostCommand`, `HostEventKind`).

---

## 3. Snapshot versus journal

Two different things flow *out* of a running world, and confusing them is a category error.

- **Snapshot** — an immutable, versioned projection of the world's *current* state for rendering,
  produced by `project_snapshot` into a `RenderSnapshot`. It is multi-subscriber: every frontend
  reads the same `Arc<RenderSnapshot>` via `latest()`, so N observers cost one projection, not N.
  A snapshot answers *"what does the world look like right now?"* and is allowed to be lossy — it
  is for drawing, not for reconstruction.
- **Journal** — the append-only stream of what *happened*: the persistence batches written to
  storage (§7) and the replay-event stream. The journal answers *"what did this run do, tick by
  tick?"* and must be complete enough to reconstruct and to audit.

**Rule.** Do not use a snapshot where you need a journal, or vice versa. A snapshot is a lossy
*view*; the journal is the *record*. Rendering from the journal would be slow and wrong; auditing
from a snapshot would silently miss everything the snapshot dropped.

Source of truth: `crates/scriptbots-runtime/src/lib.rs` (`RenderSnapshot`, `project_snapshot`,
`latest`); `crates/scriptbots-storage` (the persistence journal).

---

## 4. Brain genome and evaluator state

A brain has two parts, and both are science state:

- The **genome** — the evolved parameters (weights, topology). Reproduction copies it and mutates
  the copy.
- The **evaluator state** — the recurrent activation carried between ticks. A recurrent brain's
  next output depends on its last, so two brains with identical genomes but different evaluator
  state are **different brains** that diverge on the very next tick.

Families implement the `BrainRunner` trait (`scriptbots-core`), adapted from the `Brain` trait in
`scriptbots-brain`. Key hooks:

- `clone_runner` — duplicate including all evolved parameters. `Ok(None)` means the family is
  *non-heritable* and reproduction would spawn a fresh brain instead of the parent's.
- `mutate` — perturb in place. The default does nothing and returns `Ok(())`.
- `state_digest` — a stable hash of genome **and** evaluator state, the hook `WorldDigestV1` uses to
  see inside a brain (§6). One-way: it can prove a restored brain identical but cannot rebuild one.
- `BrainFamilyCodec::adapter_identity` — a stable versioned BLAKE3 semantic attestation owned by
  each protocol family. It identifies construction and evaluation behavior independently from the
  family/schema/codec tuple; it is not derived from addresses, Rust type names, compiler output,
  closures, or executable bytes.

**Rules.**

- **Heredity is copy AND vary, and it is proven, not declared.** `install_brains`
  (`scriptbots-app`) admits a family to the *founding population* only after probing that it both
  duplicates itself and changes under mutation. A family that clones but no-ops its `mutate` (the
  historical `ml.placeholder`) is withheld — a population that cannot evolve is not science, and
  the trait's permissive defaults make such families *invisible* unless the registry proves the
  contract. Non-heritable families stay registered for explicit selection but never found a
  population.
- **A brain's digest must cover its state, not just its family name.** Anything that compares runs
  must go through `state_digest`, never through the family label — the label is identical across a
  million ticks of divergent evolution.

Source of truth: `crates/scriptbots-core/src/lib.rs` (`BrainRunner`, `BrainBinding`,
`install_brains` in `scriptbots-app`); `crates/scriptbots-brain` (`Brain`, the MLP/DWRAON/Assembly
families).

---

## 5. RNG and determinism

Environmental and run-global stochastic decisions are drawn from one of six domain-specific
`RandomStream`s owned by `DomainStreams`. Agent-affecting decisions do not consume one shared
domain continuation: `AgentSubstreamProtocolV1` derives a fresh operation-local `SmallRngStream`
from the root seed, stable subject identity, fixed domain/operation tag, and a persisted local
ordinal. The concrete generator is xoshiro256++ on 64-bit and xoshiro128++ on 32-bit, so a `wasm32`
run legitimately draws a *different* sequence from the same derived seed. The protocol records
that exact generator identity; native and browser continuations therefore cannot be confused.

- **Both protocol layers are versioned and self-describing.** `DomainStreamsCheckpoint` carries the
  root seed, domain-derivation algorithm, fixed-object codec, and exactly six named domain fields.
  Each embedded `RandomStreamState` separately carries its concrete generator id (including the
  `rand` version) and codec; restore **refuses** either a domain-protocol mismatch or an incompatible
  stream state.
  A `rand` upgrade that changes the generator is therefore a loud, announced act —
  `crates/scriptbots-core/tests/rng_sequence_compat.rs` pins the sequence so the bump fails in a
  test whose subject *is* the dependency, rather than silently moving every digest.
- **Domain separation.** `rng_domains` derives an independent stream per `RngDomain`
  (Environment, Food, Population, Lineage, Mutation, Crossover) from the root seed, via a versioned
  FNV-1a over a *stable string tag* (never the enum discriminant — inserting a variant must not
  re-seed the others). Independent streams mean adding a draw in one domain cannot perturb another,
  so changing one domain's draw schedule leaves every other continuation fixed. Core and frontend
  callers must name the domain at every stochastic boundary; there is no fallback global stream.
- **Stable agent operations.** Existing-agent streams derive from
  `(root seed, AgentUid, AgentRngOperationV1, agent-local ordinal)`. The persisted
  `AgentRngCountersV1` owns independent next-unused ordinals for reproduction attempts, successful
  births, and brain initialization. Dense index, recycled `AgentId`, loop position, thread order,
  and wall-clock state never enter the identity.
- **Directional offspring operations.** `OffspringRngIdentityV1` is
  `(primary parent UID, optional secondary parent UID, primary-parent-local birth ordinal)`.
  Parent order is meaningful: the primary supplies base runtime/genome material and owns the
  counter, while the secondary contributes crossover material. This local birth ordinal is not the
  run-wide demographic `AgentIdentity::birth_ordinal`; the latter remains an ancestry/reporting
  sequence assigned at successful insertion.
- **One stream per operation, created once.** Body population, runtime crossover, runtime mutation,
  brain crossover, brain mutation, fallback initialization, evaluator-state crossover, and
  evaluator-state mutation have distinct stable tags. A caller constructs each required stream
  once and threads it through the complete operation instead of repeatedly deriving it or falling
  back to a shared domain stream.
- **Transactional counter ownership.** An attempted reproduction claims its local attempt ordinal;
  only an admitted offspring claims the primary parent's local birth ordinal. Failure rolls back
  the exact counter preimage together with parent energy/progress and staged population state.
  Population rollback happens before queued natural-birth refunds, preserving reverse chronological
  order when both paths touched the same parent.
- **Manifest launch binding.** `scriptbots.run-manifest.v3.3` records the root, exact six-domain
  checkpoint, `AgentSubstreamProtocolV1`, and UID-ordered `AgentRngCounterStateV1` launch rows. The
  `scriptbots.run-manifest.v3.5` bootstrap form additionally binds the tick-zero V1.6 start
  `WorldDigestV1`; the protocol/counters in that digest and manifest must describe the same launch.

**The hashing rule, learned the hard way.** Anything persisted or compared across runs uses a
**specified** hash (`characterization_fnv1a64` / a pinned FNV-1a), **never** `std::hash::DefaultHasher`
— std does not promise that algorithm across releases, so a *compiler upgrade* would silently move
the science. This bug was found feeding the characterization digest; do not reintroduce it.

Source of truth: `crates/scriptbots-core/src/lib.rs` (`RandomStream`, `SmallRngStream`,
`RandomStreamState`, `WorldState::rng`); `crates/scriptbots-core/src/rng_domains.rs`
(`RngDomain`, `DomainStreams`, `DomainStreamsCheckpoint`, `AgentSubstreamProtocolV1`,
`AgentRngCountersV1`, `OffspringRngIdentityV1`).

---

## 6. The science oracles and core checkpoint

Two digests decide whether two runs are *the same run*. `WorldCheckpointV1` is the bounded
reconstruction envelope for the core science state whose equality those digests prove.

- `CharacterizationDigestV0` — the original, explicitly legacy oracle. It *declares* its own
  limitations (in `CharacterizationLimitationsV0`, whose `superseded_by` field names
  `WorldDigestV1`): it is blind to brain weights, keys agents by the *recycled* slotmap id, and its
  RNG evidence is only a forward probe rather than a restorable continuation checkpoint.
- `WorldDigestV1` — `scriptbots.world-digest.v1.6`/codec-6 supersedes v0 with per-lane hashes:
  agents ordered by the **stable `AgentUid`**
  (not the reused slot key), brains (genome + evaluator state via `state_digest`), food, terrain,
  hydrology, the **restorable six-domain** RNG checkpoint, exact `AgentSubstreamProtocolV1`,
  UID-ordered `AgentRngCounterStateV1` rows carrying per-agent continuation counters, global
  future-affecting counters, the selected locomotion model, and the exact semantic identity
  captured for every admitted protocol adapter.
  Coverage is part of the output: if a bound brain cannot expose its state,
  `evaluator_state_covered` is false and the family is *named*, so a digest computed while blind can
  never collide with one computed while seeing. V1.6 considers protocol construction semantics
  covered only when the family identity is present; legacy factories still use their explicit
  captured-state digest.
- `WorldCheckpointV1` — captures a bounded canonical `scriptbots.world-checkpoint.v1.3`/codec-5
  `postcard+blake3-v5` envelope with an unkeyed BLAKE3 corruption checksum, only at an open,
  persistence-disabled completed boundary with no deferred host output. It carries the complete
  configuration including the locomotion model, stable-UID agents, genome/evaluator state, exact
  declarative registry roster and allocation cursor, protocol adapter identities,
  environment/effects/origins, exact
  agent-substream metadata, UID-ordered per-agent counters, global future-affecting counters, and all
  six RNG continuations.
  Restore validates the protocol, counter cardinality/order/UID correspondence, and registry recipe
  before constructing any evaluator or agent, allocates fresh physical `AgentId` values through the
  exact caller-prepared `BrainRegistry`, and rechecks the saved `WorldDigestV1`.
- `WorldStepTrace` — `scriptbots.world-step-trace.v1.6`/codec-6 carries the same embedded V1.6
  world contract at each of its six semantic capture points. Its deferred-work lane includes queued
  offspring identities/counter-relevant state, so first-divergence evidence cannot omit a claim that
  will affect the Population stage.

**Rules.**

- **`AgentUid` is identity; the slotmap key is not.** Slots are reused when agents die. Anything
  that must be stable across a restore — digests, ancestry, lineage — keys on `AgentUid`. Ordering
  a reduction or an RNG-consuming loop by dense slot index makes the science depend on allocation
  layout (a real, open determinism concern).
- **Moving a digest is a reviewed act.** A change that legitimately alters a digest (a new RNG
  domain wiring, a sense-lane cutover) records the before/after in the commit and confirms no pinned
  golden elsewhere is silently invalidated. Burying a science-wide re-baseline inside an unrelated
  commit makes it unreviewable.
- **Declare limitations; never hide them.** The honest thing v0 did — stating in the manifest what
  it does not cover — is the standard. A digest that quietly skips part of the world is worse than
  one that says it cannot see it.
- **Checkpoint bytes are data, never executable code, and a core checkpoint is not product
  resume.** `BrainAdapterIdentityV1` is a family-authored semantic attestation, not executable-byte
  authentication. A behavior change must change that identity; a serialized genome/evaluator
  interpretation change must additionally bump the family schema/codec. The envelope excludes
  storage/session ownership, retained analytics/history, configuration-audit provenance,
  UI/render state, and run-bundle discovery; application resume remains Phase 4.1.
- **The keyed-substream move is closed under pinned DSR proof.** `bd-1kxd` closed the agent-keyed
  RNG schema with native and WASM DSR evidence. `bd-2i1` closed the selected locomotion model's
  V1.6 digest/trace contract, V1.3 checkpoint, V3.5 bootstrap manifest, reviewed semantic goldens,
  and exact-class full performance baseline; its final same-class comparison passed in DSR
  `0.1.0-bd2i1-perf-compare-quiet1.20260716T160817Z`. `bd-hiv1` remains open as the separate
  correction for movement-noise and spike-speed consumers of wheel effort. Hosted workflow
  results are not accepted for these contracts.

Source of truth: `crates/scriptbots-core/src/lib.rs` (`characterization_digest_v0`,
`WorldState::world_digest_v1`, `WorldDigestV1`);
`crates/scriptbots-core/src/checkpoint.rs` (`WorldCheckpointV1`,
`WorldState::checkpoint_v1`, `WorldState::restore_checkpoint_v1`); `RunManifestV3` in
`scriptbots-app`.

---

## 7. Storage

`scriptbots-storage` is a FrankenSQLite-backed persistence layer with a strict read/write split.

- **Write path.** A live run writes through `Storage` / the durable-outbox pipeline. Admission is
  bounded by **bytes**, not just command count (a single huge batch is one command but unbounded
  memory), and in-flight bytes are capped with RAII permits released on *every* path — commit,
  refusal, timeout handoff, and shutdown — so a leak cannot slowly strangle a long run.
- **Read path.** `StorageReader` is the **only** blessed reader of a finished run. It exposes no
  mutating API, never opens a writable connection, and never competes with a live run's storage
  worker. Offline analysis (`scriptbots-analytics`, §9) goes exclusively through it.
- **Identity guard, filesystem-aware.** The writer lease refuses symlinks, multiply-linked files,
  and a database swapped underneath it (device/inode change). That inode check is *only enforced
  where the filesystem provides a stable inode*: exFAT and FAT32 synthesize an inode from the
  starting cluster, so it moves when a file is truncated-and-regrown (exactly what creating a
  database does). The guard **probes** the filesystem and, where the property is absent, skips the
  swapped-file check **loudly** (the symlink and hard-link checks still apply) rather than falsely
  accusing the database of tampering — external drives are usually exFAT.

**Rules.**

- **Offline analysis is read-only, always.** Never widen `StorageReader` into a writer.
- **The database must reconstruct the run.** A graph (e.g. ancestry) rebuilt from persisted rows
  alone must equal the one the live run held — that equivalence is what justifies the storage layer
  existing, and it is tested.
- **Test on APFS/ext4.** The dev box's default `TMPDIR` may be exFAT; run storage-backed tests with
  `TMPDIR` on APFS if you hit validated-open failures.

Source of truth: `crates/scriptbots-storage/src/lib.rs` (`Storage`, `StorageReader`, the admission
budget, the writer lease + `filesystem_has_stable_file_identity`).

---

## 8. Frontend boundaries

The frontends are consumers, never owners. They read a `RenderSnapshot` (§3) and send `HostCommand`s
(§2); none of them holds a `WorldState`.

- `scriptbots-app` — the binary a user runs: terminal (ratatui) HUD, the CLI/REST/MCP control
  surface, and run bootstrap (`install_brains`, seeding, manifest emission).
- `scriptbots-render` / `scriptbots-world-gfx` — GPU rendering (wgpu).
- `scriptbots-bevy` — the Bevy frontend.
- `scriptbots-web` — the WASM build. Builds core with `default-features = false`; **no franken
  numeric library and no franken analytics adapter may enter its dependency graph** (they are
  nightly-only / native-only). A CI guard enforces this boundary in both directions.

**Rule.** A frontend that needs the simulation to do something asks via a command and observes via a
snapshot. If a frontend needs data a snapshot does not carry, extend the snapshot projection — do
not reach past it into core.

Source of truth: the `scriptbots-app` / `-render` / `-bevy` / `-web` crates;
`ci/check_wasm_graph.sh` for the wasm boundary guard.

---

## 9. Offline science layer (`scriptbots-analytics`)

The one blessed offline reader of finished run databases: a report framework plus the `sb-analyze`
CLI. It is **native-only** (never in an app binary, never in a wasm graph) and **read-only** (§7).

- Reports register in a `Registry` and run through `StorageReader`; each execution is wrapped in a
  tracing span carrying name, params, row counts, and wall time.
- The **statistics are native and dependency-free** — bootstrap CIs, permutation tests, effect
  sizes, event certification with Benjamini-Hochberg false-discovery control, and matched-seed
  paired comparison. They were implemented natively (rather than via `fsci-stats`, which is git-only
  and nightly-only) precisely because an offline binary should not inherit a nightly-toolchain
  requirement for textbook estimators; the modules' calibration tests demonstrate the native path is
  sufficient. Distribution *fitting* is the one place a franken adapter would still earn its keep.

**Rule.** Statistics certify the science; they never enter core or a tick path. The detector in
`scriptbots-core::detect` stays hand-rolled, online, and bit-stable; certification annotates its
findings *after the fact*, offline.

Source of truth: `crates/scriptbots-analytics/src/lib.rs` (report registry, `ReaderCtx`),
`stats.rs`, `certify.rs`, `compare.rs`.

---

## Working in this repository

- **One shared working tree, many agents.** Reserve files (MCP agent-mail
  `file_reservation_paths`, exclusive) before editing; another agent's uncommitted changes appear in
  your `git status`, so stage only your own files (`git add <path>`, never `git add -A`) and never
  commit someone else's WIP. Coordinate handoffs by mail.
- **Track work in beads** via the `br` CLI (not `bd`). Mark a bead in-progress when you start, and
  close it with evidence when it lands. `bv`'s robot triage can read a stale merge artifact — verify
  its picks against `br` before acting.
- **Builds offload via RCH**; the canonical target is `x86_64-unknown-linux-gnu`. See
  `AGENTS.md` for the toolchain and the offload/lock protocol.
- **Never lose a feature to a refactor, and never make a test pass by weakening what it proves.** A
  green suite that no longer certifies the thing it names is worse than a red one.
