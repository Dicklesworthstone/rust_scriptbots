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

### Current vs Transitional vs Target State

- **[Target State: Dedicated HostCore Ownership (`bd-k7nq`)]**: In the target architecture,
  `scriptbots-runtime::HostCore` is the **sole owner** of a running `WorldState`. Frontends, the CLI,
  the REST surface, and MCP never touch `WorldState` directly; they hold a `HostClient` handle, send
  validated commands across a message-passing channel, and read back an immutable snapshot. This
  makes concurrent control surfaces thread-safe and preserves a single simulation authority while
  multiple observers inspect it.
- **[Current & Transitional State: `SharedWorld` Bridge (`bd-2z0.4.9`)]**: In current production code,
  `scriptbots-app` wraps `WorldState` in `SharedWorld` (`pub type SharedWorld = Arc<Mutex<WorldState>>`,
  `crates/scriptbots-app/src/lib.rs`). Control surfaces (`ControlHandle`, `crates/scriptbots-app/src/control.rs`),
  HTTP/MCP servers (`crates/scriptbots-app/src/servers.rs`), and the TUI loop (`crates/scriptbots-app/src/terminal/mod.rs`)
  lock this mutex directly. Closed bead `bd-2z0.4.9` completed a vital transitional milestone by
  migrating control commands to use typed, validated envelopes (`CommandEnvelope`, `CommandId`) and
  eliminating discarded receipts across Bevy, CLI, REST, and MCP. The September 3 correction
  inferred migration from closed `bd-k7nq`; source inspection on September 5 contradicts it:
  `ControlHandle` still stores `SharedWorld` and `with_world` locks it. Host ownership and
  frontend cutover remain with `bd-pcfj` and `bd-88yj`. The closed preparation bead is not
  evidence that those production callers were migrated.

The runtime control surface (`crates/scriptbots-runtime/src/lib.rs`):

- **Commands in.** `HostCommand` is the closed set of operations you may request from the host. Every
  command is `validate()`d before acceptance (`CommandValidationError`), ensuring malformed requests
  are rejected before reaching simulation stages. Accepted commands carry a `CommandId` and travel
  inside a `CommandEnvelope`.
- **Status and health out.** `HostLifecycle`, `HostHealth`, `HostFault`, `HostBlocker`, and
  `HostDriveInterest` describe host execution state and stall conditions. Status combinations are
  themselves validated (`StatusCombinationError`) — contradictory states (e.g. "healthy" and "faulted")
  are rejected by typed invariants.
- **Events out.** `HostEventKind` is an append-only event journal. Observers `poll(cursor, limit)` with
  an `EventCursor` and receive an `EventPoll` without blocking the simulation tick loop.

**Rules.**

- **[Target Invariant] Ownership is exclusive.** Do not introduce new paths holding `&mut WorldState`
  outside `HostCore`. Transitional paths must lock `SharedWorld` strictly within bounded helpers and
  prepare for migration to `HostClient`; `ControlHandle` is among those transitional paths
  tracked under `bd-pcfj`/`bd-88yj`.
- **Commands are validated, not trusted.** New control operations must add a `HostCommand` variant and
  implement validation; they must never bypass the `CommandEnvelope` or discard receipts (`bd-2z0.4.9`).

Source of truth: `crates/scriptbots-runtime/src/lib.rs` (`HostCore`, `HostCommand`, `HostEventKind`,
`CommandEnvelope`, `CommandId`); `crates/scriptbots-app/src/lib.rs` (`SharedWorld`);
`crates/scriptbots-app/src/control.rs` (`ControlHandle`); transitional receipt fixes under `bd-2z0.4.9`;
HostClient migration: preparation in closed `bd-k7nq`; actual ownership transfer under
`bd-pcfj`/`bd-88yj` (source correction, 2026-09-05).

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

Versioned evolutionary families implement `BrainFamilyCodec` and `BrainEvaluator` in
`scriptbots-core`. Existing `Brain` implementations in `scriptbots-brain` also have legacy
`BrainRunner` adapters. These contracts have different admission and checkpoint capabilities:

- Legacy `clone_runner` may decline copying; its default mutation is a no-op. Those permissive
  hooks are insufficient for versioned founder admission.
- `BrainFamilyAdapter` creates, clones, mutates and crosses validated genome material while
  preserving caller-supplied provenance. Evaluator state has a separate envelope and explicit
  offspring inheritance policy.
- `BrainEvaluator::checkpoint_state` captures future-affecting dynamic state. Reconstructing
  through the family validates the genome/state pairing; a digest alone cannot reconstruct it.
- `BrainFamilyCodec::adapter_identity` — a stable versioned BLAKE3 semantic attestation owned by
  each protocol family. It identifies construction and evaluation behavior independently from the
  family/schema/codec tuple; it is not derived from addresses, Rust type names, compiler output,
  closures, or executable bytes.

**Rules.**

- **Heredity needs behavioral evidence.** `install_brains` admits versioned protocol families;
  registry-derived heredity tests verify their declared loci and mutation behavior. The legacy
  Neuroflow adapter is withheld from mixed founders and remains explicitly selectable. The ML
  sensor-copy placeholder is not installed. Registration is not a mutation-quality probe.
- **A brain's digest covers parameters and dynamic state.** Use the scientific digest/checkpoint
  APIs rather than a family label; the label alone stays unchanged as evolution proceeds.

Source of truth: `crates/scriptbots-core/src/lib.rs` (`BrainRunner`, `BrainBinding`,
`install_brains` in `scriptbots-app`); `crates/scriptbots-brain/src/lib.rs` (`Brain`, the MLP/DWRAON/Assembly
families).

---

## 5. RNG and determinism

Environmental and run-global stochastic decisions are drawn from one of six domain-specific
`RandomStream`s owned by `DomainStreams`. Agent-affecting decisions do not consume one shared
domain continuation: `AgentSubstreamProtocolV1` derives a fresh operation-local `SmallRngStream`
from the root seed, stable subject identity, fixed domain/operation tag, and a persisted local
ordinal. The concrete generator is the project-pinned Xoshiro256++/SplitMix64 lane on every target;
the `64` in its identity names the generator word width, not the target pointer width. Native and
`wasm32` therefore draw the same sequence from the same derived seed. Historical Xoshiro256++
checkpoints remain compatible because their state and codec are identical, while legacy 32-bit
Xoshiro128++ continuations keep their distinct identity and are rejected rather than reinterpreted.

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
  `scriptbots.run-manifest.v3.6` bootstrap form additionally binds the tick-zero V1.7 start
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
- `WorldDigestV1` — `scriptbots.world-digest.v1.7`/codec-7 supersedes v0 with per-lane hashes:
  agents ordered by the **stable `AgentUid`**
  (not the reused slot key), brains (genome + evaluator state via `state_digest`), food, terrain,
  hydrology, the **restorable six-domain** RNG checkpoint, exact `AgentSubstreamProtocolV1`,
  UID-ordered `AgentRngCounterStateV1` rows carrying per-agent continuation counters, global
  future-affecting counters, the selected locomotion model, and the exact semantic identity
  captured for every admitted protocol adapter.
  Coverage is part of the output: if a bound brain cannot expose its state,
  `evaluator_state_covered` is false and the family is *named*, so a digest computed while blind can
  never collide with one computed while seeing. V1.7 excludes the orphaned
  `sense_max_neighbors` placeholder, which never influenced the C++-parity sensing transition, and
  considers protocol construction semantics
  covered only when the family identity is present; legacy factories still use their explicit
  captured-state digest.
- `WorldCheckpointV1` — captures a bounded canonical `scriptbots.world-checkpoint.v1.3`/codec-6
  `postcard+blake3-v6` envelope with an unkeyed BLAKE3 corruption checksum, only at an open,
  persistence-disabled completed boundary with no deferred host output. It carries the complete
  configuration including the locomotion model, stable-UID agents, genome/evaluator state, exact
  declarative registry roster and allocation cursor, protocol adapter identities,
  environment/effects/origins, exact
  agent-substream metadata, UID-ordered per-agent counters, global future-affecting counters, and all
  six RNG continuations.
  Restore validates the protocol, counter cardinality/order/UID correspondence, and registry recipe
  before constructing any evaluator or agent, allocates fresh physical `AgentId` values through the
  exact caller-prepared `BrainRegistry`, and rechecks the saved `WorldDigestV1`.
- `WorldStepTrace` — `scriptbots.world-step-trace.v1.7`/codec-7 carries the same embedded V1.7
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
  `0.1.0-bd2i1-perf-compare-quiet1.20260716T160817Z`. `bd-hiv1` is CLOSED ("Restore legacy
  wheel-output semantics for movement noise and spike speed"; status corrected 2026-09-03 —
  `bd-docs-status-truth-sweep-v2iw`). Hosted workflow
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

Frontends are simulation consumers, never authors of core science state. They read immutable
`RenderSnapshot`s (§3) and submit `HostCommand`s (§2).

### Current vs Target State

- **[Current & Transitional State]**: WGPU (`scriptbots-render` / `scriptbots-world-gfx`), Bevy
  (`scriptbots-bevy`), and WASM (`scriptbots-web`) consume projected `RenderSnapshot`s and communicate
  via commands. In `scriptbots-app`, however, the TUI runner (`TerminalRenderer`) and HTTP/MCP servers
  still hold a `SharedWorld` mutex bridge to drive ticks and query state directly.
- **[Target State (`bd-k7nq` + `bd-pcfj`/`bd-88yj`)]**: All frontends, including the TUI and headless server runners,
  interact strictly via `HostClient` over asynchronous channels. No frontend crate will have link or
  runtime access to `WorldState` or `SharedWorld`. `ControlHandle` still owns the transitional
  mutex, and Bevy still owns a separate simulation worker. Their cutover remains open under
  `bd-pcfj`/`bd-88yj` and the frontend migration beads.

Crate responsibilities:

- `scriptbots-app` — the binary a user runs: terminal (ratatui) HUD, the CLI/REST/MCP control
  surface, and run bootstrap (`install_brains`, seeding, manifest emission).
- `scriptbots-render` / `scriptbots-world-gfx` — native GPU rendering (wgpu/GPUI).
- `scriptbots-bevy` — alternative ECS rendering frontend.
- `scriptbots-web` — browser WASM target. Builds core with `default-features = false`; **no franken
  numeric library and no franken analytics adapter may enter its dependency graph** (they are
  nightly-only / native-only). A CI guard enforces this boundary in both directions.

**Rule.** A frontend that needs the simulation to do something asks via a command and observes via a
snapshot. If a frontend needs data a snapshot does not carry, extend the snapshot projection — do
not reach past it into core.

Source of truth: `crates/scriptbots-app/`, `crates/scriptbots-render/`, `crates/scriptbots-bevy/`,
`crates/scriptbots-web/`; `ci/check_wasm_graph.sh` for the wasm boundary guard; target unification
under `bd-k7nq`.

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

## 10. Executable extension recipes

The Rust blocks in this section are documentation tests of `scriptbots-app`:
`cargo test -p scriptbots-app --doc`. Each is a complete program using workspace dependencies.
They exercise the named library boundary; a passing local host example does not prove that the
application's transitional GUI, TUI and server callers have migrated to that boundary.

### 10.1 Recipe: Adding a New Brain Family

New evolutionary families implement `BrainFamilyCodec` and `BrainEvaluator` from
`crates/scriptbots-core/src/lib.rs`. `BrainFamilyAdapter` supplies provenance-preserving genome
creation through its blanket implementation. The legacy `Brain` and `BrainRunner` traits are
separate APIs; implementing them alone does not install a versioned evolutionary family.

This small recurrent neuron has two heritable byte parameters and one byte of dynamic state.
It exposes canonical loci, mutates both parameters through independent primary-rate gates,
crosses gain from the first parent with bias from the second, and resets offspring state.
Its inspection method honestly returns unavailable. Batch execution is optional and omitted.

<!-- recipe:brain -->
```rust
use scriptbots_core::{
    BrainAdapterIdentityV1, BrainEnvelopeKind, BrainEvaluator, BrainEvaluatorStateEnvelope,
    BrainFamilyAdapter, BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope,
    BrainGenomeMaterial, BrainHeredityCapabilityV1, BrainInspection, BrainInspectionError,
    BrainInspectionSnapshot, BrainLocusSchemaIdentityV1, BrainMutationTrialGroupV1,
    BrainProtocolError, BrainProvenance, INPUT_SIZE, MutationRates, OUTPUT_SIZE,
    OffspringStatePolicy, RandomStream, ScriptBotsConfig, SmallRngStream,
    WorldState, genome_diff::{Locus, LocusValue},
};

use rand::Rng;

const GENOME_BYTES: usize = 2;
struct CustomFamily { id: BrainFamilyId }
struct CustomEvaluator {
    id: BrainFamilyId, genome: [u8; GENOME_BYTES],
    genome_hash: [u8; blake3::OUT_LEN], previous: u8,
}

impl CustomFamily {
    fn invalid(&self, kind: BrainEnvelopeKind, detail: &str) -> BrainProtocolError {
        BrainProtocolError::InvalidPayload {
            kind, family_id: self.id.clone(), detail: detail.into(),
        }
    }
    fn genome(&self, value: &BrainGenomeEnvelope) -> Result<[u8; GENOME_BYTES], BrainProtocolError> {
        value.require_protocol(&self.id, 1, 1)?;
        value.payload().try_into()
            .map_err(|_| self.invalid(BrainEnvelopeKind::Genome, "expected gain and bias bytes"))
    }
    fn state(&self, value: &BrainEvaluatorStateEnvelope)
        -> Result<([u8; blake3::OUT_LEN], u8), BrainProtocolError> {
        value.require_protocol(&self.id, 2, 2)?;
        match value.payload().split_last() {
            Some((previous, digest)) if digest.len() == blake3::OUT_LEN =>
                Ok((digest.try_into().unwrap(), *previous)),
            _ => Err(self.invalid(BrainEnvelopeKind::EvaluatorState, "expected genome hash and state byte")),
        }
    }
}

impl BrainEvaluator for CustomEvaluator {
    fn family_id(&self) -> &BrainFamilyId { &self.id }
    fn evaluate(&mut self, sensors: &[f32; INPUT_SIZE]) -> Result<[f32; OUTPUT_SIZE], BrainProtocolError> {
        if sensors.iter().any(|value| !value.is_finite()) {
            return Err(BrainProtocolError::InvalidPayload {
                kind: BrainEnvelopeKind::EvaluatorState,
                family_id: self.id.clone(), detail: "nonfinite sensors".into(),
            });
        }
        let gain = f32::from(self.genome[0]) / 255.0;
        let bias = f32::from(self.genome[1]) / 255.0;
        let next = ((sensors[0].clamp(0.0, 1.0) + f32::from(self.previous) / 255.0)
            * gain + bias) / 3.0;
        self.previous = (next.clamp(0.0, 1.0) * 255.0) as u8;
        let mut output = [0.0; OUTPUT_SIZE];
        output[0] = f32::from(self.previous) / 255.0;
        output[1] = output[0];
        Ok(output)
    }
    fn inspect(&self, _: BrainInspection) -> Result<Option<BrainInspectionSnapshot>, BrainInspectionError> {
        Ok(None)
    }
    fn checkpoint_state(&self) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        let mut payload = self.genome_hash.to_vec();
        payload.push(self.previous);
        BrainEvaluatorStateEnvelope::new(self.id.clone(), 2, 2, payload)
    }
}

impl BrainFamilyCodec for CustomFamily {
    fn family_id(&self) -> &BrainFamilyId { &self.id }
    fn adapter_identity(&self) -> BrainAdapterIdentityV1 {
        BrainAdapterIdentityV1::from_semantic_descriptor(&self.id, 2,
            b"byte-neuron:v2;gain,bias;state=material-hash+recurrent-u8;clamped-input;divide-three;reset;xor-mutation")
    }
    fn heredity_capability(&self) -> BrainHeredityCapabilityV1 {
        BrainHeredityCapabilityV1::locus_capable(
            BrainLocusSchemaIdentityV1::from_semantic_descriptor(&self.id, 1, b"hyper0=gain;hyper1=bias"),
            vec![BrainMutationTrialGroupV1::new(u32::try_from(GENOME_BYTES).unwrap(), 1, 1)],
        )
    }
    fn random_genome_material(&self, rng: &mut dyn RandomStream) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        BrainGenomeMaterial::new(1, 1, rng.next_u32().to_le_bytes()[..GENOME_BYTES].to_vec())
    }
    fn validate_genome(&self, genome: &BrainGenomeEnvelope) -> Result<(), BrainProtocolError> {
        self.genome(genome).map(|_| ())
    }
    fn genome_loci(&self, genome: &BrainGenomeEnvelope) -> Result<Vec<(Locus, LocusValue)>, BrainProtocolError> {
        Ok(self.genome(genome)?.into_iter().enumerate().map(|(index, value)|
            (Locus::Hyper(u8::try_from(index).unwrap()), LocusValue::Scalar(f32::from(value)))).collect())
    }
    fn validate_evaluator_state(&self, state: &BrainEvaluatorStateEnvelope) -> Result<(), BrainProtocolError> {
        self.state(state).map(|_| ())
    }
    fn mutate_genome_material(&self, genome: &BrainGenomeEnvelope, rates: MutationRates,
        rng: &mut dyn RandomStream) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        let mut bytes = self.genome(genome)?;
        for value in &mut bytes {
            if rng.random::<f32>() < rates.primary {
                *value ^= (rng.next_u32() % 255 + 1) as u8;
            }
        }
        BrainGenomeMaterial::new(1, 1, bytes.to_vec())
    }
    fn crossover_genomes_material(&self, left: &BrainGenomeEnvelope, right: &BrainGenomeEnvelope,
        _: &mut dyn RandomStream) -> Result<BrainGenomeMaterial, BrainProtocolError> {
        BrainGenomeMaterial::new(1, 1, vec![self.genome(left)?[0], self.genome(right)?[1]])
    }
    fn initial_state(&self, genome: &BrainGenomeEnvelope, _: &mut dyn RandomStream)
        -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        self.validate_genome(genome)?;
        let mut payload = genome.material_hash().as_bytes().to_vec();
        payload.push(0);
        BrainEvaluatorStateEnvelope::new(self.id.clone(), 2, 2, payload)
    }
    fn offspring_state_policy(&self) -> OffspringStatePolicy { OffspringStatePolicy::Reset }
    fn offspring_state(&self, child: &BrainGenomeEnvelope, parents: &[&BrainEvaluatorStateEnvelope],
        rng: &mut dyn RandomStream) -> Result<BrainEvaluatorStateEnvelope, BrainProtocolError> {
        for parent in parents { self.validate_evaluator_state(parent)?; }
        self.initial_state(child, rng)
    }
    fn evaluator(&self, genome: &BrainGenomeEnvelope, state: &BrainEvaluatorStateEnvelope)
        -> Result<Box<dyn BrainEvaluator>, BrainProtocolError> {
        let genes = self.genome(genome)?;
        let (state_hash, previous) = self.state(state)?;
        let expected_hash = *genome.material_hash().as_bytes();
        if state_hash != expected_hash {
            return Err(self.invalid(BrainEnvelopeKind::EvaluatorState, "state belongs to another genome"));
        }
        Ok(Box::new(CustomEvaluator {
            id: self.id.clone(), genome: genes, genome_hash: expected_hash, previous,
        }))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let family = CustomFamily { id: BrainFamilyId::new("custom-byte-neuron")? };
    let mut rng = SmallRngStream::seed_from_u64(42);
    let parent = family.random_genome(BrainProvenance::default(), &mut rng)?;
    let copied = family.clone_genome(&parent, BrainProvenance::default())?;
    assert_eq!(copied.payload(), parent.payload());
    let changed = family.mutate_genome(&parent, MutationRates { primary: 1.0, ..Default::default() },
        BrainProvenance::default(), &mut rng)?;
    assert_ne!(changed.payload(), parent.payload());
    let initial = family.initial_state(&parent, &mut rng)?;
    let changed_initial = family.initial_state(&changed, &mut rng)?;
    let mut before = family.evaluator(&parent, &initial)?;
    let mut after = family.evaluator(&changed, &changed_initial)?;
    let mut before_outputs = Vec::new();
    let mut after_outputs = Vec::new();
    for sensor in [0.0, 0.5, 1.0, 0.0, 1.0, 0.5] {
        before_outputs.push(before.evaluate(&[sensor; INPUT_SIZE])?);
        after_outputs.push(after.evaluate(&[sensor; INPUT_SIZE])?);
    }
    assert_ne!(before_outputs, after_outputs, "the changed genome must affect evaluation");
    let crossed = family.crossover_genomes(&parent, &changed, BrainProvenance::default(), &mut rng)?;
    assert_eq!(crossed.payload(), &[parent.payload()[0], changed.payload()[1]]);
    let mut live = family.evaluator(&parent, &initial)?;
    let input = [1.0; INPUT_SIZE];
    assert!(live.evaluate(&input)?[0] > 0.0);
    let checkpoint = family.checkpoint_evaluator_for_genome(&parent, live.as_ref())?;
    let encoded = serde_json::to_vec(&checkpoint)?;
    let decoded = serde_json::from_slice(&encoded)?;
    let mut restored = family.evaluator(&parent, &decoded)?;
    assert_eq!(live.evaluate(&input)?, restored.evaluate(&input)?);
    assert_eq!(live.checkpoint_state()?, restored.checkpoint_state()?);
    let foreign_genome_rejected = family.evaluator(&changed, &checkpoint).is_err();
    assert!(foreign_genome_rejected, "foreign-genome checkpoint must be rejected");
    let offspring = family.offspring_state(&crossed, &[&checkpoint], &mut rng)?;
    assert_eq!(family.state(&offspring)?.1, 0);
    assert!(family.evaluator(&crossed, &offspring).is_ok());
    let mut world = WorldState::new(ScriptBotsConfig { rng_seed: Some(42), ..Default::default() })?;
    let identity = family.id.clone();
    let key = world.register_brain_family("custom-byte-neuron", Box::new(family))?;
    assert_eq!(world.brain_registry().family(key).unwrap().family_id(), &identity);
    scriptbots_app::seed_founding_population(&mut world, &[key])?;
    let founders = world.agents().iter_handles().collect::<Vec<_>>();
    assert!(!founders.is_empty());
    for agent in &founders {
        let genome = world.agent_brain_genome(*agent).expect("founder must have a protocol genome");
        assert_eq!(genome.family_id(), &identity);
        assert!(world.agent_brain_evaluator_state(*agent)?.is_some());
    }
    world.step()?;
    assert_eq!(world.tick(), scriptbots_core::Tick(1));
    println!("{}", serde_json::json!({
        "family": identity, "bound_founders": founders.len(), "tick": world.tick().0,
        "parent_payload": parent.payload(), "mutated_payload": changed.payload(),
        "before_outputs": before_outputs, "after_outputs": after_outputs,
        "checkpoint_blake3": blake3::hash(&encoded).to_hex().to_string(),
        "checkpoint_genome_hash": parent.material_hash(), "foreign_genome_rejected": foreign_genome_rejected,
    }));
    Ok(())
}
```

Register through `WorldState::register_brain_family`, then pass the returned keys to
`seed_founding_population`. This is a library extension path. `BrainPreset` in
`crates/scriptbots-app/src/brains.rs` is a closed CLI enum; `--brain custom` does not exist.
Adding a shipped preset requires explicitly wiring that enum, `install_brains`, founder admission
and the registry-derived heredity proof. Registration validates protocol and locus declarations;
it does not by itself prove mutation quality, reproduction or ecological usefulness.

Advance the adapter semantic identity whenever evaluation/construction behavior changes, and
schema/codec versions when byte interpretation changes. The example checkpoints an evaluator;
whole-world checkpoints additionally bind the family registry and adapter identity. The state
payload binds the core's canonical material hash, which excludes lineage provenance. A changed
genome cannot reuse the parent's recurrent state; offspring reset creates a new binding. Foreign
family IDs, malformed payload lengths and unsupported versions must fail before evaluation.

---

### 10.2 Recipe: Adding a New Scenario or Environmental Intervention

Scenarios configure initial world terrain, environmental parameters, population founding grids, and
timed interventions (such as droughts, floods, meteors, and predator injections).

#### 1. Define a Scenario Document (`ScenarioDocumentV1`)
Save this document as `scenarios/drought_challenge.toml`, or use it as an input to the
library example below. Parsing validates the document envelope; configuration composition and
world construction must still validate the actual fields and ranges.

<!-- recipe:scenario-document -->
```toml
schema = "scriptbots.scenario.v1"
schema_version = 1
id = "drought-challenge-v1"
description = "A food-growth drought followed by recovery"
bootstrap_ticks = 100

[config]
world_width = 400
world_height = 400
food_cell_size = 50
initial_food = 1.0
food_max = 5000.0
food_growth_rate = 0.05

[[interventions]]
tick = 500
set = { food_growth_rate = 0.001 }

[[interventions]]
tick = 1000
set = { food_growth_rate = 0.05 }
```

#### 2. Normalize, seed and execute the document

`seed_founding_population` in `crates/scriptbots-app/src/lib.rs` installs the deterministic founder
grid using admitted registry keys. Scheduled config patches run at completed-tick boundaries.
This example reads the literal TOML above, executes both scheduled changes, and compares the
drought world with a matched-seed world that receives no patches. The changed food cells prove
that the growth setting has a consumer. The first 100 steps include the declared bootstrap.

<!-- recipe:scenario -->
```rust
use scriptbots_app::{
    BrainPreset, ScenarioDocumentV1, apply_scenario_interventions, install_brains,
    seed_founding_population, precedence::{ConfigLayerKind, ConfigLayerStatement, resolve_config_layers},
};
use scriptbots_core::{ScriptBotsConfig, WorldState};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let guide = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/ARCHITECTURE.md"));
    let section = guide.split_once("<!-- recipe:scenario-document -->").unwrap().1;
    let text = section.split_once("```toml\n").unwrap().1.split_once("\n```").unwrap().0;
    let document = ScenarioDocumentV1::parse_toml(text.as_bytes())?;
    let defaults = serde_json::to_value(ScriptBotsConfig::default())?;
    let resolve = |document: &ScenarioDocumentV1| {
        resolve_config_layers(&defaults, &[
            ConfigLayerStatement { kind: ConfigLayerKind::File,
                label: document.id.clone(), fields: document.config.clone() },
            ConfigLayerStatement { kind: ConfigLayerKind::Cli,
                label: "example-seed".into(), fields: serde_json::json!({"rng_seed": 42}) },
        ]).merged
    };
    let mut current = resolve(&document);
    let config: ScriptBotsConfig = serde_json::from_value(current.clone())?;
    let mut drought = WorldState::new(config.clone())?;
    let mut control = WorldState::new(config)?;
    for world in [&mut drought, &mut control] {
        let roster = install_brains(world, BrainPreset::Mlp)?;
        seed_founding_population(world, roster.population())?;
    }
    let mut observed = Vec::new();
    for tick in 0..=1000 {
        assert_eq!(drought.tick().0, tick);
        let applied = apply_scenario_interventions(&mut drought, &mut current,
            &document.interventions, tick)?;
        if applied != 0 { observed.push((tick, applied, drought.config().food_growth_rate)); }
        drought.step()?;
        control.step()?;
        if tick == 499 { assert_eq!(drought.food().cells(), control.food().cells()); }
        if tick == 500 { assert_ne!(drought.food().cells(), control.food().cells()); }
    }
    assert_eq!(observed, vec![(500, 1, 0.001_f32), (1000, 1, 0.05_f32)]);
    assert_eq!(drought.tick().0, 1001);

    // Mutate the literal input, then pass it through the same production resolver.
    let unknown = text.replacen("food_max", "food_max_typo", 1);
    let invalid = ScenarioDocumentV1::parse_toml(unknown.as_bytes())?;
    assert!(serde_json::from_value::<ScriptBotsConfig>(resolve(&invalid)).is_err());
    println!("scenario={} input_blake3={} applied={observed:?} final_tick={}",
        document.id, blake3::hash(text.as_bytes()), drought.tick().0);
    Ok(())
}
```

#### 3. Toroidal interventions

`WorldState::enqueue_intervention` accepts the production `Intervention` enum from core's
`src/lib.rs`, and applies it at the next step. The separate record types in
`crates/scriptbots-core/src/interventions.rs` are not that queue's input type.

<!-- recipe:meteor -->
```rust
use scriptbots_core::{Intervention, Region, ScriptBotsConfig, Tick, WorldState};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = WorldState::new(ScriptBotsConfig::default())?;
    world.enqueue_intervention(Intervention::Meteor {
        region: Region::Disc { x: 640.0, y: 360.0, radius: 120.0 },
        lethality: 1.0, scorch: 0.5,
    })?;
    world.step()?;
    let applied = world.applied_interventions().back().expect("meteor application record");
    assert_eq!(applied.kind, "meteor");
    assert_eq!(applied.tick, Tick(1));
    assert!(applied.cells_affected > 0);
    assert!(world.enqueue_intervention(Intervention::Meteor {
        region: Region::All, lethality: 1.0, scorch: 2.0,
    }).is_err());
    Ok(())
}
```

Interventions execute deterministically at the start of the tick in `stage_interventions`, before
agent senses are evaluated.

#### 4. Execution & Failure Handling
- **CLI Launch:** after saving the document, `cargo run -p scriptbots-app -- --scenario scenarios/drought_challenge.toml`.
- **REST Control:** Inspect scenario via `GET /api/scenario` or apply presets via `POST /api/presets/apply`.
- **Failure Modes:**
  - `ScenarioRunError::EmptyBrainRoster`: Attempted to seed founders without registered brains.
  - `ScenarioRunError::FounderNotFinite`: Initial positions or parameters produced non-finite floats.
  - `ScenarioRunError::Intervention`: Scheduled config patch failed validation at target tick.

---

### 10.3 Recipe: Adding a New Frontend Backend

A new frontend consumes a `HostClient` port, subscribes to immutable snapshots, and sends commands
with stable IDs. `project_snapshot` in `crates/scriptbots-runtime/src/lib.rs` builds a
`ClientProjection` from a `RenderSnapshot`; it does not produce the source snapshot. Source
snapshots are shared, while distinct camera/selection requests can incur distinct projection work.

This complete example embeds the real same-thread host and explicitly drives its scheduler.
A threaded frontend uses `HostThread`/`ChannelHostPort` instead. The host owns the world after
construction; frontend reads and commands go through `HostClient`. The selected journal is
volatile, so its receipt must say `CommittedVolatile`, never `Durable`.

<!-- recipe:frontend -->
```rust
use scriptbots_app::{BrainPreset, install_brains, seed_founding_population};
use scriptbots_core::{ScriptBotsConfig, Tick, WorldState};
use scriptbots_runtime::{
    ApplicationState, CommandEnvelope, CommandId, HostClient, HostCommand, HostCore,
    HostCoreOptions, HostSessionId, JournalState, ManualHostDriver, ManualInstant, PlaybackSnapshot,
    ProjectionCamera, ProjectionClientId, ProjectionDetail, ProjectionLimits,
    ProjectionRanking, ProjectionRequest, ProjectionSelection, ProjectionViewport, project_snapshot,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 400, world_height: 400, food_cell_size: 50,
        rng_seed: Some(42), ..Default::default()
    })?;
    let roster = install_brains(&mut world, BrainPreset::Mlp)?;
    seed_founding_population(&mut world, roster.population())?;
    let mut host = HostCore::new(HostSessionId::new(42), world, HostCoreOptions {
        initial_playback: PlaybackSnapshot { paused: true, speed_multiplier: 1.0 },
        ..Default::default()
    })?;
    let mut client = HostClient::new(host.local_port());
    let mut subscription = client.subscribe_snapshots();
    let initial = client.poll_snapshot(&mut subscription)?.expect("initial publication");
    let id = CommandId::new(42);
    let envelope = CommandEnvelope::new(id, HostCommand::Step);
    let admitted = client.submit(envelope.clone())?;
    assert_eq!(admitted.application(), &ApplicationState::Admitted);
    assert_eq!(initial.world.tick, 0);
    host.drive(ManualInstant::from_nanos(0))?;
    host.drive(ManualInstant::from_nanos(1))?;
    let receipt = client.command_status(id)?.expect("retained command status");
    let ApplicationState::Applied(applied) = receipt.application() else {
        panic!("step did not apply: {receipt:?}");
    };
    assert_eq!(receipt.command_id(), id);
    assert_eq!(receipt.journal(), &JournalState::CommittedVolatile);
    assert_eq!(applied.tick, Tick(1));
    let snapshot = client.poll_snapshot(&mut subscription)?.expect("step publication");
    assert_eq!(snapshot.world.tick, applied.tick.0);
    assert_eq!(snapshot.revisions, applied.revisions);
    assert_eq!(snapshot.last_applied_command, Some(id));
    let request = ProjectionRequest {
        client_id: ProjectionClientId::new(1), viewport: ProjectionViewport { width: 80, height: 36 },
        camera: ProjectionCamera { center: [200.0, 200.0], zoom: 1.0 },
        selection: ProjectionSelection::default(), detail: ProjectionDetail::Vitals,
        chart_window: 1, chart_points: 1, top_k: 4, ranking: ProjectionRanking::Energy,
    };
    let projection = project_snapshot(&snapshot, &request, ProjectionLimits::default())?;
    assert_eq!(projection.source.session_id, snapshot.session_id);
    assert_eq!(projection.source.host, applied.revisions);
    assert!(!projection.visible_agents.is_empty());
    assert_eq!(client.submit(envelope)?, receipt, "retry must preserve the original result");
    host.drive(ManualInstant::from_nanos(2))?;
    assert_eq!(host.latest_snapshot().world.tick, 1, "retry must not step twice");
    assert!(HostCommand::SetSpeed(f32::NAN).validate().is_err());
    let mut invalid_viewport = request;
    invalid_viewport.viewport.width = 0;
    assert!(project_snapshot(&snapshot, &invalid_viewport, ProjectionLimits::default()).is_err());
    println!("command={id:?} applied={applied:?} journal={:?} visible={}",
        receipt.journal(), projection.visible_agents.len());
    Ok(())
}
```

The REST status route is `GET /api/control/status/{command_id}`. It exposes separate application
and journal axes. The transitional `ControlHandle` still holds `SharedWorld`; this example does
not certify that HTTP path or a GUI/PTY/browser session. `HostCommand::Step` has no count field.
Invalid speed, mismatched session, status contradictions and projection limits return typed
errors. A slow client polls its snapshot subscription for the newest publication; event-stream
gaps require the separate cursor/catch-up protocol.

---

## 11. Architectural Invariants, Boundaries, and Guards Map

Every architectural invariant in ScriptBots is backed by a concrete source symbol, an executable
guard or test, or an explicit open tracking bead.

| Architectural Invariant | Subsystem | Source Symbol / File | Guard / Test | Tracking Bead & Status |
| :--- | :--- | :--- | :--- | :--- |
| **Core Science Purity** (no clock, network, or filesystem in ticks) | `scriptbots-core` | `WorldState::step_outcome` (`crates/scriptbots-core/src/lib.rs`) | `tests/world_determinism.rs` | Enforced; closed in `bd-16g.11` |
| **Deterministic Stage Order** (21 ordered simulation stages) | `scriptbots-core` | `WorldState::step_outcome` (`crates/scriptbots-core/src/lib.rs`) | `tests/world_digest_v1.rs` golden digest | Enforced; closed in `bd-3n7p` |
| **Exclusive Simulation Ownership** (HostCore sole owner) | `scriptbots-runtime` | `HostCore` (`crates/scriptbots-runtime/src/lib.rs`) | Transitional `SharedWorld` in `crates/scriptbots-app/src/lib.rs` and `ControlHandle` | Production ownership transfer open: `bd-pcfj`/`bd-88yj`; closed preparation is insufficient (corrected 2026-09-05) |
| **Command Receipt Accounting** (no discarded receipts) | `scriptbots-app` | `CommandEnvelope`, `CommandId` (`crates/scriptbots-runtime/src/lib.rs`) | `no_control_command_discards_its_receipt` (`crates/scriptbots-app/src/servers.rs`) | Closed in `bd-2z0.4.9`; workspace guard in `bd-d6gv` |
| **WASM Dependency Purity** (no native franken in WASM graph) | `scriptbots-web` | `crates/scriptbots-web/Cargo.toml` | `ci/check_wasm_graph.sh` | Enforced in CI; purity slice closed in `bd-2z0.12.3`, which has since REOPENED for the broader browser-frontend deliverable (guard itself stays enforced; corrected 2026-09-03) |
| **Brain Heredity Gate** (copy AND vary proven before founding) | `scriptbots-app` | `install_brains` (`crates/scriptbots-app/src/lib.rs`) | `tests/heredity_gate.rs` | Enforced; closed in `bd-2z0.13.2` |
| **Read-Only Offline Analytics** (StorageReader cannot mutate) | `scriptbots-storage` | `StorageReader` (`crates/scriptbots-storage/src/lib.rs`) | Type system: no `&mut self` or write methods | Enforced; closed in `bd-2z0.8.9` |
| **Storage Lease & Inode Protection** (swapped DB detection) | `scriptbots-storage` | `filesystem_has_stable_file_identity` (`crates/scriptbots-storage/src/lib.rs`) | `tests/lease_persistence.rs` | Enforced; closed in `bd-2z0.5.6` |
| **Agent Identity Invariance** (keyed on stable AgentUid, not slot) | `scriptbots-core` | `AgentUid` (`crates/scriptbots-core/src/lib.rs`) | `WorldDigestV1` UID ordering | Enforced; closed in `bd-1kxd` |
| **Six Independent Stochastic Streams** (domain RNG isolation) | `scriptbots-core` | `DomainStreams` (`crates/scriptbots-core/src/rng_domains.rs`) | `tests/rng_sequence_compat.rs` | Enforced; closed in `bd-1kxd` |
| **Core Checkpoint Reconstruction** (full state without code exec) | `scriptbots-core` | `WorldCheckpointV1` (`crates/scriptbots-core/src/checkpoint.rs`) | Checkpoint roundtrip tests | Enforced; core closed in `bd-3n7p`, production replay in `bd-2z0.5.13` |

---

## Working in this repository

- **One shared working tree, many agents.** Reserve files (MCP agent-mail
  `file_reservation_paths`, exclusive) before editing; another agent's uncommitted changes appear in
  your `git status`, so use `scripts/shared_tree_commit.py` to review and commit exact owned
  paths. Never commit someone else's WIP.
- **Track work in beads** via the `br` CLI (not `bd`). Mark a bead in-progress when you start, and
  close it with evidence when it lands. `bv`'s robot triage can read a stale merge artifact — verify
  its picks against `br` before acting.
- **Acceptance executes through pinned DSR profiles** with retained source-bound evidence.
  RCH runs are diagnostic. See `README.md`, `AGENTS.md`, and `ci/dsr_verify.yaml` for the
  correctness lanes; performance requires the checked-in golden's exact DSR machine class.
- **Never lose a feature to a refactor, and never make a test pass by weakening what it proves.** A
  green suite that no longer certifies the thing it names is worse than a red one.
