# Franken Ecosystem Integration — Program Guide (bd-2js6)

Last reconciled: **2026-07-15** (update this line whenever a program bead
closes — that is part of each bead's close checklist by convention). This
document is what a new contributor reads INSTEAD of re-running the six-repo
survey that produced the program. Style: terse, factual. Authority order when
in doubt: code/Cargo.lock > beads (`br show bd-2js6` and its notes) > this doc.

## 1. What is in the tree today

| Library | Door | Pin | Feature gate | Status |
|---|---|---|---|---|
| `fsqlite` (frankensqlite) 0.1.16 | direct (`scriptbots-storage`) | git rev `1eec0d2669d0a7938e155b62ce8ebcd72e5bed78` — guard: `ci/check_fsqlite_pin.sh` | `default-features=false, features=["native"]` (extensions JSON/FTS5/R-tree still compile in transitively) | **in production** — sole embedded DB; clean V6 multi-run/provenance schema |
| `asupersync` 0.3.6 | direct (`scriptbots-runtime`, `scriptbots-app`) and transitive via fsqlite | crates.io exact `=0.3.6` — guard: `ci/check_asupersync_universe.sh` | runtime: optional `native-asupersync`; app: direct | production native ingress/lifecycle plus bounded legacy-app command ingress |
| `franken-kernel` / `-evidence` / `-decision` 0.3.x | transitive | crates.io | n/a | in tree via fsqlite |
| `ft-*` (frankentorch) 0.1.0 | direct optional via `scriptbots-brain-ml` | git rev `e4c6bdd5ec629ae70b40da9314da345ade012ca7` | `brain-ft` (non-default) | dependency admitted; FtBrain implementation remains bd-2z0.3.12.3 |
| everything else (ftui, fnx, frankenpandas, fsci, fnp) | **not in tree** | planned pins in `docs/licenses.md` §2 | admission beads below | planned |

## 2. Verdicts (with the one-paragraph why)

**Adopt profoundly:**
- **asupersync** — bounded MPSC command ingress with two-phase
  reserve/commit, native structured cancellation and ordered shutdown,
  region-owned background services, LabRuntime deterministic chaos testing of the persistence
  protocol, `Cx::scoped_cpu` spike as a cancellable rayon alternative,
  BrowserRuntime evaluation for `scriptbots-web`. Decision lineage:
  bd-2z0.4.3 (closed spike, `=0.3.6`); beads bd-2z0.4.12/.13/.14,
  bd-2z0.8.9.15, bd-2z0.12.4. Immutable latest snapshots use the canonical
  `SnapshotHub`; Bevy consumes that hub directly under bd-2z0.7.2 rather than
  adding an Asupersync watch competitor.
- **frankentui** — Evolution Lab TUI (owned by the pre-existing bd-2z0.6 epic,
  not this program). Program contribution: the pin must contain upstream
  lifecycle fix `15cc6543` (postdates published 0.5.0) — recorded on
  bd-2z0.6.2/bd-2z0.8.8.
- **fsqlite depth** — FTS5+BM25 narrative search (bd-16g.2.6/.7 — the
  capability is ALREADY compiled into our binary; `fsqlite-ext-fts5` is in the
  lock today), `async-api` (`AsyncConnection` + `Cx`) for control-plane reads
  (bd-2z0.8.9.12), BEGIN CONCURRENT decision for archipelago persistence
  (bd-2z0.8.9.13, default = per-island DBs unless evidence says otherwise).
- **Analytics quartet** — new native-only `scriptbots-analytics` crate +
  `sb-analyze` CLI (bd-2z0.11.5): fsci-stats statistical rigor validating the
  narrative detectors offline (bd-2z0.11.6), fnx lineage/community/centrality
  reports over the ancestry store (bd-2z0.11.7), frankenpandas
  Parquet/Arrow/CSV export + groupby/rolling summaries (bd-2z0.11.8),
  pipeline e2e with intervention-planted ground truth (bd-2z0.11.9), pairwise
  interaction persistence feeding graph analytics (bd-2z0.5.9).
- **frankentorch** — the FtBrain family (bd-2z0.3.12 epic): batched cohort
  inference via the library-agnostic BatchBrain substrate, flat-genome weights
  via `parameters_to_vector`, optional hybrid lifetime gradient learning
  (Baldwin/Lamarck toggle). Retires NeuroflowBrain's f64 + serde_json
  round-trip anti-pattern (comparison memo feeds bd-2z0.8.10's disposition).

**Principled skips (do not re-propose without new evidence):**
- **frankenjax** — self-described reference implementation; ~2.3–3.2 µs
  dispatch per jitted call; no optimizers, no parameter-vector primitive.
  Wrong engine for thousands of tiny per-tick forwards.
- **franken_numpy (general)** — `UFuncArray` is flat `Vec<f64>` tuned for
  large arrays + the Python surface; frankenscipy dominates for our shapes.
  Exception: `fnp-random` (bit-exact NumPy PCG64DXSM) stays a candidate under
  bd-2z0.3.10.
- **frankensearch** — fsqlite FTS5 covers the lexical need with zero new deps;
  its high-level API lacks a filter parameter for tick ranges anyway. Revisit
  only for cross-run SEMANTIC search in the Lab UI.
- **Any franken library in the tick hot loop** — the hand-rolled `wide::f32x4`
  stages + toroidal `UniformGridIndex` beat fsci's KDTree (f64-only,
  Vec-per-point, no torus) for per-tick queries. Franken value is the shell
  (concurrency, storage, UI, analytics), never the inner loop.

## 3. Constraint matrix

| Library | crates.io? | Toolchain | wasm32 | License |
|---|---|---|---|---|
| asupersync | yes (0.3.6 in lock) | stable subset exists; nightly default | **yes** (BrowserRuntime, incl. deterministic profile) | MIT+Rider |
| fsqlite | git-pin only | MSRV 1.85 | experimental upstream | MIT+Rider |
| ftui family | yes (0.5.0) + git rev for lifecycle fix | stable-ish | yes (ftui-web) | MIT+Rider |
| fnx-classes/-algorithms | **yes 0.2.0 — git repo unusable** (absolute `/dp/frankentui` path dep) | stable-ish | **no** (rayon) | MIT+Rider |
| frankenpandas | yes (0.1.2) | stable-ish | **no** (rusqlite default feature) | MIT+Rider |
| frankenscipy | **git only** | nightly (`std::simd`) | untested | MIT+Rider |
| frankentorch | **git only**, no tags; pinned `e4c6bdd5…` | nightly | **no** (rayon) | MIT+Rider |
| franken_numpy | **git only** | nightly | **no** (getrandom backend) | MIT+Rider |

Workspace pins `nightly-2026-07-09`, so nightly-only deps are admissible.
License: one byte-identical MIT + OpenAI/Anthropic-rider LICENSE family-wide
(sha + full analysis + distribution obligations: `docs/licenses.md`;
release-packaging obligation: bd-2z0.13.6).

## 4. Boundary rules (one place, so they cannot drift)

1. **fsqlite is the only DB and the only query engine.** frankenpandas is an
   interchange/summary layer, never a query engine. (AGENTS.md storage
   contract; bd-2z0.8.9.)
2. **`detect.rs` stays hand-rolled, online, bit-stable in `scriptbots-core`.**
   fsci-stats VALIDATES it offline (bootstrap CI, permutation tests); it never
   replaces it and never enters a tick path.
3. **Franken analytics crates live only in `scriptbots-analytics`** (native,
   never a dependency of the app binaries; enforced by
   `ci/check_wasm_graph.sh` guard B).
4. **UI paint paths never issue SQL** — they read `Arc<AnalyticsSnapshot>`.
5. **One asupersync universe** — a single 0.3.x in the lock, shared by
   fsqlite/fastmcp/first-party (`ci/check_asupersync_universe.sh`).
6. **wasm graph is denylisted + snapshotted** (`ci/check_wasm_graph.sh`):
   no rayon/wide/tokio/rusqlite/franken numeric crates in `scriptbots-web`.
7. **Single Cargo.lock mutation lane** (bd-2z0.8): franken admissions are
   serialized, one at a time, each with a `docs/licenses.md` row (enforced by
   `ci/check_franken_licenses.sh`).
8. **Determinism first**: the world tick stays synchronous; brains and buses
   must prove same-seed bit-identity across thread counts before shipping.
9. **V6 is run-bound and fail-closed**: canonical nonzero 128-bit `RunId`
   scopes every scientific and operational row. Writers own one run; append is
   atomic and allowed only after prior runs are fully durable. Multi-run reads
   select a run explicitly, catalog discovery is bounded and structurally
   validated, recovery revalidates canonical V3/V3.1 manifest projections and
   digests under the writer lease, and V3-V5 files are refused without rewrite
   (`bd-2z0.5.1`).
10. **Every supported scientific write uses one outbox protocol**: both
    `StoragePipeline` and same-thread `Storage::persist` assign a stable BLAKE3
    batch identity before applying rows, advance explicit admitted/applied/durable
    watermarks, reuse exact retries, and reject changed payloads. Raw agent insert
    SQL is private; the isolated FrankenSQLite conformance target owns its own
    explicitly non-production SQL. Receipt and shutdown/join observations share
    the exact typed terminal database cause (`bd-2z0.8.9.4.4`).

## 5. Program bead index + status

Umbrella: **bd-2js6** (`br show bd-2js6` — notes hold the authoritative index
and review log). Status at last reconcile:

- **CLOSED**: bd-2z0.8.15 (license/rider audit → `docs/licenses.md` + CI
  guard), bd-2z0.8.17 (asupersync universe guard), bd-2z0.8.9.14 (fsqlite pin
  reconciliation → `ci/check_fsqlite_pin.sh`), bd-2z0.8.16 (wasm graph
  denylist + golden), bd-2z0.11.5 (analytics scaffold), bd-2z0.13.6 (rider
  release packaging), bd-2z0.3.12.1 (frankentorch dependency admission),
  bd-2z0.8.18 (performance gates; DSR-only acceptance), bd-2z0.4.12
  (legacy app Crossfire removed; bounded Asupersync command bus DSR-verified),
  bd-2z0.5.1 (V6 multi-run schema, pre-tick-zero canonical provenance,
  run-bound recovery/readers, and bounded catalog; DSR workspace-verified at
  `8861e55f`), bd-2z0.8.9.4.4 (all supported writes use the durable outbox;
  exact typed terminal causes survive receipt plus shutdown/join; DSR
  workspace-verified at `60e06b3`).
- **OPEN, ready**: bd-2z0.3.12.2 (BatchBrain — sequenced after the
  bd-16g.15.x sense-lane digest move), bd-2z0.8.9.12 (fsqlite async-api),
  bd-2z0.4.14 (`Cx::scoped_cpu` spike), and bd-2z0.12.4 (BrowserRuntime
  evaluation).
- **OPEN, gated**: bd-16g.2.6/.7 (FTS5, needs bd-16g.2.2), bd-2z0.8.9.13
  (MVCC decision), bd-2z0.4.13 + bd-2z0.8.9.15 (structured-shutdown chain),
  bd-2z0.11.6/.7/.8/.9 (analytics implementations), bd-2z0.3.12.3–.6
  (FtBrain chain), bd-2z0.5.9 (interaction persistence, rides bd-2z0.5.2).

The pinned local DSR verification profile is the only acceptance lane. It runs
`check_franken_licenses.sh`, `check_asupersync_universe.sh`,
`check_fsqlite_pin.sh`, and `check_wasm_graph.sh` together with their relevant
build/test surfaces. Each standalone guard has a `--self-test` mode; hosted
workflow results are not acceptance evidence.

The `bd-2z0.5.1` close proof is DSR run
`bd-2z0-5-1-v6-20260715-8` on `darwin/arm64`: formatting, touched-file UBS,
workspace all-target check, strict workspace Clippy, and the complete workspace
test suite passed at commit `8861e55f`.

The `bd-2z0.8.9.4.4` close proof is DSR run
`bd-2z0-8-9-4-4-20260715-2` on `darwin/arm64`: formatting, touched-file UBS,
three focused identity/watermark and terminal-root-cause tests, workspace
all-target check, strict workspace Clippy, and the complete workspace test suite
passed at commit `60e06b3`.
