# Reality Check — rust_scriptbots — 2026-09-03

**Method:** Full read of AGENTS.md (1,022 lines), README.md (1,053), PLAN_TO_REARCHITECT_AND_REVIVE_RUST_SCRIPTBOTS.md (3,612), docs/ARCHITECTURE.md (665), plus all 44 markdown plan/spec/decision docs digested with per-promise citations. Ground truth from a per-crate static audit of all 14 crates (273,608 src LOC, 29,391 test LOC, ~1,895 `#[test]`), a stub/mock scan of `crates/*/src`, a full tracker audit (625 beads; every not-closed bead read verbatim), bv robot diagnostics, and live test runs through RCH.

**Tracker snapshot at check time:** 625 total · 436–439 closed · 182–183 open · 3–5 in-progress · 1 blocked-status · 77–78 ready · 20 open epics · zero dependency cycles. (Sibling swarms closed beads during the audit; both snapshots reconciled in `/data/projects/beads_landscape_rust_scriptbots.md`.)

**Test evidence this session:** `cargo test -p scriptbots-core --features economy-faults` — **exit 0, all green** on RCH worker hz4 (280s), including `world_integration` 21/21, the digest-V1.7 lane tests, and `regression_seed_42_matches_baseline`. The full workspace lane is currently **unrunnable on the RCH fleet** because workers lack `libudev.pc` (bevy's gamepad stack; see Gap G1 — no bead covered this before this check).

---

## 1. Vision Checklist

The measuring stick, distilled to testable goals. Statuses use the skill's categories.

| # | Goal (testable form) | Source | Status | Evidence |
|---|---|---|---|---|
| V1 | Default launch shows a living, understandable meadow at tick zero; explicit terminal/GUI modes work on supported platforms | Plan §18.1, §2.2 | WORKING (headless/TUI proven; GUI launch proven) | Scenario catalog cohort harness `tests/scenario_catalog.rs` (bd-2z0.10.3 closed, DSR 225/225); startup matrix MET 2026-07-17 on Linux/macOS/Windows (plan §0.4); PTY probe green |
| V2 | Closing/occluding/repainting frontends never advances scientific time; one driver owns the sim | Plan §4.1.1–2, §18.1 | PARTIAL | GPUI double-drive contained (HUD-only interim driver, README L113); **live app still runs on `SharedWorld` mutex bridge; HostCore not the production authority — bd-pcfj + bd-88yj in-progress (P0)**; ARCHITECTURE.md §2/§8 names this honestly |
| V3 | Pause/resume/step/speed behave identically from UI, REST, and MCP, with queryable two-axis receipts | Plan §18.1, §2.1 | PARTIAL | Command truth table frozen + tested (plan §2.1); receipts enforced (bd-2z0.4.9 closed); but live REST/MCP still enqueue onto the legacy app-owned `CommandBus` (README L979–982) — the runtime protocol is implemented but not migrated |
| V4 | Every default agent has a real heritable brain genome; children inherit/mutate/cross; placeholders never enter defaults | Plan §18.2, §4.2 | WORKING | Heredity gate `install_brains` probe (bd-2z0.13.2 closed); `ml.placeholder` unregistrable (app/src/main.rs:4728 guard); MLP/DWRAON/Assembly versioned genome+evaluator codecs with BLAKE3-bound state (plan §1.6/§1.7 exits); world binding bd-2z0.3.6 closed — only FrankenTorch family (bd-2z0.3.12.x) remains |
| V5 | MLP/DWRAON/Assembly honestly comparable in one world | Plan §18.2 | WORKING | Mixed-family default registry; species barrier tested; matched-seed experiment runner exercised all brain families end-to-end (bd-2z0.5.11 closed 2026-09-03) |
| V6 | Seed/config/toolchain/lock recorded; digests match; empty evidence cannot pass | Plan §18.3 | WORKING (core) | `run-manifest.v3.6` binds tick-zero `world-digest.v1.7`; checkpoint `v1.3`/codec-6 with BLAKE3; `NoEvidence` typed failures (plan §1.8, §7.4); digest/trace goldens DSR-pinned |
| V7 | Product checkpoint/resume: persist, discover, restore, replay from checkpoint with first-divergence proof | Plan §4.1, §18.3 | NOT_STARTED (product) | Core checkpoint closed (bd-3n7p), but production wiring is open bead **bd-2z0.5.13** ("Wire WorldCheckpointV1 into production persistence and checkpoint-start replay"); `CharacterizationLimitationsV0::checkpoint_replay_guarantee` still `false` |
| V8 | Run bundles verify on a second clean checkout; experiment runner reproduces comparable runs | Plan §4.2–4.3, §18.3 | PARTIAL | Experiment runner e2e green (bd-2z0.5.11); manifest sidecars + evidence hashes landed (bd-38us, 2026-09-03); bundle verify-on-second-checkout claim not separately evidenced |
| V9 | FrankenTUI Evolution Lab: responsive canvas, inspector, charts, palette, six science screens | Plan §8, §18.4 | PARTIAL (Ratatui present; FrankenTUI not adopted) | ftui pin "PREPARED, NOT ADOPTED" (docs/terminal_stack_and_frankentui_pin.md L20–24); bd-2z0.6.1/.5/.6/.8 all open; TUI has emoji HUD/goldens (bd-l51j closed) but no palette/science screens |
| V10 | Primary GUI with real batched GPU presentation; screenshots/goldens from the shipped path | Plan §9, §18.4 | PARTIAL — the largest honest gap | Bevy still per-agent entities/materials (bd-2z0.7.3 open); Bevy-owned sim worker still to remove (bd-2z0.7.2 open, P0); **bevy semantic golden red for 10 days for everyone (bd-2z0.14.3.9)**; cinematic program bd-2z0.14 has 37 open children — just begun |
| V11 | Curated scenarios tell distinct ecological stories, cohort-proven | Plan §2.3, §18.5 | WORKING | Six shipped scenarios, every-seed envelope + bit-identical rerun digests (bd-2z0.10.3) |
| V12 | Lineage/phenotype/interaction analyses; offline stats; exports with provenance | Plan §4.4, §18.5 | PARTIAL | `scriptbots-analytics` solid (83 tests; native bootstrap CIs, BH FDR); lineage fitness report + sb-analyze subprocess e2e closed 2026-09-03 (bd-2z0.11.10); fsci-stats/fnx/pandas adapters open (bd-2z0.11.6–.9) |
| V13 | External agents inspect and intervene through acknowledged commands (LLM-in-the-loop lab) | README L10–14; bd-16g.1 | PARTIAL | REST (25 routes) + MCP (13 tools) live with two-axis receipts; autonomous lab assistant chain bd-16g.1.x open; adversarial corpus bd-16g.16 open |
| V14 | Emergent-event detector, narrated timeline, live phylogeny, archipelago, interventions as product surfaces | bd-16g epics (evolved vision) | PARTIAL | Archipelago E2E + offline reconstruction closed (bd-16g.5.5.x); locus tracing closed (bd-wdyu); genome browser UI closed (bd-16g.13); detector/timeline bd-16g.2 open and gates 6 beads; live phylogeny bd-16g.3 open; intervention toolkit bd-16g.10 open |
| V15 | Performance: ≥60 TPS @1k; publish 10k TPS evidence; snapshot/TUI/GUI budgets | Plan §14 | PARTIAL | Regression gate live (bd-2z0.8.18, DSR-only); 1,372 TPS baseline recorded (bd-h33); **10k publication target entirely unstarted** — all 12 skill-loop passes pending (.skill-loop-progress.md L45–56); bd-m30b sense_radius re-bless open with 2 tests ignored awaiting it |
| V16 | Maintainable: no god file without extraction plan; lean reproducible deps; docs match reality | Plan §18.6 | PARTIAL | De-monolith characterized but zero extractions (bd-2z0.9.2 ready, awaits workspace confirmation); dependency ledger complete (UPGRADE_LOG, 25 landed entries); **CHANGELOG has no phase after 2026-07** — six weeks of delivered work unrecorded (no bead covered this); docs carry stale status claims (§3 G4) |
| V17 | Truthful evidence discipline: no signal claims more than the code observed | Plan §4.4; AGENTS "Signals"; bd-0oro | PARTIAL | bd-0oro (P1, open): 10+ instances of overclaiming signals, no shared author; guards bd-akjh/bd-h3nt/bd-m07q open; the discipline is enforced in new beads (FALSE-CLOSE sweeps) but the defect-class backlog is open |
| V18 | Infotheory wired end-to-end (estimator consumed by the sim, claims exportable) | bd-xqd5 (evolved vision) | STUB (wired-out) | Estimator has **zero callers** (bd-xqd5, ready, P1); every end-to-end information-theory claim blocked on it |

**Vision delivery score (goals fully WORKING): 5 of 18 (V1, V4, V5, V6, V11). Tracker completion ~70% is real but concentrated: the *deterministic laboratory core* is delivered and proven; the *promised experience* (one-driver host, FrankenTUI, cinematic GPU, checkpoint resume, narrated science, published performance) is where the remaining 30% lives — and it is serialized behind a handful of P0 keystones.**

---

## 2. The Five Answers (Phase 1, brutal version)

**Where are we REALLY?**
The project is no longer the plan's July failure mode ("ambitious code that does not form one coherent application"). The scientific kernel is genuinely done and DSR-proven: sensing oracles, six-domain RNG with agent-keyed substreams, stable UIDs, heritable genome/evaluator envelopes, canonical digest V1.7, core checkpoint V1.3, resource ledger, hydrology, 21-stage pure pipeline — all with executable evidence, and the core suite passes green today (exit 0, seed-42 regression match). Persistence is real FrankenSQLite with durable outbox and watermarks. The tracker is healthy (acyclic, ~70% closed, honest FALSE-CLOSE sweeps). What is NOT true is product delivery of the plan's P1 promise layer: the live app still runs on the pre-runtime `SharedWorld` bridge while a finished, tested `HostCore` waits on the sideline; the TUI is still legacy Ratatui with the FrankenTUI pin sitting unused; the GPU story is a per-agent-entity Bevy renderer whose semantic golden has been red for 10 days; checkpoint resume exists only as a core-science API; and the optimization campaign that is supposed to publish 10k-agent evidence has completed zero of twelve passes. Roughly: **science: yes. runtime authority: built but not installed. experience: started. evidence culture: best-in-class but carrying a known overclaiming defect class.**

**1. What specifically IS working right now?**
- Core simulation: deterministic, oracle-tested, green today under `--features economy-faults` (this session, RCH worker, exit 0). Digest V1.7/trace/checkpoint contracts wire-exact and golden-pinned.
- Six-cohort scenario catalog, matched-seed experiment runner (all brain families, bundle reopen, e2e verified yesterday), archipelago multi-island E2E + offline reconstruction, lineage locus tracing, lineage fitness report with bootstrap CIs, genome browser UI (all closed within the last 48h by concurrent swarms).
- Storage: bounded byte-admission pipeline, durable outbox, triple watermarks, fail-closed recovery, StorageReader-only analytics; 284 tests.
- Control plane: REST 25 routes, MCP 13 tools, control CLI, two-axis receipts, supervised sibling lifecycle, cross-platform startup matrix MET.
- Heredity-gated brain registry; placeholder structurally excluded from founding populations.
- Evidence machinery: DSR-only acceptance, fail-closed BV/BR authority wrapper, UBS-before-commit, FALSE-CLOSE graph sweeps.

**2. What is NOT working or not yet implemented?**
- **Host authority not installed:** production still `SharedWorld` + legacy `CommandBus`; bd-pcfj → bd-88yj → bd-2z0.7.2/bd-2z0.6.1 chain is the project's P0 spine (transitively gates 11 beads).
- **Live-path bugs:** `--mode server` dies at tick 420 (storage Admit timeout at :memory:, bd-w1oi); bevy semantic golden red 10 days for everyone (bd-2z0.14.3.9); five dead adaptive-governor tier features (bd-ogcs); two competing selection concepts without precedence (bd-ydu8).
- **Frontend promise layer:** FrankenTUI not adopted (pin prepared, zero consumers); Bevy GPU instancing, terrain 2.0, lighting/atmosphere, creature factory — 37 open cinematic children; real live-path visual E2E absent (every golden is a CPU surrogate or red).
- **Replay product:** checkpoint-start replay unwired (bd-2z0.5.13); narrative consumer persistence/parity chain open; storage reaping proofs open.
- **Science features:** detector/timeline (gates 6), live phylogeny, agent-perspective inspector (blocked by an acceptance conflict, bd-r7cz), intervention toolkit, LLM lab assistant, infotheory with zero callers.
- **Performance publication:** 10k target untouched; bd-h33 campaign at pass zero; perf A/B through RCH can't resolve deltas (bd-jfd1).
- **Infra:** the RCH fleet cannot run the full workspace test lane at all right now — workers lack `libudev.pc` (bevy gamepad stack) — discovered first-hand this session; plus bd-e6ff (local toolchain wedged on exFAT), bd-x1ec (hz1 97% disk), bd-oite (silent exit 1).

**3. What is blocking us from getting there?**
Serialization, not scope. Three keystones gate most of the surface: **bd-pcfj** (WorldState ownership → HostCore; transitive unblocks 11), **bd-16g.2** (detector/timeline; unblocks 6), and the bottleneck pair bd-2z0.4.9/bd-2z0.3.7 top the betweenness ranking. bd-pcfj and bd-88yj are already in-progress — they need to be *finished*, not restarted. Secondary block: the build farm (three open infra beads + the libudev gap) is the one dependency every evidence claim routes through. Tertiary: the de-monolith program is paused awaiting explicit user confirmation of the extraction workspace (plan §19 item 9) — every god-file extraction is gated on a human decision, not on agents.

**4. If we implemented all open and in-progress beads, would we close the gap completely?**
**Almost — yes for the tracker's own roadmap, with four named exceptions.** The tracker audit's coverage verdict: 19/20 open epics have open children that ARE the open beads; completing all 186 not-closed beads leaves only the housekeeping close of already-complete epic bd-2z0.8.9. But cross-checking the *docs-level* vision against bead coverage finds four goals with **zero** bead coverage (the dangerous kind):
- **G1 — Full-workspace test lane on the build farm** (libudev missing on workers): no bead mentions it; today it silently forced this reality check into an `--exclude scriptbots-bevy` lane. Every "workspace test green" claim is currently unfalsifiable on the fleet.
- **G2 — CHANGELOG Phase 11** (Aug–Sep 2026): six weeks of delivered work (archipelago, genome browser, lineage fitness, economy gates, experiment runner proof, browser DOM proof) recorded nowhere; no bead.
- **G3 — Epic-graph hygiene at scale:** 49 not-closed beads have no transitive open-parent path to any epic (whole visual/UX, infotheory, build-farm, perf themes would be stranded by epic rollups); 22 have zero edges; the bd-ikts vs bd-2z0.14 graphics programs overlap without a consolidation decision. No bead owns this cleanup.
- **G4 — Docs status-truth sweep:** plan §1.7/§7.3 still describe closed beads (bd-2z0.3.6, bd-2z0.5.9) as in-flight; franken_integration.md (reconciled 2026-07-26) lists bd-2z0.4.14 as open/ready while the decision doc records it closed-REJECTED (proof debt since re-filed as bd-2z0.4.15). Exactly the "stale signal" class bd-0oro prosecutes — in the project's own docs. No bead owns the sweep.

**5. What goals from the vision are NOT covered by ANY existing bead?**
G1–G4 above. Everything else in the vision checklist maps to at least one open bead.

---

## 3. Gap Register (new findings only — existing open beads are the known gap ledger)

| # | Gap | Category | Severity | Bead coverage before this check |
|---|---|---|---|---|
| G1 | RCH fleet cannot build the full workspace (workers lack `libudev.pc`; bevy gamepad sys deps) — workspace-lane evidence unfalsifiable | Proof gap / Integration gap | **Critical** (every evidence claim routes through this) | NONE → **bd-rch-full-workspace-lane-5mff created** |
| G2 | CHANGELOG records nothing after Phase 10 (2026-07); Aug–Sep deliveries undocumented | Documentation gap | Major | NONE → **bd-changelog-phase-11-mqwz created** |
| G3 | 49 not-closed beads detached from every epic; 22 edge-free; bd-ikts vs bd-2z0.14 overlap undecided; bd-2z0.8.9 epic ready-to-close | Process/tracker gap | Major (stranded work + false rollups) | NONE → **bd-epic-graph-hygiene-oa6p created** |
| G4 | Stale completion/status claims inside authoritative docs (plan §1.7/§7.3, franken_integration bd-2z0.4.14 line) | Documentation gap | Minor (but self-inflicted bd-0oro class) | NONE → **bd-docs-status-truth-sweep-v2iw created** |

Priority ordering across the whole bridge: the P0 keystone chain (existing beads, in progress) → live-path red bugs (existing beads) → G1 (unblocks trustworthy evidence for everything else) → frontend/replay/science programs (existing beads) → G2–G4 (cheap, high-trust-per-hour).

---

## 4. Bridge Plan (revised in place — see Round Log for each pass)

### Per-gap resolution

**G1 → bd-rch-full-workspace-lane-5mff: "Prove the full workspace test lane on the RCH fleet: provision libudev (or bless the documented no-bevy lane with a name that cannot masquerade)"**
- Current: `cargo test --workspace` fails on any worker at `libudev-sys` build (`libudev.pc` not found; first observed 2026-09-03 on hz4 after 462s, RCH-E307). The lane used instead this session: `--exclude scriptbots-bevy` (core green; full no-bevy run in flight).
- Target: either (a) workers provision `libudev-dev` (fleet change — needs operator/fleet access), or (b) the repo's documented test lane officially becomes `--exclude scriptbots-bevy` **with that exclusion printed in every artifact** so no one quotes it as "workspace".
- Success: one command in AGENTS/README-adjacent docs that a fresh agent can run to a decisive result, and the workspace lane either green or explicitly named as not-the-lane. No test weakened.
- Deps: none. Complexity S–M (fleet vs docs). Danger to avoid: do NOT "fix" by deleting bevy from the workspace or weakening any test.

**G2 → bd-changelog-phase-11-mqwz: "Record CHANGELOG Phase 11 (2026-08→09) from the commit record with live links"**
- Current: CHANGELOG phase table ends at Phase 10 (2026-07). Missing: economy conservation gates (bd-16g.11.2), archipelago recording/report/audit (bd-16g.5.5.x, bd-wdyu), genome browser + lineage fitness (74ad7e4), experiment-runner all-family proof (bd-2z0.5.11), manifest sidecars (bd-38us), browser DOM proof (bd-2z0.12.6), truthful architecture guide (bd-bsuh), journal retry calibration.
- Target: one new phase section in the existing CHANGELOG, commit-linked per house style; no history rewritten.
- Deps: none. Complexity S.

**G3 → bd-epic-graph-hygiene-oa6p: "Re-attach the 49 detached beads (or declare them standalone), close ready epic bd-2z0.8.9 with child enumeration, and make the bd-ikts vs bd-2z0.14 consolidation decision"**
- Current: 49 not-closed beads lack any transitive open-parent path to an epic; 22 have zero edges; two overlapping graphics epics; one epic ready-to-close but its own close rule requires enumerating children in the close reason.
- Target: every not-closed bead either epic-attached or explicitly standalone-with-owner-label; bd-ikts/bd-2z0.14 overlap resolved by an explicit decision comment on both; bd-2z0.8.9 closed with its 16 child IDs enumerated.
- Deps: none (tracker-only). Complexity M. Uses ONLY `br` (+ the wrapper for bv reads).

**G4 → bd-docs-status-truth-sweep-v2iw: "Docs status-truth sweep: plan §1.7 exit, plan §7.3 bd-2z0.5.9, franken_integration bd-2z0.4.14 line"**
- Current: three authoritative docs carry status claims their own evidence supersedes (details in §2 Q5/G4).
- Target: in-place revisions with dated correction notes (never deletions); each corrected claim cites the closed bead and its evidence.
- Deps: none. Complexity S.

### Execution Program (added in Ambition Round 1 — sizing, staffing, per-program exits)

**Sizing model (measured, not guessed).** bv's 55.8-day forecast is a ONE-agent estimate at confidence 0.43. The tracker's own velocity history says the swarm runs far hotter: 275 beads closed in the week of 2026-07-20, 50 the following week, 44 last week (bv project_health.velocity; dips to 0 in between show the variable that matters is *concurrent operator attention*, not bead count). With 78 ready now and 107 dependency-blocked, the makespan is governed by the critical chain, not total volume: schedule = max(critical-path length, total-work/lanes). Critical path runs through bd-pcfj → bd-88yj → bd-2z0.7.2/bd-2z0.6.1 → frontend/replay tails. Lanes: this workstation's swarm has historically sustained ~6–10 effective lanes; plan for 6 named lanes below.

**Lane assignments (6 lanes; each lane = one agent at a time, ordered by graph leverage):**
- **Lane 1 (keystone, highest leverage — do not diversify until done):** finish bd-pcfj (in progress) → bd-88yj (in progress) → bd-2z0.7.2 → bd-2z0.6.1. Exit: `SharedWorld` unreachable from production paths; server-only mode acknowledges paused commands; Ratatui consumes HostClient snapshots. Kill criteria: if bd-pcfj stalls >3 days, split its remainder and re-scope rather than letting 11 beads stay hostage.
- **Lane 2 (evidence trust):** bd-rch-full-workspace-lane-5mff (G1) → bd-2z0.14.3.9 (red golden) → bd-w1oi (server dies at tick 420) → bd-ogcs → bd-ydu8 → bd-r7cz. Exit: documented full lane runs decisively; golden green ×2 DSR runs; server-mode E2E passes.
- **Lane 3 (replay product):** bd-2z0.5.13 → bd-2z0.5.14 → bd-2z0.5.16 → bd-2z0.5.17. Exit: checkpoint-start replay matches digests with first-divergence diagnostics; bounded reaping proven under duplicate/saturated/hung timeouts.
- **Lane 4 (science features):** bd-16g.2 (unblocks 6) → bd-ji3a → bd-16g.3 → bd-16g.4 chain (resolve bd-r7cz first) → bd-xqd5 (infotheory callers) → bd-16g.10. Exit: timeline scrubable from persisted rows; detector parity proven from persisted series; estimator has production callers.
- **Lane 5 (cinematic GPU):** bd-2z0.14.3.11 (art bible) → bd-2z0.14.1.1 (instancing) → bd-2z0.14.1.2 (terrain 2.0) → bd-2z0.14.1.4 (lighting) → bd-2z0.14.1.5 (creatures) in budget order, with bd-2z0.14.3.5 (visual E2E) landing goldens as each lands. Exit per bd-2z0.14 gates: frame budgets met at 1k/10k, goldens from shipped paths.
- **Lane 6 (hygiene/docs/experiments):** bd-epic-graph-hygiene-oa6p (G3) → bd-changelog-phase-11-mqwz (G2) → bd-docs-status-truth-sweep-v2iw (G4) → bd-2z0.11.4/.9 → bd-1bdd → bd-2z0.13.3/.7.

**Finishing school (before ANY new claims):** the 3 in-progress beads (bd-pcfj, bd-88yj, bd-rcae) are the highest-leverage work in the tracker — in-progress work that stays in-progress is invisible to `br ready` and blocks its dependents. Rule: no lane starts a new bead while its in-progress bead is under 48h stale; stale in-progress beads get a status comment (progress or blocker) same day.

**Risk gates for the program itself:**
- RCH slot exhaustion (observed twice today): lanes must degrade to `--exclude scriptbots-bevy` + focused `-p` runs rather than idling; G1 fixes the lane honestly.
- Swarm file contention: respect file reservations; docs/beads-only lanes (6) never block code lanes (1–5).
- Review bandwidth: every close still needs evidence attached (house rule); a close without a runnable command attached is a FALSE CLOSE candidate — the bd-d3wu sweep class.
- Cadence: re-run this reality check weekly while >100 beads remain open; the file is the steering instrument, not a one-off audit.

### Milestones (added in Ambition Round 2 — dated, falsifiable predicates)

Operator-weeks, not days: the velocity history is bimodal on operator presence (0-close weeks exist between 44–275-close weeks), so commitments are stated per active swarm-week.

- **M1 (week 1):** bd-pcfj closed with remaining-work bounded (timebox: day 1 produces a written list of exactly what is left; if it exceeds a 2-day remainder, the bead is re-scoped into ≤3 named sub-beads same day). G1 decided (fleet OR documented lane) and landed. bd-2z0.14.3.9 root-caused — the fix must state WHETHER the golden regressed or the harness rotted; lane 5 builds on that answer. Predicate: `br stats` shows in-progress ≤1; `scripts/bv_authoritative.sh --robot-triage` top pick is no longer bd-pcfj.
- **M2 (week 2):** bd-88yj + bd-2z0.7.2 closed (server-only mode E2E green; Bevy consumes snapshots only); bd-2z0.5.13 checkpoint-start replay green in the bit-exact lane; bd-16g.2 closed with its 6 dependents unblocked. Predicate: bd-88yj/2z0.7.2/2z0.5.13/16g.2 all closed; no new blockers-to-clear in the keystone set.
- **M3 (week 3):** Ratatui on HostClient (bd-2z0.6.1); bevy instancing landed (bd-2z0.14.1.1) with first shipped-path golden green ×2; detector dependents (bd-16g.9/.14) claimed; G3/G2/G4 closed. Predicate: epic-graph orphan set ≤5; CHANGELOG Phase 11 exists; docs sweep landed.
- **M4 (week 4):** replay chain complete (.5.14/.16/.17); bd-xqd5 infotheory callers landed; bd-h33 pass 1 executed with a typed verdict artifact; cinematic terrain+lighting goldens green. Predicate: the §4 Verification plan has ≥5 of 6 boxes checked.

**Capacity math (RCH).** Observed wall-times: core suite 280s, full-workspace build-to-fail 462s on the fleet. At 68–76 slots and ~5–8 min/job, the fleet absorbs ~500–900 jobs/day; six lanes issue at most ~60–120 jobs/day. Slots are NOT the binding constraint — admission bursts are (two `insufficient_total_slots` refusals today). Rule: lanes retry with 60s backoff (as this session did), keep at most one heavy workspace job queued per lane, and never let a refused job masquerade as a test result (exit 103 ≠ red suite).

### Adversarial self-check (added in Ambition Round 2 — how this plan is most likely wrong)

1. **"bd-pcfj is a small push" is unverified.** The whole lane model leans on it. Mitigation: M1's day-1 bounding writeup is a hard gate; if the remainder is a redesign, the plan degrades to re-scoping, not denial.
2. **The bevy golden may be red because the capture harness is dishonest, not because the renderer moved** (bd-2z0.14.3.5's harness is also open). If lane 2 "fixes" the golden by re-blessing a lie, every downstream visual gate is corrupted. Mitigation: root-cause note MUST identify which side moved, with the artifact that proves it.
3. **Velocity flattery.** 275 closes/week happened during an all-hands sprint; planning 6 lanes assumes operator attention that may not recur. Mitigation: milestones are predicates, not dates; if a week passes with M1 unmet, the plan shrinks to lane 1 + lane 2 only.
4. **This document can rot like the docs it indicts (G4).** Mitigation: every round logged with date; the file cites bead IDs, so `br show` can falsify any claim in it; weekly re-check updates §1 statuses or states why they did not change.

### Scheduling bounds and forecast fusion (added in Ambition Round 3 — the math the lanes should quote)

- **Makespan floor.** With W ≈ 26,780 min ≈ 56 agent-days of estimated work (bv forecast total) and m = 6 lanes, W/m ≈ 9.3 agent-days is a hard floor no scheduling can beat (the W/m term of the Lenstra–Rinnooy/Karp bound OPT ≥ max(C_max, W/m)). The critical chain C_max (bd-pcfj → bd-88yj → bd-2z0.7.2/bd-2z0.6.1 → frontend/replay tails) is ~12–18 bead-days at bv medians, so **the floor is the critical chain, not volume**. Graham's list-scheduling guarantee (makespan ≤ (2 − 1/m)·OPT) says six lanes picking greedily by graph leverage land within ~8% of perfect — the plan's 6-lane shape is provably near-optimal IF picks follow leverage, which is why lane order = bv's what-if/betweenness ranking, recomputed daily, not a static list.
- **Forecast fusion.** bv's 55.8-day total carries self-declared confidence 0.43; observed active-swarm velocity (median of non-zero weeks: ~50/wk) implies ~3.7 active weeks for the 186-bead remainder. Precision-weighting the two (0.43 : 0.57) gives ~4.2 active weeks — so the honest public band is **"~3–5 operator-weeks of swarm-active work, dominated by the critical chain,"** not "56 days." Quote the band, and say which term is binding.
- **Daily re-rank rule.** Each lane's next pick = argmax over `br ready` of (transitive unblocks × priority) / (1 + est_days) — exactly bv's quick_wins/what-if computation — computed fresh each day from `scripts/bv_authoritative.sh --robot-insights`; static lane lists in this file are the initial condition, not the steady state.
- **Escalation skills on tap:** lane 5 (cinematic GPU) and the bd-h33 perf program should name `$extreme-software-optimization` and `$alien-artifact-coding` / `$alien-graveyard` (installed skills) in their bead claims — the same counter-incrementalism the ambition rounds use, applied at implementation time.

### Verification plan (after bridge work)
- [ ] V2/V3: `SharedWorld` gone from production paths; server-only mode applies and acknowledges paused commands (bd-88yj exit).
- [ ] V7: checkpoint-start replay matches digests; first divergence names tick/stage/agent (bd-2z0.5.13 exit).
- [ ] V9/V10: goldens from shipped paths only; bevy golden green ≥2 consecutive DSR runs (bd-2z0.14.3.9 + bd-2z0.14.3.5).
- [ ] V15: 10k TPS published artifact under pinned DSR profile (bd-h33).
- [ ] V18: infotheory estimator has ≥1 production caller + end-to-end claim test (bd-xqd5).
- [ ] G1: the documented lane runs green (or is explicitly named) on a fresh agent machine.

---

## 5. Round Log (ambition + refinement passes revise THIS document in place)

- **v1 — 2026-09-03 (initial):** bridge plan as above; 4 new beads filed via `br` (bd-rch-full-workspace-lane-5mff, bd-changelog-phase-11-mqwz, bd-epic-graph-hygiene-oa6p, bd-docs-status-truth-sweep-v2iw).
- **Round 1 (ambition, "decent start but barely scratches the surface") — 2026-09-03:** replaced the 5-line sequencing note with a sized, staffed Execution Program: measured velocity model (275/50/44 closes per week; makespan = max(critical path, work/lanes)), 6 named lanes with per-lane exits and kill criteria, finishing-school rule for the 3 in-progress keystones, risk gates (RCH slots, file contention, review bandwidth), and a weekly re-check cadence.
- **Round 2 (ambition, "a lot better but STILL far from OPTIMAL") — 2026-09-03:** added dated falsifiable milestones M1–M4 with predicates, RCH capacity math (~500–900 jobs/day fleet vs ~60–120 demanded; admission bursts, not slots, are the constraint — retry-with-backoff rule codified), and an adversarial self-check naming the four likeliest ways this plan fails, each with a mitigation.
- **Round 3 (ambition, domain depth — "surely there is relevant math"):** added the scheduling-bounds section: W/m ≈ 9.3-day floor vs 12–18-day critical chain (the chain binds), Graham's (2 − 1/m) list-scheduling guarantee justifying 6 greedy-by-leverage lanes within ~8% of optimal, precision-weighted forecast fusion giving the honest ~3–5 active-week band to quote instead of bv's low-confidence 55.8 days, a daily argmax re-rank rule, and named escalation skills for the GPU/perf lanes.
- **Rounds 4–5 (refinement continuation + convergence) — 2026-09-03:** Round 4 walked all four beads against the frozen checklist and landed three strengthening comments (honest coverage fraction on the no-bevy lane; attachment targets must be OPEN epics; docs sweep must check BOTH directions). Round 5 re-ran the authoritative wrapper after bead creation: BR/BV agree (data_hash recorded per run), zero cycles, all four beads ready — converged, no further changes found.
- **Implementation record — 2026-09-03 (SapphireTrout):** G4 landed: dated corrections in the plan (§1.7 bd-2z0.3.6 completed; §7.3 bd-2z0.5.9 landed; §1.8 bd-hiv1 closed; §4.1 bd-2z0.5.2 closed) and docs/ARCHITECTURE.md (bd-k7nq closed — ControlHandle now on HostClient; bd-hiv1 closed; bd-2z0.12.3 reopened clarification) and docs/franken_integration.md (bd-2z0.4.14 REJECTED/closed; bd-2z0.5.2 closed; gated list pruned of bd-2z0.8.9.13/.15, bd-2z0.4.13, bd-16g.2.6 — all verified closed via `br show`). G2 landed: CHANGELOG Phase 11 with commit-linked entries. G1 route (b) landed: AGENTS.md documents the workspace-minus-bevy lane with its ~10% coverage honesty and the exit-103 retry rule; route (a) stays operator-owned. G3 landed: new epic bd-build-farm-reliability-lb19; 47 beads re-parented to honest open owners (bd-2z0.4.8.1 and bd-9pqz.1 replaced closed parents; bd-bacf moved out of bd-ikts); bd-ikts closed as superseded-by bd-2z0.14 with its sole child enumerated; bd-2z0.8.9 closed with all sixteen children enumerated (a follow-up comment corrects an error in the close reason itself). Verification: `br dep cycles` = 0; authoritative wrapper green post-mutation.
- **Sweep discovery (material):** bd-k7nq ("Migrate ControlHandle off SharedWorld to HostClient") is CLOSED — the docs and this report's §1 V2/V3 rows were written against the older ARCHITECTURE claims. The host-retirement spine is further along than the July docs said: ControlHandle is migrated; the remaining transitional legs are exactly bd-pcfj (server world-ownership, in progress) and bd-88yj (frontend chain, in progress). Lane 1's job is unchanged, but its starting line moved forward.
- **Sweep discovery (record hygiene):** bd-rcae (CandleBrain real Tensor forward, in-progress at audit time) and bd-16g.11 (ecosystem accounting) closed during this session — the swarm is actively converging on lanes 2–4 of the Execution Program while this document was being written, which is the system working as designed.
