# Reality Check — rust_scriptbots — execution refreshed 2026-09-05

## Execution TODO — started 2026-09-05

Continuation block — **bd-16g.5.5.2, closed after pinned DSR21, TurquoiseLake**:

- [x] Resolve the seven failures from completed DSR16 (`4a1c8e5`) before full-suite acceptance.
  - [x] Retain the completed failed run: storage unit tests 208/208; multi-run 17/19;
    journal 3/8; other default integration targets passed, including reaper 4/4.
    Three existing timing tests remained ignored. No full-suite pass is claimed.
  - [x] Implement tick-zero narrative admission and the typed legacy-schema refusal check
    in `ec0fe9d`; their runtime qualification remains pending.
  - [x] Trace `UpdateConfig` corruption to conditional config field omission in positional
    Postcard. Candidate `c0f2d6c` uses named MessagePack fields inside the existing command
    envelope, streams sizing before the allocation permit, and leaves scientific digest
    serialization unchanged. Add round-trip, changed-payload, truncation and float-bit tests.
  - [x] Correct the combat fixture's invalid zero alignment cosine to 0.5; the victim remains
    directly ahead. Preserve all required observed lifecycle/combat assertions.
  - [ ] Diagnose two journal durability timeouts. Candidate adds last status/snapshot evidence;
    it does not raise polling budgets or claim those failures fixed.
  - [x] Finish pinned DSR17: codec 2/2 and multi-run 19/19 passed; journal 3/8 passed.
    Configuration decoding reached real application, then tick two rejected revision 0→1.
  - [x] Finish pinned DSR18 at `8ce006b`: runtime drain 2/2, codec 2/2, allocation bound 1/1,
    missing-claim corruption 1/1 and multi-run 19/19 passed. Journal improved to 5/8, including
    the real concurrent file channel. Two commands remained pending within unchanged polling
    limits; the other failure was a copied corruption fixture missing its writer-lock companion.
    Both runs failed overall; copied logs and verdicts have matching remote/local SHA-256 hashes.
  - [x] Complete **bd-0wek**, witnessed live narrative configuration boundaries. Authenticate
    monotonic revisions against applied commands at the preceding tick, require exact policy,
    and reject unwitnessed jumps/regressions. Candidate extends GUI-origin command replay with changed
    policy, a planted policy mismatch, and explicit fixed-config offline-reader refusal.
  - [x] Qualify the copied-fixture repair and trace journal admission, state transitions,
    command projection and science persistence with actual worker span timings in DSR19.
    Instrumentation is diagnostic evidence, not a timeout fix or a performance-gate result.
    DSR19 (`e85faf5`) failed overall: runtime 2/2, codec 2/2, allocation 1/1, missing-claim 1/1,
    narrative admission 1/1 and multi-run 19/19 passed; journal remained 5/8. The channel's
    applied transition took 24.5 seconds, while its command projection took only 48.3 ms.
    The original domain lifecycle/combat positives passed before another copied fixture's
    absent lock companion failed. The command ordinal-corruption repair passed.
  - [x] Complete **bd-ek7o**, preserving a crossover's occurrence tick one when its lifecycle
    payload is published at tick two. DSR19 got past narrative revision admission and exposed
    the old equality guard. Candidate preserves both timestamps and exact archive matching,
    rejects future occurrences, and requires the distinct pair in GUI-origin journal readback.
  - [x] Correct the new policy mutation test to exercise `Storage::recover_existing_run`:
    ordinary finished-reader opening does not validate the narrative policy. `4ef8a2e` requires
    successful recovery before the mutation and typed policy refusal afterward. No runtime
    credit from DSR19, which failed earlier; DSR20 executed and passed both controls.
  - [x] DSR20 at `a16f27b`: lifecycle unit 1/1, narrative admission 1/1 and journal 8/8 passed
    (73.13 seconds for the integration target). This includes full two-run GUI command/digest/
    receipt equality, crossover occurrence/publication ticks (1,2), valid-copy recovery followed
    by policy mutation refusal, offline mixed-config refusal, and domain corruption controls.
    Workspace all-target check passed in 5m13s. Strict Clippy failed with 421 core diagnostics;
    the overall diagnostic run exited 6. Both targeted fix beads closed with this precise limit;
    the full storage suite is not covered by DSR20.
  - [ ] Track the intermittent journal latency separately as **bd-j8o2**. DSR20's channel passed
    without a causal latency fix; the observed DSR19 24.5-second applied transition remains
    unresolved. Do not conflate it with the original 120-second memory-admission server failure.
  - [x] Execute the full suite on corrected source. DSR21 at `a16f27b` passed all 11 steps:
    five guards, workspace all-target check, both CLI tests, capture retry, multi-run 19/19,
    full storage tests and analytics build. Storage units 211/211 and integration tests 83/83
    passed, with three existing timing tests ignored. The harness summary says 300 because
    six child-process test executions are counted again; distinct parent tests total 294.
    Journal 8/8 passed in 71.44 seconds. Retained verdict SHA-256:
    `34d485a5ca4bec2276c3bc4846e9be976fdc9d070d5fbb9540b6dec860649968`.
  - [ ] Execute both original 600-second server scenarios. DSR24 at `a16f27b` stopped at
    an incorrect MCP negotiation test before reaching either long run. **bd-s5ou** corrects
    the test against the pinned SDK and normative 2024-11-05 protocol: unsupported initialize
    versions receive a supported alternative; malformed parameter types still require -32602.
    Pinned DSR27 `server-d4cab5d-20260905-27` now reruns the lane on hz4.
- [ ] Complete **bd-vlp7**, strict core lint qualification without changing scientific results.
  - [x] Retain DSR20's exact diagnostics; confirm its core source is unchanged from `9b8490b`.
  - [x] Assign disjoint file groups under AGENTS.md's parallel-small-change instruction;
    root owns `lib.rs`, with three agents covering the other 15 affected files.
  - [x] Implement root documentation, const, equivalent control flow and existing validated
    dimension conversions; remove the unused UID helper after proving no consumers.
  - [x] Separate archive eligibility diagnostics from candidate insertion without changing order.
  - [x] Review every delegated hunk, especially numeric expectations and extracted calculations.
  - [x] Implement **bd-mjkd** audio budget/sample-count wrapping, **bd-fb32** phylogeny count
    truncation, and **bd-sknl** MI sizing before bin validation, with discriminating boundary tests.
    DSR22 executed audio 7/7, phylogeny 1/1 and MI bounds 1/1 successfully. Closure review remains.
  - [x] Verify formatting and inspect UBS findings for all changed files. Raw scanner findings
    are retained; existing assertion/const guards and simulation-state crypto misclassification
    are not a clean compiler result.
  - [x] Commit reviewed source as `3a08217`; compare the frozen diff byte-for-byte with the
    complete source diff reviewed before committing.
  - [ ] Run pinned strict core/workspace Clippy and compiler checks. DSR22 at `3a08217` finished
    failed: both compiler checks and formatting passed; strict Clippy stopped on one remaining
    core-library length diagnostic. DSR25 got past that fix and rejected an obsolete lint
    expectation; its removal is being checked in DSR26. Every physical worker keeps one DSR job.
  - [ ] Run scientific regressions, existing unchanged goldens and the economy-fault feature lane.
    DSR22 default units: 687 passed / 6 failed / 5 existing ignored; economy-fault units:
    690 / 6 / 5. Every core integration target passed. The same six failures affect checkpoint
    decoding (four tests), complete config decoding (one), and archive knob classification (one).
  - [ ] Complete **bd-6xr2**, full configuration serialization independent of the scientific
    digest projection. Both structs now derive from one field declaration; all recording fields
    serialize positionally without holes. Preserve science goldens; explicitly advance checkpoint
    codec 7→8 because payload layout changes. Keep the old wire golden until actual byte
    capture and review; add default/non-default and float-bit tests. That review is recorded below.
    Candidate `02ec805` failed DSR23 compilation because root missed one validator reference
    to the removed stride sentinel constant. Corrected locally to the explicit `u32::MAX`
    bound. Remove the old archive capacity sentinel exemption and test both enabled/disabled
    configurations. No execution credit for the failed candidate; qualification continues.
    DSR25 at `6c89bef` passed 697 default / 700 economy-fault unit cases; the sole failure
    was the intentionally old checkpoint wire golden. Both builds produced the same canonical
    8,698-byte wire. Its 149-byte growth equals the seven archive fields byte-for-byte. Reviewed
    codec-8 golden and obsolete-lint-expectation removal landed in `d4cab5d`; DSR26 now runs
    complete core suites, storage config codec/journal tests, compiler checks and strict Clippy.
  - [ ] Complete **bd-3cf0**, classify all eight archive paths from actual consumers and run
    witnesses for differing operational telemetry with equal full science digests, plus typed
    invalid-cap refusal. Both witnesses and unchanged completeness gates passed DSR25; final
    broader source qualification continues in DSR26.
  - [ ] Complete **bd-lhrv**, preserve real collected elites across unchanged, cadence-only,
    chart-only and valid-cap changes. Remove the comparison between actual grid size and a
    configured ceiling. Keep rebuilds for changed quality meaning; real MLP witness passed
    DSR25 1/1. Final broader source qualification continues in DSR26.
  - [ ] Record actual outcomes and close only after required acceptance passes; keep wider
    build-farm, native frontend and exact-class performance qualification separate.
- [x] Trace checkpoint replay and archipelago production entry points against current code.
  Checkpoint restore still needs the host/session boundary. At initial inspection, the app's
  archipelago only ran inside the determinism self-check; that check stays free of database writes.
- [x] Add a separate bounded isolated-island CLI run using canonical config, production
  brain installation and founders, sole-owner hosts, and one complete barrier per tick.
- [x] Bound journal capture and preserve volatile versus durable receipt semantics.
- [x] Register run and island provenance before advancing science; atomically persist every
  island's real batch; stop before another tick if persistence fails; explicitly close storage.
- [x] Repair duplicate barrier admission so refusal retains the original payload.
- [x] Repair newly exposed genome/narrative collisions with island-scoped schema keys,
  outbox rows, search joins, explicit genome readers, and per-island recovery validation.
  Verify populated V17 migration, same local IDs with different genomes, and missing input
  on one island without weakening single-world assertions.
  Implemented beginning at `47c76eb`; the new migration/identity/recovery unit tests passed
  in the `c012dfc` DSR run. One bounded delegated change updated 27 single-world genome-read
  callsites; every diff was reviewed. Full storage regression acceptance remains pending.
  - [x] Refuse ambiguous pre-V18 multi-island genome/narrative migrations before any DDL;
    exercise both row kinds against metadata and tick evidence and verify unchanged data.
  - [x] Preserve interaction edges derived from science replay, including equal local event
    IDs on different islands; require existing graph readers to refuse unsupported cross-island
    UID conflation. Candidate comment claiming this mode could not write interactions was
    wrong and has been corrected in `4db83b9` after tracing the actual writer.
  - [x] Refuse ambiguous retained pre-V18 multi-island outbox payloads before migration;
    verify when the pending payload is the only island evidence and preserve it unchanged.
  - [x] Preserve lineage edges derived inside birth insertion; test equal child IDs with
    different parent ordinals across islands. My previous automatic-projection review was
    incomplete. Reject general table exports that cannot represent the island dimension.
  - [x] Wire a same-thread `flush_with_watermarks` barrier through the existing application
    and finalization stages; retain the assertion requiring admitted/applied/durable equality.
- [x] Execute the actual CLI, reopen its database, and verify every island at every tick,
  nonempty agents/replay rows, distinct island digests and durable batch watermarks.
  Observed at `965ea99`: four tables × four ticks, three islands in all 16 cells;
  16 genomes per island; distinct final energies match their own stored island summaries.
- [x] Complete report JSON, explicit recovery and existing-file refusal checks. The report
  command passed conservation, but the test artifact helper overwrote its JSON with command
  metadata. Candidate `c012dfc` separates those filenames; no failed artifact is credited.
  Actual `c012dfc` rerun passed both CLI tests and all ten subprocess checks, including intact
  JSON with three islands/nine conservation transitions/zero breaches. The capture retry
  test also passed. Storage-suite and strict-lint qualification are still pending.
- [x] Exercise existing-file, invalid count/cadence, changed duplicate and incomplete-barrier
  controls. Preserve original assertions and golden files.
- [ ] Run formatting, required compiler/lint checks and focused correctness through pinned DSR;
  retain actual counts/source and perform a fresh self-review before any closure.
  First candidate `7f73d53` failed the real DSR compiler check (private journal constructor);
  no tests ran. The next candidate uses the public default and expands the lane from two
  barrier tests to the complete storage suite because it changes schema and read APIs.
  Second candidate `47c76eb` also failed compilation: a missing `RowExt` import in the
  CLI test caused nine diagnostics. Candidate `c7d6d0d` fixes it and adds the pre-V18
  ownership refusal. Third DSR `archipelago-c7d6d0d-20260905-03` failed compilation on
  nonexistent `reader.islands()`; corrected to `archipelago_islands()`. All three attempts
  ran zero tests. DSR `archipelago-4db83b9-20260905-04` then passed the five guards and
  workspace all-target check; both new CLI tests executed (one passed, one failed). The
  invalid-run controls passed. The real run applied barrier one but stopped because
  `Storage::flush` left the durable watermark unset. Candidate `5af2d1b` adds public
  finalization and lineage attribution. Its DSR run passed compilation and completed four
  real barriers with reopened admitted/applied/durable watermarks all equal to four, then
  failed the source assertion: DSR had not embedded the verified commit in the executable.
  Candidate `965ea99` embeds observed build provenance after validation and asserts that all
  three final energies differ, so the island-label comparison can detect permutations.
  DSR `archipelago-965ea99-20260905-06` reached all table checks, then failed parsing the
  overwritten report artifact. DSR `archipelago-c012dfc-20260905-07` also
  preserves the recovery fingerprint fixture's intended current-schema forgery and runs
  all storage targets with `--no-fail-fast`; any failing target still fails the lane.
  The `c012dfc` run passed CLI/capture and 204 of 206 storage unit tests, then 9 of 11
  admission-bound integration tests. It was intentionally stopped during ancestry integration
  after the following corrections were ready; remaining targets did not complete.
  - [x] Re-execute the forced-termination writer-lease test with recovery/shutdown stage
    diagnostics. It passed at `ec5c41c`: recovery plus shutdown took 1.81 seconds within
    the unchanged ten-second limit. The previous timeout's cause remains unresolved.
  - [x] Re-execute V16-to-V17 migration with a valid run-policy fixture. It passed at
    `ec5c41c`; the previous row supplied both a bounded tick budget and a live policy,
    violating the frozen V6 constraint.
  - [x] Re-execute narrative pressure with 512 distinct event identities and an explicit
    uniqueness observation. The old fixture repeated one identity; the corruption guard
    predates this block and remains intact. Both pressure cases passed at `ec5c41c`.
    My initial production-bug interpretation was wrong.
  - [x] Verify the original 5k/default-window test, which exceeded its unchanged 600-second
    flush limit. `a7f3ac6` wires the pinned engine's existing statement-savepoint opt-in only
    into prevalidated agent insertion; its sole caller rolls back the entire science batch.
    The existing nine-boundary rollback/recovery matrix passed at `ec5c41c`; the 5k case
    again exceeded 600 seconds. The insertion change is insufficient, with no speedup claim.
    Admission integration finished 10/11 passing in 711.21 seconds. Post-failure diagnostics
    on the separate ancestry worker captured replay insertion doing foreign-key validation
    through repeated schema metadata reload. Investigate bounded multi-row statements;
    preserve foreign keys, statement order, the exact engine pin, and all deadlines.
  - [ ] Complete a passing archipelago DSR lane, graph regression and strict Clippy.
    Attempt 08 ran zero tests and was cancelled after declaration review caught `Tick`
    lacking `Hash`; `ec5c41c` hashes the underlying integer. Interrupting metadata alone
    had not stopped the run because the guard fell back to the lockfile; the isolated DSR
    process group was then stopped and observed gone before updating source/profile.
    Current `ec5c41c` has passed all five guards, workspace all-target compilation, both
    CLI tests (38.53 seconds, ten subprocesses) and the capture retry test. The complete
    storage unit target passed all 206 tests (zero failures/ignored, 886.95 seconds).
    Admission integration failed only the 5k deadline. Attempt09 was intentionally stopped
    during the 600-tick ancestry test after CPU/stack diagnostics and a replacement candidate
    were ready: DSR process exit 6, typed failed, eight completed steps, 3,856 seconds.
    Its unfinished ancestry and later targets earn no pass credit.
  - [ ] Qualify `976bb1d`: bounded multi-row agent/replay/interaction inserts using the
    original statement savepoints, preserved input order and strict narrative-input policy.
    New real-engine coverage reads every row across chunks, forces a late CHECK failure
    through the actual whole-batch rollback, and retains the strict-duplicate guard.
    UBS ran before commit: 807 critical heuristic matches, 1,618 warnings, 626 info.
    Inspected critical categories were existing test panics, SQL/tick comparisons, typed SQL
    decodes, and test SQL fault controls; no clean scan or runtime qualification claim.
    Actual attempt10 passed all five guards, workspace compilation, both CLI tests (46.31s,
    ten subprocesses) and capture. The new test read back all 1,025 rows correctly and observed
    a rolled-back failure, then failed my incorrect expectation of `CheckViolation`. Pinned
    engine source instead emits an `Internal` VDBE CHECK halt. The assertion also omitted the
    actual error, so that cause remains source-derived until the corrected test executes.
    The correction requires the scope CHECK halt specifically and prints unexpected errors;
    it also requires the narrative uniqueness error. Zero-row rollback and prior-commit
    preservation assertions are unchanged and have not yet executed in this new test.
    `18abbec9` also matches the pinned AST's normalization of `<>` to `!=`, caught before
    a rerun. `9b8490b` applies the same helper to tick summaries, metrics and event counters;
    the three-island fixture writes 114 metric rows per barrier, formerly 114 statements.
    All six helper callers retain their original SQL field mapping and the enclosing
    `flush_attempt` rollback owner. This candidate still needs the actual DSR regression run.
    Attempt10 admission finished 10/11 passing in 711.61s; the 5k case again exceeded 600s.
    Its worker had exited before inspection, so no stack from that failure was obtained.
    The run was intentionally interrupted during ancestry after checking the exact Cargo/test
    working directories; both exited, DSR process exit 6/typed failed/eight completed steps.
    Unfinished ancestry and later integration targets receive no pass credit. The source and
    profiles advanced only after exit. Current `workspace-9b8490b-20260905-11` now executes
    compiler and strict Clippy checks before graph/storage qualification.
    That actual workspace run passed all five guards and compilation (3m54s), then failed
    strict Clippy on 421 core diagnostics. Core source is unchanged from pre-block `1f3158c`.
    DSR exit 6/typed failed/six completed steps; workspace/economy-faults tests did not run.
    Logs are retained at `/tmp/scriptbots-workspace-9b8490b-proof-20260905`. The independent
    graph/archive regression lane `graphs-9b8490b-20260905-12` is now executing.
  - [x] Re-execute graph/archive regressions at `9b8490b`. Actual DSR lane12 finished with
    process exit zero, typed pass, ten completed steps and matching source on every command:
    graph tests 10/10 (51.17s), archive unit 1/1, archive integration 1/1 (15.49s).
    This does not substitute for the unfinished full storage or workspace suites.
  - [x] Capture the actual 5k worker stack under a dedicated pinned DSR diagnostic, then
    choose a measured fix. `storage-diagnostic-9b8490b-20260905-13` first executes the
    corrected late-chunk rollback test and requires one actual pass, then the exact original
    5k test. Its verdict is always diagnostic; debugger-assisted timing cannot approve a gate.
    The original corrected rollback test passed 1/1 in 72.50s, including all 1,025 rows,
    late CHECK rollback and strict duplicate checks. Two snapshots of the exact five-k
    worker showed agent inserts entering multi-VALUES replay and fully hydrating the in-memory
    table contents per replayed row. The named test thread waited in `flush_and_wait`.
    Both debugger sessions detached and the worker resumed. The diagnostic test failed
    0/1 at its 600-second limit (601.10s total); typed diagnostic, test exit 101, DSR exit 6.
    Logs are retained at `/tmp/scriptbots-storage-diagnostic-9b8490b-proof-20260905`.
  - [x] Implement `4fbde8f`: use bounded `INSERT ... RETURNING 1`, compare returned row
    count to actual chunk input count, and keep original FK/conflict/transaction semantics.
    Pinned source selects direct VDBE execution for this SQL and still validates FKs after
    writing. Add a missing-parent rollback check and an actual two-input/one-returned-row
    SQL control for the new count guard. These new controls have not yet run.
  - [ ] Execute `storage-returning-4fbde8f-20260905-14`: both declared `tests::chunked_`
    cases, then the unchanged five-k workload without debugger attachment. Its scope remains
    focused storage checks; no full-suite or exact-class performance claim.
    Actual result: one passed, one failed in 2.56s. The complete 1,025-row/CHECK/FK/uniqueness
    rollback case passed on the direct path. My count-guard control assumed `OR IGNORE`
    would return one row, but the helper observed two and returned success; the test failed
    before its table readback. No claim about the actual ignored write is justified by that
    result. The replacement uses `INSERT ... SELECT ?1 WHERE 0` and reads the table before
    interpreting the guard error. The count guard is unchanged; the local variable now names
    returned rows rather than affected rows. Five-k did not execute, DSR exited 6, and the
    post-precheck diagnostic verdict was never reached.
  - [x] Execute corrected `4a1c8e5` under actual pinned DSR attempt15. Both chunk tests
    passed (2/2, 2.52s), including the independently observed zero-row control. The unchanged
    five-k workload passed (1/1, 24.09s; raw `flush_ms=22690`) without a debugger, within
    its unchanged 600-second deadline. Process exit zero, typed diagnostic, test/log exits
    zero and one executed five-k test. This is focused correctness/liveness evidence,
    not a full-suite or exact-class performance result.
  - [ ] Complete actual `archipelago-returning-4a1c8e5-20260905-16`, now running the
    all-target compiler check, actual CLI, capture retry and complete storage suite.
    Five guards and compiler passed (4m03s); CLI passed 2/2 (8.37s) and capture 1/1.
    Readback of retained `recorded-archipelago-WFF8o8` confirms all 16 three-island
    table/tick cells, 48 founder genomes, distinct per-island energies and watermarks 4/4/4.
    All 208 storage unit tests then passed (700.22s), followed by all 11 admission integration
    tests (127.67s), zero failures/ignored. Five-k passed again with `flush_ms=22656`.
    Child re-execution subtests are not added to the 208-test denominator. The 600-tick
    ancestry target then passed all 3 tests (658.29s), including the live-versus-rebuilt
    canonical graph after actual births and deaths. Async integration passed 3/3.
    Durability and later integration targets are still running; no aggregate lane pass yet.
    Durability passed 11/11 (243.92s), conformance 2/2, database genome browser 1/1
    (161.88s), genome persistence 8/8, historical golden 1/1, and locus tracing 1/1.
    Multi-run integration then failed 2 of 19 tests: obsolete legacy-refusal prose and a
    production tick-zero admission defect. The latter also exists at pre-block `1f3158c`.
    Candidate `ec0fe9d` allows only an empty narrative prefix at the initial zero boundary;
    the existing admission matrix now accepts tick zero and still refuses missing tick-one
    input without advancing watermarks. The legacy test checks file byte identity before
    the typed Startup/NotAdmitted/schema-version refusal. Full lane16 is still collecting
    later results with `--no-fail-fast` and cannot pass. A fresh full lane is required;
    it now runs the multi-run target early as well as in the complete storage suite.
  - [ ] Verify the original `bd-w1oi` server workload for its full 600-second reproduction
    window with live REST tick observations and retained logs. A five-k flush pass alone
    cannot establish long-running server progress. Replace the old bug-observed-is-green
    diagnostic with a positive regression that fails on simulation errors or stalled ticks.
    Implemented at `70cb3b5`, including both memory and file backends, retained command,
    stdout/stderr/status observations, the exact Cargo executable, and a server lane in the
    existing DSR runner. The progress bound derives from the production admission deadline;
    deliberate test termination is reported as kill/reap. Runtime execution remains pending.
  - [ ] Complete final-source application/full-storage and graph checks. Strict workspace
    Clippy remains blocked by the 421 core diagnostics observed at `9b8490b`.
- [ ] Update Beads and this checklist with observed results and remaining migration, host-session
  replay and outer-island parallelism limits. The broader archipelago remains incomplete.
- [ ] **Additional observed gap: bd-16g.13.3, reopened with parent bd-16g.13.** The affected
  render genome-browser test can skip failed persisted-genome reads and absent newborns,
  accepts unexpected diff statuses by printing, constructs a view model without rendering,
  and deletes CSV/PNG artifacts. Re-execute mandatory positive newborn/database comparisons
  and retain real UI screenshot/CSV/PNG evidence before reclosure. This source finding does
  not claim the UI implementation is absent, and this block has not executed that UI test.

This checklist is the operator-requested working memory for the eight added tasks.
Its consumers are the operator and implementing agent; Beads remains the claim and
dependency authority. It retires as an active checklist when those tasks are
implemented and verified. It earns no product-completion credit. No additional
process machinery is needed; these records do not substitute for working simulation
and analysis.

Delivery snapshot, 2026-09-05: five of the eight added beads are closed; three remain
open. Actual clean DSR `combined-c5ac1a9-20260905-06` completed all 14 named correctness
steps at `c5ac1a989444f48b0f5ff3d34f90a46cd6d2c85b`, with process exit zero, typed
`pass`, and successful source/profile/command-log readback. This is focused Linux
correctness evidence. Full workspace, performance and product-journey acceptance
remain outstanding. The historical audit below retains its original source boundary.

- [x] **bd-2z0.11.11 — interaction semantics, closed, TurquoiseLake.**
  Validate complete half-open windows versus newest pages; reject one-sided bounds;
  define zero limits; bound materialized input and graph work before loading;
  retain repeated-edge count and magnitude; reject invalid magnitudes; share selection
  between reports and exports; expose selection/capture/algorithm semantics; remove
  the unbounded, differently ordered replay fallback; update user-facing examples.
  - [x] Shared canonical storage selection, complete half-open windows, recent pages,
    zero-limit behavior and pre-load row/projected-byte/work admission implemented.
  - [x] Directed repeated edges retain count and magnitude in both export formats;
    invalid magnitudes and unsupported parameters are rejected.
  - [x] Capture uncertainty, exact selected event identities, algorithm weighting and
    seeded sampling are exposed. Candidate implementation committed as `428d102`.
  - [x] Re-execute the final fixture additions through clean pinned DSR; inspect failures
    and original acceptance criteria before closing the implementation bead.
    DSR at `f98a819` executed all ten graph tests successfully; the overall lane
    failed later in the unrelated archive fixture. Fresh DSR at `3bb1fb4` again
    passed all ten (106.02 seconds); all 14 combined-lane steps subsequently passed.
    I prematurely closed this bead, then reopened it when review found the tested
    window had no event exactly at its exclusive end. Candidate `fe2271d` supplies
    that discriminator and the independent row-cap control. Clean DSR at `c5ac1a9`
    passed all ten graph tests (312.88 seconds), and the full combined lane completed
    with typed pass. Re-closed only after the missing discriminator actually ran.
- [x] **bd-2z0.11.12 — real storage and CLI proof, closed, TurquoiseLake.** Persist a hand-enumerated fixture
  with repeated/directed/boundary events and two runs; close and reopen it; exercise
  actual report/export processes; independently parse counts, weights and selection;
  test missing capture evidence, empty/reversed/one-sided windows, zero limits,
  over-budget selection, nonfinite data and source isolation; retain execution logs.
  - [x] Ten tests written, including real file persistence, reopening, both CLI formats,
    two-run attribution, boundary selection and seeded centrality controls.
  - [x] Latest graph candidate ran 10/10 through actual clean DSR, including capture
    truncation, typo rejection and retained subprocess records.
  - [x] Persisted a third run with internally balanced counters contradicting actual
    edge rows; report and both export formats refused it in that DSR run.
  - [x] Claim companion proof after implementation closes. The implementation was
    subsequently reopened for missing boundary evidence; both are now closed after
    that evidence was supplied and executed.
  - [x] Execute the final graph suite in clean DSR `combined-c5ac1a9-20260905-06`
    through the actual analytics E2E wrapper: ten tests passed. Readback parsed 28
    unique CLI records (10 successes, 18 expected refusals), all expectations true.
    `[5,6)` selected only the two tick-5 rows, excluding the actual tick-6 event.
    The isolated row-cap case returned `interaction_graph.max_rows` with generous
    byte/work budgets. Real SQLite and subprocess outputs are retained on hz4 under
    `/data/tmp/scriptbots-dsr-20260905-TmJwOJ/proofs/combined-c5ac1a9-20260905-06/tmp/.tmprfk6Be`.
  - [x] Emit explicit per-pair count/magnitude residuals from all four saved exports.
    The one-off external readback accepted all four with zero residuals and rejected
    ten altered artifact/input expectations (changed expected row, dropped edge,
    missing weight, swapped run and false completeness, in both formats). Records:
    `/tmp/scriptbots-c5-graph-artifact-readback-20260905.jsonl`. These are artifact
    checks; the actual CLI denominator remains 28 and the graph test count remains ten.
  - [x] Require the complete combined lane's typed pass before closure. The paired
    graph beads share these ten tests; they are counted once. Fresh solo review
    is not independent verification.
  - [x] Replace the analytics E2E wrapper's ignored simulation failure, empty database
    and unconditional success with the actual pinned DSR graph lane. Full seeded
    simulation/report acceptance remains open under `bd-2z0.11.9`.
- [x] **bd-build-farm-reliability-lb19.1 — usable pinned DSR profile, closed.** Inspect installed
  DSR schema and available hosts; provide a portable reviewed profile/materialization
  path; bind source, toolchain, target and external evidence; distinguish preflight
  refusal from test failure; include workspace checks and core economy-fault tests.
  - [x] Reviewed native correctness profile and runner committed as `d8083c6`.
  - [x] Fresh isolated configuration on hz4 validates; actual DSR executes its guards.
    Missing `yq` supplied as an isolated official binary with verified SHA-256.
  - [x] First clean run retained a real WASM snapshot failure before compilation:
    `graphs-d8083c6-20260905-01`, external proof root
    `/data/tmp/scriptbots-dsr-20260905-TmJwOJ/proofs` on hz4.
  - [x] DSR accepted the two-line WASM snapshot update for existing genome PNG
    dependencies (`adler2`, `miniz_oxide`); no performance golden was changed.
  - [x] Workspace all-target check passed diagnostically after stale archive/meadow
    interface repairs. Strict Clippy still fails on 421 existing core diagnostics.
  - [x] Archive unit test passed in DSR. Archive integration exposed contradictory
    demographic fixture counts. Repair `2c091ef` keeps explicit births/deaths refusal
    checks and then admits a consistent tick; its diagnostic runtime rerun passed
    (one executed test, zero failures, eleven unrelated tests filtered).
  - [x] Clean DSR at `3bb1fb4` passed all 14 named combined-lane steps; process and
    typed verdict both report zero/pass. Required logs, actual test counts and
    source/profile/target identity were checked by the runner and launcher.
  - [x] Seven actual preflight cases parsed and retained at
    `/tmp/scriptbots-dsr-negatives-20260905/cases.jsonl`: missing config/profile/host,
    wrong source/target, dirty source and reused proof version. The undeclared-host
    case exposed DSR's SSH fallback; launcher `dc32bf4` rejects it locally.
  - [x] Added retained-evidence readback that rechecks required commands, log hashes,
    actual test counts and source/profile/target identity, plus a combined graph/recipe
    lane to share dependency compilation. This is verification infrastructure only.
  - [x] The same checker accepted the real positive bundle and rejected nine altered
    copies (missing/changed logs, stale source, non-pass verdict, duplicate commands,
    zero test count, missing profile/source record and wrong target). Parsed index on
    hz4: `/data/tmp/scriptbots-evidence-readback-20260905-bwl7kzmy/cases.jsonl`.
    These are retained-record checks, not additional simulation executions.
- [ ] **bd-build-farm-reliability-lb19.2 — actual DSR qualification.** Materialize from
  an empty configuration; exercise wrong-source/dirty/missing-host negatives; run
  check, strict Clippy, formatting and tests; retain actual counts and source identity;
  run performance only on the checked-in golden's exact class through DSR.
  - [x] Fresh configuration and seven real preflight controls exercised.
  - [ ] Workspace check, strict Clippy, workspace tests and explicit economy-fault
    tests must all execute successfully through DSR. Diagnostic check passed;
    strict Clippy's 421 existing core diagnostics remain unresolved.
  - [ ] Locate the checked-in golden's exact M4/macOS/toolchain machine class and
    run its pinned DSR comparison. No compatible host is available in this session;
    no performance command, fingerprint change or re-baseline was attempted.
- [x] **bd-2z0.13.8 — repair extension recipes, closed, TurquoiseLake.** Reconcile exact Brain/registry APIs;
  compile the literal custom-family recipe; normalize and execute the scenario;
  demonstrate actual host command application and projection; correct architecture
  claims at their source without removing extension topics.
  - [x] Confirmed the current tests never execute the advertised custom family,
    normalized scenario effects, or host command application.
  - [x] Four literal Rust recipes compile and run under rustdoc with real protocol APIs;
    diagnostic run: four passed, zero failed/ignored/filtered.
  - [x] Literal scenario executes both scheduled effects through production config
    composition, with matched-seed food-cell change and no-change controls.
  - [x] Host example observes applied command, volatile journal, matching projection
    and idempotent retry. It does not certify a production frontend.
  - [x] Current ownership, supported CLI choices and recipe scope corrected in place.
  - [x] Strengthened custom-family example checks changed output sequences and actual
    founder-bound genomes. Added four valid compiler/runtime cases and eight mutations
    of the actual guide, with retained exact compiler inputs and process outputs.
  - [x] Complete clean DSR `combined-3bb1fb4-20260905-04` on hz4. The prior
    `recipes-628cd49-20260905-03` executed four literal programs and nine ordinary
    integration tests, then was explicitly cancelled during an unnecessary second
    dependency build. It is not a pass. Candidate `3bb1fb4` uses artifact records
    from the integration test that already ran. The expensive mutation test is
    explicitly selected by DSR; ordinary discovery does not execute it.
  - [x] Latest workspace all-target check passed diagnostically after mutation-test
    additions. The standalone checker now derives target-specific library, metadata
    and transitive dependency paths from Cargo records; the final revision executed
    successfully in the complete `c5ac1a9` DSR run.
  - [x] Fresh review found the custom recipe accepted another genome's recurrent
    checkpoint. Candidate `209d5d4` binds state to the core material hash, advances
    the state schema/codec and adapter identity, and resets offspring with the
    child's binding. This is a real correction found after the earlier positives.
  - [x] Finish `recipes-209d5d4-20260905-05` through the actual recipe wrapper.
    Its four documentation programs, nine ordinary tests and all eleven standalone
    cases (four valid, seven mutations) passed; all ten required steps completed
    with DSR exit zero and typed pass. This verifies the checkpoint binding fix.
  - [x] Readback found meteor had no saved observation and scenario/frontend used
    prose. Candidate `c5ac1a9` emits parsed, versioned scientific records from all
    four literal programs. It retains measured food-cell comparisons, applied
    meteor cells/tick and actual command/journal/projection states.
  - [x] Finish the combined run at `c5ac1a9`, including its eighth mutation:
    the meteor process exited zero after redirecting its observation to stderr,
    and the same acceptance predicate rejected its empty stdout. All 14 lane steps
    passed; the twelve standalone cases finished in 13.86 seconds.
- [x] **bd-2z0.13.10 — execute extension recipes, closed, TurquoiseLake.** Register/evaluate/mutate/reproduce
  and checkpoint the documented family; execute both scheduled scenario effects;
  correlate host application, journal and projection; mutate the real recipe as the
  negative control; replace tests that only grep names or check disconnected strings.
  - [x] Four literal programs executed under rustdoc in DSR at `628cd49`.
  - [x] Run those four through the retained standalone compiler/runtime checker at `209d5d4`.
  - [x] Observe all seven declared mutations fail that same checker at `209d5d4`: broken evaluator
    method, missing founder installation, disabled genome/state binding, nonexistent
    CLI preset, unknown literal TOML key, omitted meteor step and admission
    substituted for application.
  - [x] Observe the additional missing-observation mutation and parse all four valid
    scientific records at `c5ac1a9`: twelve unique source-bound cases, all expectations
    observed. Four positives, two compiler refusals, five runtime refusals and one
    successful process refused for missing observation. The ordinary suite's ignored
    expensive test was explicitly selected and executed by the DSR mutation step.
  - [x] Read actual compiler diagnostics and runtime outputs; the four positive saved
    records match stdout exactly. Observed 16 bound custom founders, changed output
    sequences, rejected foreign-genome state, both scenario patches, food-cell
    differences of 0/64 before and 35/64 after drought application, 17 meteor cells,
    and applied/volatile-journal/projection agreement with 14 visible agents and no
    duplicate retry tick. Fixture on hz4:
    `/data/tmp/scriptbots-dsr-20260905-TmJwOJ/proofs/combined-c5ac1a9-20260905-06/tmp/literal-recipes-caTyY0`.
  - [x] Claim after the paired implementation closed, then close after fresh review
    against the original criteria. These twelve cases support both recipe beads and
    count once. This was solo re-verification, not an independent verifier.
- [ ] **bd-2z0.13.9 — compose the Evolution Lab journey.** Reuse production startup,
  scene inspection, control receipts, durable history, checkpoint continuation,
  matched-seed experiment and report/bundle paths; bind each stage to predecessor
  identities; derive required frontend cells from declared support; expose actual
  subset coverage while required upstream work remains incomplete.
  - [x] Inspect existing consumers and declarations. No canonical versioned product
    matrix exists yet; `bd-1bdd` owns it. Browser hardware coverage is a different
    matrix and cannot define release support. The manifest is not implemented.
    Recorded the actual blocking dependency on `bd-1bdd`; no active cycle resulted.
  - [ ] Consume the owning product matrix without shrinking its required cells.
  - [ ] Join startup, inspection, applied/journaled controls, durable run, checkpoint,
    matched-seed comparison and report/bundle records by their actual identities.
  - [ ] Require every declared stage/cell and preserve failed/refused/unexecuted
    dispositions. Component passes cannot satisfy the final product journey.
- [ ] **bd-2z0.13.11 — final journey proof.** Execute the composed journey and its
  identity/seed/checkpoint/receipt/artifact negative controls after dependency proofs;
  verify all required frontend cells; keep this open until actual production evidence
  satisfies every original acceptance condition.
- [ ] **Existing dependency work:** preserve current owners of bd-pcfj/bd-88yj
  (host ownership), bd-2z0.10.5 (meadow), and other in-progress leaves; implement
  unowned ready prerequisites when they become the highest-value next step.
  Track checkpoint bd-2z0.5.13, scenarios bd-2z0.10.4/.5, analytics bd-2z0.11.9,
  release bd-1bdd, TUI bd-2z0.6.8, Bevy bd-2z0.7.2 and browser bd-2z0.12.3.
  - [x] Meadow helper test now uses production config composition and actual
    Ratatui TestBackend frames; CPU PNG and TestBackend scope is explicit.
  - [ ] Keep full meadow acceptance open: diagnostic cohort failed for seed
    `20260717` (zero deaths in 300 ticks versus the declared floor of one).
    Seeds 42 and 137 completed both helper paths with matching digests and no
    ledger breaches. The third CPU path and later controls were not reached.
    No seed, horizon, population envelope or demographic floor was weakened.
- [ ] **For each implementation block:** meaningful positive and discriminating
  negative tests; required compiler/lint/format checks; fresh self-review against the
  original bead; accurate status and evidence; reviewed exact-path commits and sync.
  No test weakening, fixture-as-live claims, or closure of unmet positive requirements.

Work-block corrections retained: the premature graph closure was reopened and repaired;
the `209d5d4` UBS scan was mistakenly run after its commit (the existing test-only panic
finding was inspected, with no suppression). Subsequent code commits were scanned
before committing. The full scratch audit, including all twenty honesty prompts, is
`/tmp/scriptbots-work-block-audit-20260905.md`. Strict Clippy remains red; no clean
scanner, full-workspace, performance, GUI, PTY or browser result is claimed.

## Current verdict and evidence boundary

**ScriptBots contains a substantial simulation and research substrate, but it does not yet deliver the complete Evolution Lab journey.** The limiting work is connecting the production host, frontends, checkpoint continuation, scientific reports, and release evidence. Completing components or closing their issues does not prove those connections.

This refresh supersedes the September 3 assessment preserved below. Source audited: `fa1fb08b1bc1a341f2ea4f0638ae2ecaa7eb7dbf` on `main`. The initial checkout already had a modified `.beads/last-touched` and an untracked compiler ICE report; neither was treated as audit-authored code. No application code is changed by this assessment.

Method: read all 1,031 lines of AGENTS.md and 1,053 lines of README.md; read the recovery plan, four historical root plans, architecture guide, rendering plans/specification, browser ADRs/plans, analytics contracts, and integration decisions. The recovery plan is the active measuring stick; historical proposals do not override it. Inspect implementation, production callers, feature declarations and tests by capability. Search the full tracker snapshot, including closed issues, before adding completion debt. Keyword scan found 73 matching Rust lines; AST scan found five `unimplemented!()` calls, all in tests/test-only modules. These counts are scan results, not defect counts. Behavioral inspection found more consequential gaps without those macros.

Initial tracker: **636 issues: 456 closed, 173 open, 6 in progress, 1 blocked; 180 not closed.** `br ready` returned 63 entries, including aggregates. The authoritative BV wrapper reported 87 graph-actionable entries and explicitly warned that only BR authorizes claims. Active dependency cycle count was zero. No percentage of product completion is inferred from these counts.

Fresh execution evidence is limited:

| Observation on this host | Result | What it proves |
|---|---|---|
| `dsr --json status` | configuration invalid; no configured hosts | The required acceptance lane is not provisioned here. |
| `dsr build --tool rust_scriptbots --target darwin/arm64 --no-sync --version reality-20260904-fa1fb08` | refused: repository profile absent, before compilation | Neither a red suite nor a green suite; no current-source runtime acceptance. |
| `bash ci/check_fsqlite_pin.sh` | exit 0 | Current manifest/lock/AGENTS pin consistency only. |
| `bash ci/check_franken_licenses.sh` | exit 0 | Family inventory has documentation entries; not a legal or packaged-archive audit. |
| `br dep cycles --json` and authoritative BV triage | zero active cycles; snapshot accepted | Tracker structure/freshness, not product correctness. |
| GitHub Releases API | v0.1.1, published 2026-07-22, zero attached assets | A release record exists; it does not provide downloaded native/WASM binaries to verify. |

Raw local observations are retained in `/tmp/scriptbots-reality-20260904-G9j3WJ` (tracker snapshot, triage, scans, DSR refusal, guard logs, release JSON). These are diagnostic files, not a portable acceptance bundle. A discovered `/data/tmp/.../pkg/scriptbots-app` was an 18-byte shell fixture that only prints `ok`; it was excluded as runtime evidence. No current-source application test suite, GUI/PTY/browser journey, or benchmark was executed successfully in this audit. Historical green tests remain historical evidence at their recorded commits.

## Vision checklist: component evidence versus product delivery

`PARTIAL` means substantial code exists but the stated journey has missing implementation or integration. `UNPROVEN` means the claim exceeds available execution evidence. `STUB` is reserved for an observed placeholder. No row is marked `WORKING`: the skill requires tests and E2E verified, and this host could not provide fresh acceptance. This does not assert that every component is broken.

| ID | Testable promise / source | Current reality and representative source | Gap owners already present |
|---|---|---|---|
| V01 | Living, understandable default meadow; recovery plan §18.1 | PARTIAL. Scenario catalog, founder installation and terminal paths exist. `tests/meadow_acceptance.rs` manually copies selected config fields; its GUI lane calls `world.step()` and a CPU PNG renderer. That is not default production GUI startup. | `bd-2z0.10.4`, `.10.5`; `bd-bacf` |
| V02 | One simulation owner; rendering cannot advance science; plan §4.1 | PARTIAL. `runtime/host_core.rs`, native lifecycle and `app/host_thread.rs` are real. `app/main.rs:434` still creates `SharedWorld`; `control.rs:390` retains it; Bevy takes `Arc<Mutex<WorldState>>` and spawns a worker. | `bd-pcfj`, `bd-88yj`, `bd-2z0.7.2`, `.6.1`, `bd-37m` |
| V03 | UI/REST/MCP acknowledged controls, ordering and stream recovery; README control contract | PARTIAL. Runtime lifecycle/journal contracts exist, but `ControlHandle::with_world` still locks the world. Its legacy status cannot be upgraded to durable merely because runtime/storage implement journal types. | `bd-5dkk`, `bd-6mus`, `bd-g6wf`, `bd-ydu8`, `bd-2z0.12.2` |
| V04 | Real heritable default brains; plan §18.2 | UNPROVEN at this HEAD, with substantial implementation. `app/brains.rs::install_brains` structurally admits MLP/DWRAON/Assembly protocol families. NeuroFlow is withheld from mixed founders but explicitly selectable. Placeholder ML is not installed. | Protocol family work largely closed; `bd-2z0.3.12.3`–`.6` retain FtBrain completion/proof. |
| V05 | Honest ML/native family capability claims; README brains | PARTIAL. Candle tensor and batch functions are real in `brain-ml/src/candle.rs`; no Candle `BrainPreset` or protocol family is installed. Default brain-ml still copies sensors; Tract/tch remain selection/probe paths. Tensor errors fall back to scalar in `forward`. | `bd-1bdd` owns feature truth; closed `bd-rcae` delivered tensors, not product family admission. |
| V06 | Determinism, stable identity, meaningful evolution; plan §18.2–3 | UNPROVEN at this HEAD, with real six-domain RNG, UID substreams, genome/evaluator codecs, staged step, resource ledger, digest v1.7 and trace contracts. Existing tests include mixed-family batch/scalar parity and rollback. Remaining knob witnesses and default sensing discrepancy matter scientifically. | `bd-dorx`, `bd-3mul`, `bd-6i23`, `bd-m30b`, `bd-16g.2.9` |
| V07 | Production checkpoint continuation and first divergence; plan §4.1/18.3 | PARTIAL. Core checkpoint v1.3 is implemented but explicitly excludes persistence/session ownership. `main.rs::run_replay_cli` logs the latest checkpoint, then starts ordinary headless replay. `experiment_runner.rs` refuses interrupted running-status resume. | `bd-2z0.5.13`, `.11.4` |
| V08 | Exact durable admission, recovery, bounded waits; storage contract | PARTIAL at product level. Real sole-thread fsqlite worker, outbox identities, three watermarks, leases, recovery and read snapshots exist. Deadlines cannot cancel in-flight SQL or guarantee a bounded reaper. Live memory-mode admission timeout remains reported. | `bd-w1oi`, `bd-2z0.5.14`, `.5.17`; storage protocol children largely closed. |
| V09 | Portable bundles and repeatable multi-seed experiments; plan §4.2–3 | PARTIAL. CLI bundle create/verify and `MatchedSeedExperimentRunner` execute real persisted runs and verify finished bundles. Mid-run resume and independent-checkout reproduction remain separate acceptance obligations. | `bd-2z0.5.13`, `.11.4`, `.11.9`, `bd-16g.1.7`, `bd-1bdd` |
| V10 | Immutable, bounded snapshots with proven latency; plan runtime budgets | PARTIAL. Runtime projections and hub exist; retained allocation/latency proof and some event scaling remain open. No application-wide lock-free world claim is justified. | `bd-2z0.4.8.1`, `.4.16`, `bd-g6wf` |
| V11 | FrankenTUI canvas, inspector, science screens and palette; plan §8 | PARTIAL. Ratatui is the actual dependency. `terminal/frankentui_shell.rs` is a model/harness; no workspace crate consumes the prepared ftui pin. Live pointer, screens and receipt integration remain. | `bd-2z0.6.1`, `.6.5`, `.6.6`, `.6.8`, `.14.2.3`–`.5`, `bd-dkd9` |
| V12 | Primary batched Bevy GUI, preserved scientific visual semantics; plan §9 | PARTIAL. Real Bevy renderer, camera/HUD and offscreen render-graph capture exist. Per-agent multipart entities/materials and separate simulation ownership remain; semantic golden failure is an existing report, not rerun here. | `bd-2z0.7.2`–`.4`, `.7.14`, `.14.1.1`, `.14.3.9`–`.11`, `bd-d26y` |
| V13 | Cinematic terrain/water/creatures/light/VFX/camera/audio; cinematic program | PARTIAL. Many helpers and settings exist, but connected effects, instancing and measured frame budgets remain a large program. A config declaration or CPU image is insufficient proof of an actual frame. | `bd-2z0.14.1.1`–`.9`, `.1.11`, `.1.18`, `.14.3.5`, `bd-ogcs`, `bd-rl1h` |
| V14 | Brain/sense/lineage inspection explains behavior; plan science UX | PARTIAL. Bounded inspection and attribution algorithms/panels exist. Production edge projections can yield `NoConnections`; live phylogeny layout/views and per-eye selection are incomplete. | `bd-16g.4.4`, `.4.5`, `.3.4`, `.3.5`, `bd-r7cz`, `bd-2z0.7.15` |
| V15 | Distinct cohort-proven scenarios and interventions; plan §2.3 | UNPROVEN at this HEAD, with six checked-in scenarios, normalization/seeding, schedules and catalog tests. Final onboarding and full production meadow proof remain. | `bd-2z0.10.4`, `.10.5`, `bd-2z0.13.3` |
| V16 | Scientific graph/stats/dataframe exports; plan §4.4 | PARTIAL. Analytics registry, native stats, fnx graph calls, persistent interaction readers and exporters are real. Full pipeline proof remains. Graph doc promises weighted/count edges; `build_interaction_digraph` only inserts node pairs. Query fallback and window semantics differ. | `bd-2z0.11.6`, `.11.9`, `bd-metric-summary-test-memory-8wxv`; specific graph completion debt below. |
| V17 | Narrated persisted timeline and calibrated detector; evolved science vision | PARTIAL. `detect.rs`, narrative persistence and text exist; complete persisted-input parity, bounded cleanup and scrub/focus UI remain. | `bd-16g.2.9`–`.11`, `bd-ji3a`, `bd-farh`, `bd-2z0.5.17` |
| V18 | Information theory / quality diversity / communication affect useful analyses | PARTIAL. MI/TE and MAP-Elites algorithms exist; MI/TE production consumers, archive persistence, novelty selection and communication scenarios remain. | `bd-xqd5`, `bd-r4ja`, `bd-16g.6.1`–`.3`, `.7.1`, `.7.3` |
| V19 | Multi-island evolutionary experiments with one durable history | PARTIAL. Archipelago barriers/migration are substantial; canonical overrides, real outer parallelism and production `persist_barrier` connection remain. | `bd-5tyo`, `bd-16g.5.4`, `.5.5.2`, `.5.5.4`, `bd-t3ge`, `bd-brw4` |
| V20 | LLM proposes, validates, runs and reports bounded experiments | PARTIAL. `lab_assistant.rs` has a state machine and a real `MatchedSeedExecutor`; this is more than a stub. Full budget/refusal/notebook/adversarial acceptance remains. | `bd-16g.1.3`, `.1.7`, `.16`, `bd-6mus` |
| V21 | Browser rendering, controls, parity and durable storage | PARTIAL. `web/main.js` directly runs WASM and Canvas on RAF; no WebGPU import, renderer selector or binary-format control. Browser WASM APIs support more than this page exposes. Storage Worker/recovery and runtime isolation remain. | `bd-2z0.12.3`, `.12.4`, `.12.7`, `bd-ywtv`, `bd-azi3`, `bd-ac4l` |
| V22 | Shareable runs, forks, gallery, reels, tournament and sonification | PARTIAL. Core gallery/permalink/reel/audio and app montage/tournament code exist. End-to-end fork/gallery/reel/ratings/live transport remain tracked. Browser audio ADR remains a deferred decision, not an accepted shipped capability. | `bd-16g.8.2`–`.3`, `.9.2`–`.3`, `.12.2`–`.3`, `.14.2`–`.3` |
| V23 | Faster GPU science with exact CPU parity | STUB for current WGSL accumulation. `world-gfx/src/sense_wgsl.rs` only zeroes `saturations`; its validator checks two substrings. CPU fixed-point primitives do not prove GPU sensing. | `bd-16g.15.2`, `.15.3` |
| V24 | CPU/frame/memory budgets on named hardware | UNPROVEN here. CPU gate covers 1k/5k default MLP/Neuro lanes; 10k publication is separate. GPUI zero-copy/FPS and scoped-CPU decision numbers lack inspected retained artifacts. No fresh performance result claimed. | `bd-h33`, `bd-kuho`, `bd-2z0.7.7`, `.4.15`, `.14.3.5.3` |
| V25 | Maintainable, reproducible builds, correct extension recipes and releases | PARTIAL. Strong pin/license/graph guards and reviewed commits exist. Required DSR profile is absent here; recipe tests do not execute documented extensions; release has no attached binaries. | `bd-build-farm-reliability-lb19`, `bd-1bdd`, `bd-2z0.9.2`–`.3`; focused debt below. |

## What the old report got wrong

1. **ControlHandle is not migrated.** The old follow-up inferred implementation from closed `bd-k7nq`. The field and lock in `control.rs` falsify it. `bd-88yj` already explicitly records preparation, not completed migration.
2. **The libudev blocker is historical.** `bd-rch-full-workspace-lane-5mff` closed after worker provisioning; this audit's blocker is missing local DSR configuration, not a repeated Bevy compiler failure.
3. **Candle tensor execution exists.** It is neither just the original scalar probe nor a fully admitted product family. Both halves must be stated.
4. **Checkpoint capability is overstated in code too.** `CharacterizationLimitationsV0::default` currently sets `checkpoint_replay_guarantee: true`, although evaluator coverage is false and replay does not restore the discovered checkpoint. `bd-2z0.5.13` already owns this exact defect.
5. **Graph exports and fnx code are implemented.** Their remaining semantics/proof must be inspected, not described as wholly absent. Whole-window extraction and recent-page/fallback modes are different scientific populations.
6. **There is no defensible delivery date here.** The old claim that `2 - 1/6` means within 8% is arithmetically false (it is about 1.833). Its confidence-weighted ETA is not a calibrated estimator. Fleet capacity was inferred from incomplete failed-run observations. All old dates/capacity/near-optimality claims below are withdrawn.

## Bridge plan and coverage verdict

**Completing the existing open beads would close most named implementation gaps, but would not by itself prove the whole vision.** Existing release and science epics are broad; they need explicit source-bound acceptance at the joins. Closed recipe/graph work also leaves concrete requirements without an open leaf owner. Preserve all existing features and delivery; add narrow completion tasks rather than reopen large delivered epics.

| Gap | Concrete work and acceptance | Existing coverage / planned addition |
|---|---|---|
| G1: extension recipes prove different operations | Compile the actual documented custom-family source; normalize and execute the documented scenario; attach a frontend to the real host and correlate application/journal receipts. Correct current-state claims from callers. Guard by mutating the actual recipe, not a disconnected phantom string. | New recipe repair plus companion production E2E under `bd-2z0.13`; preserve closed `bd-bsuh` contribution. |
| G2: graph population and edge semantics drift | Define bounded window/recent/fallback selection, zero-limit behavior, explicit capture completeness, duplicate aggregation and weighted versus unweighted algorithms. Preserve doc-promised count/magnitude attributes in exports; independently verify hand-built directed multievent fixtures. | New graph semantics repair plus real-storage/CLI tests under `bd-2z0.11`; integrate with `.11.9`. |
| G3: acceptance lane cannot be reproduced by a fresh agent | Provide a reviewable portable DSR profile and host prerequisites, bound to clean main/expected SHA, exact target and external unique evidence directory. Retain typed refusal/build/test/runtime/performance outcomes; never synthesize machine fingerprints or reuse dirty evidence. | New profile portability plus fresh-host proof under build-farm epic; existing fleet repair remains separate. |
| G4: no single verified Evolution Lab journey | Bind the supported product/platform matrix to a real scenario → control receipt → persisted run → inspection → checkpoint continuation → comparison → report → second-checkout bundle journey. Each stage must consume preceding artifacts and prove the observation it claims. | New composed acceptance task and negative-control companion under `bd-2z0.13`; reuse `.10.4`, `.10.5`, `.5.13`, `.11.9`, `bd-1bdd`. |
| G5: remaining documented promises | Host first, then checkpoint/persistence/control joins, then frontend and scientific consumers, while renderer implementation proceeds against the same snapshots. Extend existing acceptance instead of making duplicate feature tickets. | Owners mapped in V01–V25; no performance or browser promotion based on a static scan. |

Execution order is dependency-based: prepare reproducible acceptance; finish `bd-pcfj` and `bd-88yj`; migrate TUI/Bevy/GPUI consumers and command transports; integrate checkpoint continuation and scientific artifacts; complete frontend/science feature gates; compose the real release journey. Independent docs/graph repairs can proceed without waiting for host cutover. A local infrastructure failure blocks an acceptance result, not independent implementation. Use `br ready` for claims and BV for analysis, never a synthetic timetable.

## Current-session phase record

- Phase 1: documentation-first vision extraction, source/caller/feature/test audit, keyword + AST + behavioral checks, tracker coverage and release inspection completed; fresh DSR execution attempted and explicitly refused by missing configuration.
- Phase 2: V01–V25 and G1–G5 above define the bridge and preserve all existing feature programs.
- Phase 3a: created four bounded completion tasks and four companion proof tasks using only `br`. Each has standalone background, source findings, implementation boundaries, positive/negative tests, retained logging and explicit acceptance. IDs are bound in the round record below; no application fixes or acceptance passes are implied by planning completion.

### Ambition round 1: preserve the complete user journey

The first bridge could still become four isolated improvements. Strengthen it around the actual joins: scenario/config → tick-zero world → stable-agent inspection → command application → persistence → checkpoint continuation → report/export. The same run, source and cohort identities must flow between those stages. Production GUI/PTY/browser evidence must name the surface actually executed, and every declared required matrix cell must be represented. A component test or a successful recipe parser cannot substitute for any join.

The eight new tasks are `bd-2z0.13.8` (recipes), `bd-2z0.13.10` (recipe proof), `bd-2z0.11.11` (graphs), `bd-2z0.11.12` (graph proof), `bd-build-farm-reliability-lb19.1` (DSR profile), `.2` (fresh-host proof), `bd-2z0.13.9` (journey manifest), and `bd-2z0.13.11` (journey proof). Existing feature owners remain authoritative.

### Ambition round 2: failure and resource boundaries are part of the product

Require rejection evidence at each boundary: stale source, foreign run, partial seed cohort, lost receipt, indeterminate admission, unknown capture completeness, truncated report input, unavailable adapter and corrupted checkpoint. These states remain distinct from valid empty results. Boundedness means row/byte/work admission limits plus honest disclosure of in-flight calls that cannot be cancelled; a timeout alone is not proof of reclaimed resources.

For interaction graphs, selecting a complete tick window and selecting a newest page are different operations. The implementation must validate the mode before loading data and refuse an unbounded complete request rather than quietly sample it. For releases, profile portability can be implemented before host access is available, but qualification and final journey execution stay blocked until actual evidence exists. This keeps independent work available without turning missing infrastructure into success.

### Ambition round 3: use scientific invariants, not decorative scheduling math

Make proof strength concrete. For an event multiset E and directed pair (a,b), require `count(a,b) = number of selected events with that pair` and `weight(a,b) = sum of their declared magnitudes`; total edge counts must equal selected events, including duplicate encounters. Check graph export against these raw-event identities. Keep unweighted centralities explicitly unweighted unless a separate weighted contract is implemented. A lineage DAG can merge founder components through two-parent reproduction: the old analytics E2E promise that component count always equals founder-cohort count is valid only for a fixture that prohibits such cross-cohort births.

For experiments, compare matched seeds as pairs, retain every declared seed, and distinguish the distribution of independent runs from repeated snapshots within one run. A planted effect and null control test different failure modes. Do not replace a false-discovery rule with a single favorable p-value or infer absence of behavior from sampled-out events. Keep these requirements in the existing science owners rather than add a speculative algorithm program.

Completion is a conjunction over required evidence cells and identity joins, not an average percentage: one required missing/refused cell prevents final pass. This is directly computable and falsifiable. There is no ETA without measured remaining durations, resource availability and uncertainty calibration.

### Bead regeneration and refinement

After all three ambition rounds, Phase 3a was reapplied to the same bridge: strengthened the four implementation acceptance fields; added scientific and boundary notes; updated existing `bd-2z0.10.5`, `bd-2z0.11.9` and `bd-1bdd` without taking their ownership or closing them.

1. **Refinement 1 — ownership and dependency review:** all eight new tasks retain open parent owners and companion implementation/proof separation. Added the final journey proof's explicit dependencies on recipe/graph/fresh-host proof and existing host, checkpoint, onboarding, meadow, analytics and release-matrix work. Harness authoring remains unblocked. Active cycle check still returns zero. This corrects a plan-space gap: a final acceptance task could otherwise be claimed before its required surfaces exist.

2. **Refinement 2 — test discrimination:** reviewed all new proof tasks against the observation they claim. Added actual live-family binding/output, scheduled config effects, receipt-to-projection correlation, different old/new graph fixtures, zero-limit fallback, graph error propagation, and preflight-before-execution checks. Valid zero observations remain distinct from missing evidence. Each negative changes an input used by the actual positive path.

3. **Refinement 3 — coverage and infrastructure:** prevented a supported-subset release result from claiming full V01–V25 completion; added explicit final-proof prerequisites for real FrankenTUI, snapshot-only Bevy and browser rendering. Linked full analytics E2E to graph semantics proof. Recorded the existing hz4 memory failure as a host-specific resource disposition, without inventing a failure on the unprovisioned DSR host or silently skipping that test.

4. **Refinement 4 — feasibility and contract consistency:** corrected an over-broad sensitivity requirement in the journey proof: an intervention must change the quantity it is designed to affect, not every unrelated metric. Valid null results remain legitimate. Required explicit edge-list attribute representation or typed refusal so GraphML/count/weight promises cannot disappear in a less expressive export format. No feature or negative control was removed.

5. **Refinement 5 — convergence review:** rechecked the eight standalone descriptions/acceptance fields, existing coverage notes, dependency directions and final journey prerequisites. All eight remain open; the four implementation/harness tasks are BR-ready and each companion proof depends on its implementation. No additional change was found in this bounded review. This is convergence of this bridge, not a claim that all repository defects were found. Tracker now has **644 issues: 456 closed, 181 open, 6 in progress, 1 blocked; 67 BR-ready**. No existing issue was closed or reassigned by this audit. Active dependency cycles remain zero.

Final document verification: `git diff --check` passes. UBS was invoked on this Markdown report and returned exit 3, explicitly **nothing scanned**; that is not a code-scan pass. No Rust changes were made, so no compiler result is asserted. DSR provisioning and all unexecuted runtime/performance acceptance remain visible work in the new profile/proof tasks and existing capability owners.

## Frozen operators

Phase 3a is applied verbatim to the bridge and again after the ambition rounds:

```text
OK so please take ALL of that and elaborate on it and use it to create a comprehensive and granular
set of beads for all this with tasks, subtasks, and dependency structure overlaid, with detailed
comments so that the whole thing is totally self-contained and self-documenting (including relevant
background, reasoning/justification, considerations, etc.-- anything we'd want our "future self" to
know about the goals and intentions and thought process and how it serves the over-arching goals of
the project.) The beads should be so detailed that we never need to consult back to the original
markdown plan document. Remember to ONLY use the `br` tool to create and modify the beads and add
the dependencies.
```

Every refinement pass uses this unchanged operator:

```text
Check over each bead super carefully-- are you sure it makes sense? Is it optimal? Could we change
anything to make the system work better for users? If so, revise the beads. It's a lot easier and
faster to operate in "plan space" before we start implementing these things! DO NOT OVERSIMPLIFY
THINGS! DO NOT LOSE ANY FEATURES OR FUNCTIONALITY! Also make sure that as part of the beads we
include comprehensive unit tests and e2e test scripts with great, detailed logging so we can be
sure that everything is working perfectly after implementation. Make sure to ONLY use the `br` cli
tool for all changes, and you can and should also use the `bv` tool to help diagnose potential
problems with the beads.
```

## Historical assessment (superseded; not current acceptance evidence)

The original text is retained for auditability. Its working labels, closed-bead inferences, claimed host migration, scheduling/capacity math, and forecasts are explicitly superseded by the refresh above. Historical executions were not repeated here.

<details>
<summary>September 3 assessment and follow-up record</summary>

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

</details>
