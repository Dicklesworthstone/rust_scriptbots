# Phase 1 WebAssembly Sibling Crate TODOs

_Created: 2025-10-22 (UTC). Keep this list exhaustive; mark completion inline with `[x]`._

## Legend
- `[ ]` Task not started
- `[~]` Task in progress
- `[x]` Task complete
- Indent subtasks beneath parent tasks; avoid deleting completed items (append completion note instead).

## Top-Level Objectives
1. Produce baseline research artifacts for Phase 1 (dependency audit, browser matrix, ADR scaffolding).
2. Establish repeatable research workflow (logs, citations) to feed future ADRs.
3. Capture security requirements (COOP/COEP, CSP) before any code lands.

---

## Tasks

- [~] Maintain planning doc status markers
  - [~] Update `PLAN_TO_CREATE_SIBLING_APP_CRATE_TARGETING_WASM.md` with `[Currently In Progress]` tags for active subsections.
  - [ ] Refresh markers when tasks exit Phase 1.

- [~] Dependency audit deliverables
  - [~] Run `cargo tree --workspace --edges normal,build,dev` to gather dependency graph.
  - [ ] Parse and classify dependencies into `dependency_audit.csv`.
  - [ ] Document blockers/owners in `PHASE1_FINDINGS.md`.

- [ ] Browser capability matrix
  - [ ] Collect WebGPU / SharedArrayBuffer / SIMD availability data with citations.
  - [ ] Populate `docs/wasm/browser_matrix.csv`.
  - [ ] Note refresh cadence and triggers in the matrix header.

- [ ] ADR-001 rendering experiments
  - [ ] Create ADR scaffold at `docs/wasm/adrs/ADR-001-wasm-rendering.md`.
  - [ ] Document evaluation criteria sourced from latest WebGPU/WebGL guidance.
  - [ ] Schedule prototype spikes for WebGPU and Canvas paths.

- [ ] Security baseline
  - [ ] Draft `docs/wasm/SECURITY_NOTES_PHASE1.md` covering COOP/COEP, CSP, SAB requirements.
  - [ ] Capture references (Chrome, Mozilla, W3C) for header guidance.

- [ ] Research log
  - [ ] Initialize `docs/wasm/RESEARCH_LOG.md` with date-stamped entries.
  - [ ] Record links and summaries for all findings generated today.

- [ ] Reporting
  - [ ] Summarize completed work and outstanding items in final agent response for traceability.

---

## Parking Lot (Revisit Later)
- Evaluate `cargo component` readiness (Phase 1.7) once dependency audit is complete.
- Determine whether to integrate automated tools (`carton`, `wasm-opt`) into CI during Phase 2.
