# Terminal Stack Integration & FrankenTUI Pin Specification (bd-2z0.8.8)

**Date**: 2026-07-22  
**Bead**: `bd-2z0.8.8`  
**Author**: AntiGravity / StormyFern  

---

## 1. Executive Summary

This document specifies the exact terminal stack modernization and FrankenTUI dependency pin for ScriptBots.

### Pinned Integration Details
- **Crate**: `ftui`
- **Version Constraint**: `=0.5.0`
- **Git Repository**: `https://github.com/Dicklesworthstone/frankentui`
- **Pinned Revision**: `15cc6543f76b814394c590f9e7719dedd6684e4c` (includes upstream lifecycle/simulator model completion fix `15cc6543`)
- **Default Features**: `false`
- **Enabled Features**: `["crossterm"]`
- **Adoption status**: **ADOPTED IN SCRIPTBOTS-APP (bd-2z0.6.8).** Consumed by
  `scriptbots-app` (`ftui.workspace = true`) for the actual interactive `Program` and
  `Model`, shared science providers, command receipts and ProgramSimulator harness.
  Pinned revision resolves in `Cargo.lock`. Existing Ratatui widgets render into
  FrankenTUI frames; their source is retained. Adoption does not close the original
  native acceptance under `bd-2z0.6.8` or the actual PTY byte/resize work under `bd-dkd9`.

> **Why the revision was wrong, recorded so it is not repeated (bd-phj8).** This
> field previously read `15cc65438a2095fbe8dd0dfce9adcfc7edab7612`, which is not
> an object in the upstream repository — `git cat-file -t` against the local
> frankentui git db reports "could not get object info", while the corrected SHA
> resolves to a commit. The two share the `15cc6543` prefix, i.e. the real short
> SHA was expanded with a fabricated tail.
>
> It survived because **an unconsumed pin is never resolved**. Cargo only fetches
> a `[workspace.dependencies]` entry when some crate actually depends on it, so a
> nonexistent `rev` in an unused entry cannot fail a build, cannot reach
> `Cargo.lock`, and cannot be caught by any amount of compiling. That is the
> general hazard, not an ftui quirk: a dependency pin with no consumer is
> unverified by construction.

---

## 2. Terminal Stack Compatibility Matrix

| Dependency | Version / Source | Role | Status |
| :--- | :--- | :--- | :--- |
| `ftui` | `=0.5.0` (`rev = "15cc6543..."`) | Pinned FrankenTUI engine | Pinned in `Cargo.toml` |
| `ratatui` | `0.30.0-alpha.5` | Retained widgets and headless buffer adapter | Consumed by the live FrankenTUI Model (`bd-2z0.6.1`) |
| `crossterm` | `0.27` | Terminal raw mode & backend | Active |
| `supports-color` | `3.0.2` | Color level detection | Active |

---

## 3. Current runtime and qualification

- Source through `cdab86ae` stages bounded commands and checkpoint requests until their
  complete pending identity is visible in a successfully presented frame. Skipped frames,
  insufficient frame size and write failure cannot authorize dispatch. Quit and shutdown
  cancel requests that have not reached the owner.
- Each own final result then requires its exact successful presentation before another
  request dispatches. Active receipts are pinned against bounded-history eviction; actual
  checkpoint catalog ID/tick details survive another queued `k` and reduced rendering.
- Own palette Shutdown additionally requires the final journal observation before automatic
  owner exit; overlapping attempts preserve each refusal/result and exact retry.
- A wrapped pending label removes the covered pointer bindings and cancels an active
  gesture. Command/checkpoint presentation alternates when both queues have work.
- The shared production and fault-probe configuration lets the existing governor select
  `EssentialOnly`; it retains the 16 ms budget and original skip policy. The reduced view
  identifies its quality level and preserves pending receipts.
- Completed R28 at `f2b0748a` passes formatting, dependency guards, all-target check,
  strict Clippy, 218 runtime tests and 309 terminal tests. All three actual fault PTYs
  select reduced detail, positively present, trigger their fault and restore. The executable
  observer then fails before app launch on builtin Python module identity. R32 fails formatting
  and R34 fails Clippy with zero tests. R36 fails formatting; R38 passes strict Clippy/check,
  218 runtime and 309 terminal tests, but three terminal assertions fail. Independent review
  corroborates their physical-wrap, active-Pending and journal-finality observation defects;
  test-only `cdab86ae` retains wrong-ID, corrupted-cell and unfinished-journal negatives.
  R40 is executing; full native acceptance remains open. Injected simulator callbacks do not
  prove terminal writes.
- Linux/unwind production-stack test children do not establish release-abort recovery or
  the separate full storm/TERM/locale/color/signal matrix under `bd-2z0.14.3.5.2`.
- [The current reality check](../REALITY_CHECK_2026-09-03.md) keeps the granular TODO,
  exact proof identities and original acceptance owners. No source files were removed.
