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
- **Adoption status**: **PREPARED, NOT ADOPTED.** No workspace crate consumes
  `ftui.workspace`, and `ftui` does not appear in `Cargo.lock`. This matches
  `docs/franken_integration.md`, which lists the ftui family as *planned* rather
  than in-tree. Adoption is an admission decision under that document, not a
  side effect of correcting a pin.

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
| `ratatui` | `0.30.0-alpha.5` | Legacy Ratatui TUI fallback | Retained (`bd-2z0.6.1`) |
| `crossterm` | `0.27` | Terminal raw mode & backend | Active |
| `supports-color` | `3.0.2` | Color level detection | Active |

---

## 3. Rollback & Legacy Compatibility

- The legacy Ratatui adapter in `scriptbots-app/src/terminal/mod.rs` is retained as a compatibility fallback.
- No legacy code is removed without explicit deletion permission.
