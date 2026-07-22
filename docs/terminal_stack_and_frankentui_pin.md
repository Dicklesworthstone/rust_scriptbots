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
- **Pinned Revision**: `15cc65438a2095fbe8dd0dfce9adcfc7edab7612` (includes upstream lifecycle/simulator model completion fix `15cc6543`)
- **Default Features**: `false`
- **Enabled Features**: `["crossterm"]`

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
