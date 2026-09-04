# Cinematic Evolution Lab: Design & Feature Preservation Ledger

## Overview

The Cinematic Evolution Lab transforms ScriptBots from a simple simulation viewer into a game-grade, high-fidelity 3D and truecolor TUI simulation.

## Quality Tiers & Performance Matrix

> **Status: design target — not yet a delivered or gated promise.**
> (Reality check 2026-09-04, bead bd-m02n.) No row of this matrix is yet
> measured, owned, or enforced by a CI/DSR gate. Frame-rate budgets become
> gated only when the visual perf harness (bd-2z0.14.3.5.3) lands; tick-rate
> and snapshot budgets live in the bd-2z0.8.18 harness, which does not
> measure FPS. Until a row carries its own status/owner/gate annotation
> here, treat the numbers below as the aspiration the bd-2z0.14 cinematic
> program is building toward, not a claim about shipped behavior.

| Tier | Frame Budget (1k agents) | Frame Budget (10k agents) | Features Enabled |
|------|-------------------------|--------------------------|------------------|
| Potato | 60 FPS (16.6ms) | 60 FPS (16.6ms) | Flat shading, disabled AA, no shadows |
| Low | 60 FPS (16.6ms) | 45 FPS (22.2ms) | 1-cascade shadows (1024), FXAA, basic bloom |
| Medium | 60 FPS (16.6ms) | 30 FPS (33.3ms) | 2-cascade shadows (2048), FXAA, HDR bloom, DoF |
| High | 60 FPS (16.6ms) | 30 FPS (33.3ms) | 4-cascade shadows (2048), TAA, SSAO, planar reflections |
| Ultra | 60 FPS (16.6ms) | 30 FPS (33.3ms) | 4-cascade shadows (4096), TAA, SSAO, full motion blur |

## Feature Preservation Ledger

| Feature | Legacy Location | Current Location | Preservation Status |
|---------|----------------|------------------|---------------------|
| 5 Accessibility Palettes | `scriptbots-render/src/lib.rs` | `scriptbots-core/src/visual.rs` | Preserved (Natural, Deuteranopia, Protanopia, Tritanopia, HighContrast) |
| Keyboard Remapping | `scriptbots-app/src/control.rs` | `scriptbots-app/src/control.rs` | Preserved (22 CommandActions remappable) |
| Emoji/Narrow/ASCII Vocabularies | `scriptbots-app/src/terminal` | `scriptbots-app/src/terminal` | Preserved (Sub-cell painter fallback tiers) |
| Headless FNV Evidence | `scriptbots-storage` | `scriptbots-storage` | Preserved (Deterministic hash contract) |
| Screenshot Export (PNG/ASCII) | `scriptbots-app` | REST `/api/screenshot` & CLI `--export-screenshot` | Preserved |
| Introspection Brain Views | `scriptbots-core` | `scriptbots-core/src/visual.rs` | Preserved (Demand-driven introspection) |
| Spatial Audio | `scriptbots-render` | `scriptbots-render/src/audio.rs` | Preserved (Kira engine & SPSC channels) |

