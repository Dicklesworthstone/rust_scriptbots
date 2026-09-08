# Cinematic Evolution Lab: Design & Feature Preservation Ledger

## Overview

The Cinematic Evolution Lab transforms ScriptBots from a simple simulation viewer into a game-grade, high-fidelity 3D and truecolor TUI simulation.

## Quality Tiers & Performance Matrix

### Look-development capture work (bd-2z0.14.1.20)

The first implementation slice adds independently rendered camera bookmarks at
the same science tick. `captures[].camera_key` indexes `camera[]`; omitting it
retains the latest-keyframe behavior. This lets near, mid-distance and overview
images depict the same world without stepping science between views. The scene
log retains the complete requested manifest, including camera and render inputs.
The initial candidate is `crates/scriptbots-app/tests/scenes/ecosystem_look.toml`:
near, mid and overview views with fixed ACES exposure of -0.35 stops. Compare
against the same manifest with exposure bias 0.0; this is a tuning candidate,
not a measured visual improvement yet.

The offscreen camera now consumes `render.tonemap_mode` and
`render.tonemap_exposure_bias` through the native renderer's conversion and resets
both between sessions. Automatic exposure is explicitly refused for deterministic
offscreen captures. This does not establish that every other render setting has
an offscreen consumer. Hardware images, the meadow/shoreline art target, night and
accessibility review, and the complete look-development acceptance remain open.

To define same-tick views in a scene manifest, use separate camera entries and
captures (positions use Bevy's X-Z ground plane and +Y up):

```toml
[[camera]]
tick = 0
pos = [0.0, 450.0, 450.0]
yaw = 3.1415927
pitch = -0.7853982
fov = 55.0

[[camera]]
tick = 0
pos = [0.0, 900.0, 900.0]
yaw = 3.1415927
pitch = -0.7853982
fov = 55.0

[[captures]]
tick = 0
name = "near"
camera_key = 0

[[captures]]
tick = 0
name = "overview"
camera_key = 1
```

Use the existing `--dump-scene-png SCENE.toml` command with `bevy_render`, inside
a pinned DSR profile. Inspect each frame's nonempty world digest, tick, adapter,
and pixels. Distinct labels alone are not evidence of distinct views. Baseline
and proposed exposure runs must retain identical scientific inputs; their PNGs
are candidates for review, not automatically approved goldens.

### Target matrix

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
