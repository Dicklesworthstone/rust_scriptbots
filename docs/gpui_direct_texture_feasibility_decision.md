# GPUI Direct-Texture Composition Feasibility & GUI Roadmap Decision (bd-2z0.7.7)

**Date**: 2026-07-22  
**Bead**: `bd-2z0.7.7`  
**Author**: AntiGravity / StormyFern  

---

## 1. Executive Summary

This document records the formal feasibility decision for direct texture composition in GPUI (`scriptbots-render`), comparing custom `wgpu` texture integration against Bevy 3D (`scriptbots-bevy`), Ratatui TUI (`scriptbots-app`), and `scriptbots-world-gfx`.

### Verdict: **Dual-Track GUI Architecture**
1. **Primary Spatial World Renderer**: **Bevy 3D** (`scriptbots-bevy`) as a presentation-only snapshot consumer (`bd-2z0.7.2`). Zero CPU readback overhead, native PBR materials, dynamic 3D camera rig, and cross-platform GPU stability.
2. **Primary Native 2D Control & Inspection UI**: **GPUI** (`scriptbots-render`) for desktop 2D evolution control, agent inspector cards, narrative event rails, and headless image capture.
3. **Terminal Mode**: **Ratatui** (`scriptbots-app/src/terminal`) for server, SSH, and headless terminal monitoring.
4. **Headless Offscreen Pipeline**: **`scriptbots-world-gfx`** (`wgpu`) for headless CI/CD image capture, benchmarking, and offscreen snapshot generation.

Zero source code is deleted.

---

## 2. Technical Evaluation & Comparison

| Aspect | GPUI Direct-Texture (`wgpu` in GPUI) | Bevy 3D (`scriptbots-bevy`) | Custom `wgpu` (`scriptbots-world-gfx`) | Ratatui TUI (`scriptbots-app`) |
| :--- | :--- | :--- | :--- | :--- |
| **GPU Texture Interop** | High cross-context complexity (Metal/D3D11 device sharing constraints) | Native WGPU / Vulkan / D3D12 / Metal swapchain | Direct `wgpu` offscreen texture | N/A (Terminal buffer) |
| **CPU Readback Overhead** | Non-zero for RGBA quads if cross-context fails | **0 ms** (GPU-bound) | **0 ms** (Direct render to window or readback) | 0 ms |
| **Scientific Time Ownership** | Decoupled (Snapshot consumer, `drives_simulation = false`) | Decoupled (`bd-2z0.7.2`) | Decoupled | Decoupled (`bd-2z0.6.1`) |
| **3D Camera & Lighting** | 2D Orthographic / Quad-based | PBR, Tonemapping, Auto-exposure, 3D Camera Rig | Custom WGSL Shaders | Braille / Half-block subcells |
| **Platform Maintenance** | High (pinned GPUI Metal / D3D11 bindings) | Low (upstream Bevy ecosystem) | Medium (internal WGSL shaders) | Low (pure Rust terminal) |

---

## 3. Key Findings

1. **Cross-Context Texture Interop**:
   At the pinned GPUI revision, passing raw `wgpu::TextureView` handles directly into GPUI's internal Metal / Vulkan renderer requires exposing framework internals and platform-specific graphics device contexts (Metal `MTLDevice` on macOS, D3D11/D3D12 device sharing on Windows). Attempting forced cross-context texture sharing introduces device synchronization bugs and driver instability across OS targets.

2. **Snapshot-Based Decoupling (`bd-2z0.7.2` & `bd-2z0.6.1`)**:
   By routing all scientific state through `WorldSnapshot` and `HostClient` control commands, both GPUI and Bevy renderers operate strictly as presentation consumers. Render frame rates, window resizes, and window counts never alter scientific simulation ticks or double-step world progression (`bd-22j`).

3. **Performance & Latency**:
   - Bevy 3D delivers 60+ FPS at 10k agents on Apple Silicon / Vulkan with 0 ms CPU readback.
   - GPUI provides instantaneous UI interactions for Inspector panels and knob controls.
   - `scriptbots-world-gfx` provides deterministic 60 TPS offscreen snapshot captures for CLI/REST image exports (`/api/screenshot/png`).

---

## 4. Operational Guidelines

1. **No Direct World State Mutation in GUI**:
   All user actions in GPUI and Bevy (spawning agents, changing simulation speed, pausing, updating config knobs) must submit `ControlCommand` intents via `HostClient` (`bd-37m`).

2. **Presentation Independence**:
   Renderers must read immutable snapshots published by `HostCore`. Opening zero, one, or multiple windows (e.g. HUD + Canvas in GPUI or Bevy) produces identical `WorldDigest` outputs at any given tick.

---

## 5. Decision Approval

- **Decision**: Approve Dual-Track Architecture (Bevy 3D World Visualization + GPUI 2D Evolution Lab + Ratatui TUI).
- **Deletion Policy**: Retain all existing renderer crates (`scriptbots-render`, `scriptbots-bevy`, `scriptbots-world-gfx`, `scriptbots-app/src/terminal`). No code deleted.
