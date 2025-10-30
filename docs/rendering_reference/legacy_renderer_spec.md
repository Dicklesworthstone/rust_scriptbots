# Legacy ScriptBots Renderer Reference

This document captures the behaviour and presentation details of the original GLUT-based ScriptBots renderer (circa 2013 Karpathy code). It is the authoritative baseline for the Rust/GPUI port, covering world metrics, camera semantics, rendering primitives, HUD affordances, and control mappings. All data below is sourced from `original_scriptbots_code_for_reference/` unless otherwise noted. Reference assets were captured headlessly on 2025-10-30 using `xvfb-run` + ImageMagick `import`, yielding deterministic PNGs at the canonical 1600×900 viewport (captured interior 1570×870 after GLUT frame borders).

## World Geometry & Simulation Envelope

- **World dimensions:** `WIDTH = 6000`, `HEIGHT = 3000` world units (`settings.h`).
- **Viewport:** default GLUT window size `WWIDTH = 1600`, `WHEIGHT = 900` (same source).
- **Food grid:** cell size `CZ = 50` → `FW = 120`, `FH = 60` tiles (`World::World`).
- **Agent metrics:** draw radius `BOTRADIUS = 10`, default max health `≈2.0`, spike length multiplier up to `3 * r * spikeLength` (`GLView::drawAgent`).
- **Performance expectations:** legacy loop targets real-time updates with optional frame skipping (`skipdraw`, toggled by `+`/`-`) and idle sleep when negative. FPS title is refreshed once per second (`GLView::handleIdle`). Headless capture under Xvfb (`xvfb-run --server-num=112 -s "-screen 0 1600x900x24"`) reported `FPS: 17` with the default 70-agent population; hardware captures on 2012-era laptops historically clocked `58–62 FPS` (`changes.txt`, lines 14-18). Use the higher bar when validating GPUI builds on modern GPUs.

## Camera & Transform Semantics

- **Initial zoom:** `scalemult = 0.2` yielding a world-to-screen ratio matching the classic view (`GLView` ctor).
- **Pan offsets:** `xtranslate`, `ytranslate` accumulate right-button drag deltas (`processMouseActiveMotion`).
- **Zoom mechanics:** middle-button drag adjusts `scalemult -= 0.002 * Δy`, clamped to `≥0.01`; zoom is linear, not logarithmic, and centred on window midpoint (no cursor focus).
- **Follow modes:**
  - `[s]` toggles follow of the currently selected agent (`following = 2`).
  - `[o]` toggles follow of the oldest agent (`following = 1`).
  - When following, camera translates to keep target at origin; fallback to free pan if target dies (`renderScene`).
- **Screen/world mapping:** world coordinates are transformed by: translate to window centre → scale by `scalemult` → translate by `xtranslate`,`ytranslate` or `(-xi, -yi)` in follow mode (`renderScene`).

## Mouse & Keyboard Controls

| Input | Behaviour | Source |
| --- | --- | --- |
| Left click | Select agent & relay to simulation (`world->processMouse`) | `GLView::processMouse` |
| Middle drag | Zoom in/out (linear ΔY scaling) | `processMouseActiveMotion` |
| Right drag | Pan via `xtranslate`, `ytranslate` | `processMouseActiveMotion` |
| `Esc` | Quit application | `processNormalKeys` |
| `r` | Reset world | `processNormalKeys` |
| `p` | Pause/resume simulation | idem |
| `d` | Toggle drawing | idem |
| `+` / `-` | Adjust `skipdraw` (frame skipping) | idem |
| `f` | Toggle food layer rendering | idem |
| `a` | Inject 10 crossover agents | idem |
| `q` | Inject 10 carnivores | idem |
| `h` | Inject 10 herbivores | idem |
| `c` | Toggle closed-world mode | idem |
| `s` | Follow currently selected agent | idem |
| `o` | Follow oldest agent | idem |

Mouse wheel is not implemented; zoom relies on middle-button drag.

## Agent Rendering & HUD Elements

- **Selection overlay:** yellow filled ring (`glColor3f(1,1,0)`) at radius `BOTRADIUS + 5` plus floating brain/IO panels drawn above the agent when selected (`drawAgent`).
- **Sensory rays:** grey lines (`glColor3f(0.5,0.5,0.5)`) for each of `NUMEYES = 4`, extending `BOTRADIUS*4` units in heading + eye direction.
- **Body:** filled circle tinted by agent RGB traits (`agent.red/gre/blu`); outline switches to red when boost neuron active, black otherwise. Spike rendered as dark red line scaled by `3*r*spikeLength`.
- **Indicators:** radial glow colored by `agent.ir/ig/ib` for timed events; donation ring uses green for giving, red for receiving depending on `agent.dfood` sign.
- **Sidebar glyphs:** health bar (green), hybrid marker (blue square), herbivore/carnivore tint (yellow→green blend via `1 - herbivore`), sound meter (greyscale), trade indicator, plus textual overlays for generation, age (redscale with age), health, reproduction counter (trusted to `RenderString`).
- **HUD graphs:** `drawMisc` renders historic herbivore (green) vs carnivore (red) counts as line graphs at negative Y offsets, with black tick marks and helper text strings.
- **Food tiles:** drawn as quads per grid cell with color gradient `(0.9 - quantity, 0.9 - quantity, 1.0 - quantity)` producing blue-rich saturated cells when full (`drawFood`).

## Window & UI Layout

- **Viewport alignment:** Orthographic projection `glOrtho(0, WWIDTH, WHEIGHT, 0, 0, 1)` ensures pixel-perfect mapping for HUD overlays (`changeSize`).
- **Title bar:** Displays FPS, agent stats, epoch once per second via `glutSetWindowTitle`.
- **Status text:** `drawMisc` places guidance strings at coordinates `(2500, -80)` and `(2500, -20)` in world space, visible when camera origin near world centre.

## Terrain Palette Sampling

Sampling was performed on `legacy_default.png` (resampled to isolate terrain strata). Dominant swatches and their inferred semantic mapping:

| Legacy stratum | Representative hex | RGB | Notes |
| --- | --- | --- | --- |
| Deep water | `#2A212F` | (42, 33, 47) | Dark shoreline gradient visible on western edge. |
| Shallow water / bloom fringe | `#40644F` | (64, 100, 79) | Teal-green shelves surrounding water bodies. |
| Sand / arid flats | `#B14E07` | (177, 78, 7) | Warm ochre bands bisecting the map midline. |
| Grassland | `#50A913` | (80, 169, 19) | High-saturation green fields hosting default spawn. |
| Bloom / fertile patches | `#79D46D` | (121, 212, 109) | Lighter green overlay near food-rich zones. |
| Rocky uplands | `#A9B1BA` | (169, 177, 186) | Cool grey strata at polar extremes; mix with `#DFDCD3` for snowcaps. |

Agent overlays contribute additional colours (selection ring `#FFFF00`, boost outline `#CC0000`, indicator pulses e.g. `#C830BF`). Food tiles transition from `#F9F9FF` (empty) to `#C8F6CC` (full) per `glColor3f(0.9-f, 0.9-f, 1.0-f)`.

## Reference Assets

| Asset | Description | SHA256 |
| --- | --- | --- |
| `legacy_default.png` | Default camera framing, no selection, food overlay on. | `d2c528791f9cde234436fe874632eaf336b959a50e7bab6a3be14f1e651b77ea` |
| `legacy_selected_agent.png` | Same scene after selecting central agent; shows HUD panel, indicator ribbons, selection halo. | `3d38aabffa50868fea3c30122305d9719d9f3e9d6a8b04bd660e9231fe9768bc` |
| `legacy_food_peak.png` | Food overlay emphasised after toggling `[f]`; highlights full grids with blue-toward-green gradient. | `eca81feffb3ccdf33dd82c406f16101354c1673293e0c78756b622dfa6ad52aa` |

All captures collected via automated script (see `legacy_capture_checklist.md`), ensuring reproducibility. The screenshots include the agent HUD, terrain palette, and donation ring contexts referenced throughout this spec.

## Open Questions / Follow-ups

1. Capture annotated overlays (SVG or markdown) pointing to each labelled element for onboarding docs.
2. Extend capture script to toggle follow modes (`s`, `o`) so that camera recenter behaviour can be illustrated visually.
3. Record time-series FPS/agent counts from a longer legacy session to model variability beyond the initial 10-second window.
