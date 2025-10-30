# Legacy Renderer Screenshot Callouts

Baseline imagery captured on 2025-10-30 under Xvfb (`legacy_default.png`, `legacy_selected_agent.png`, `legacy_food_peak.png`). The viewport exported by ImageMagick measures 1570×870 px; the tables below document the regions of interest (ROIs) that should be annotated in downstream documentation or automated visual tests.

## `legacy_default.png`

| Callout | Pixel bounds (x1:x2, y1:y2) | Normalised centre (x,y) | Feature |
| --- | --- | --- | --- |
| A | `799:1569`, `390:869` | `(0.754, 0.724)` | Main world canvas (terrain, agents, food overlay). |
| B | `940:1320`, `440:640` | `(0.719, 0.622)` | Agent cluster spawn area; use to verify crowd density & palette blending. |
| C | `1230:1500`, `520:610` | `(0.865, 0.584)` | Population history overlay lines (green herbivore, red carnivore). |
| D | `1320:1500`, `770:830` | `(0.900, 0.921)` | Instruction strings (“Press d for extra speed”, etc.). |

## `legacy_selected_agent.png`

| Callout | Pixel bounds | Normalised centre | Feature |
| --- | --- | --- | --- |
| E | `820:1180`, `410:660` | `(0.638, 0.614)` | Selected agent body + selection halo and donation ring. |
| F | `860:1400`, `430:520` | `(0.720, 0.546)` | Neural I/O strip (grey quads) displayed when agent selected. |
| G | `1120:1500`, `580:660` | `(0.825, 0.662)` | Brain activation grid (8 px cells); verify grayscale gradient continuity. |

## `legacy_food_peak.png`

| Callout | Pixel bounds | Normalised centre | Feature |
| --- | --- | --- | --- |
| H | `799:1569`, `390:869` | `(0.754, 0.724)` | Food overlay maxima (tiles approach `#C8F6CC`); compare with baseline to ensure gradient intensity. |
| I | `950:1250`, `470:600` | `(0.727, 0.615)` | Localised high-food region; use histogram sampling when validating ported shaders. |

When creating visual overlays, draw semi-transparent rectangles for these ROIs, label them according to the callout letter, and cross-reference `legacy_renderer_spec.md` for the expected colours and interaction semantics. Normalised centres are provided to help drive automated annotation tooling or ensure adaptability to minor viewport adjustments (e.g., if future builds capture the full 1600×900 interior).
