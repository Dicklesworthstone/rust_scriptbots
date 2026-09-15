# Visual Art Direction: Bioluminescent Dark-Field Specification (v1)

> Canonical visual guidelines, palette authority, figure/ground architecture, and renderer consumption boundaries for ScriptBots.

---

## 1. Single Numeric Authority & Core Rule

All visual semantics in ScriptBots are governed by a single numeric authority: [`scriptbots_core::visual::BIOLUMINESCENT_DARK_FIELD_V1`](../crates/scriptbots-core/src/visual.rs).
No renderer, shader, frontend layer, or documentation artifact may author competing or divergent palette literals. Frontends may project or rasterize differently according to hardware capability, but they must **never decide differently**.

- **Numeric Source of Truth:** [`crates/scriptbots-core/src/visual.rs`](../crates/scriptbots-core/src/visual.rs) (`VisualStyleV1`)
- **Accessibility Boundary:** Natural semantic colors are resolved first; [`apply_accessibility_palette`](../crates/scriptbots-core/src/visual.rs) is applied at the final presentation boundary.
- **Reference Image:** [`docs/rendering_reference/bioluminescent_dark_field_v1.png`](rendering_reference/bioluminescent_dark_field_v1.png)
- **Provenance Manifest:** [`docs/rendering_reference/bioluminescent_dark_field_v1.provenance.json`](rendering_reference/bioluminescent_dark_field_v1.provenance.json)

---

## 2. Figure/Ground Rationale: Dark-Field Microscopy

ScriptBots adopts a **dark-field microscopy** visual metaphor. In dark-field microscopy, unstained specimens are illuminated against an unlit field, causing living organisms and active materials to glow with intrinsic optical luminance while the surrounding medium remains dark.

- **Figure/Ground Separation:** World substrate and terrain occupy low albedos and subdued luminances within a unified blue-violet tonal range. They provide physical and topological orientation without competing for visual salience.
- **Luminance Carrier:** Biological entities (agents, living cells, food motes) and dynamic metabolic transitions (combat strikes, birth blooms, death embers, boost trails) carry the scene's primary HDR emissive energy.
- **Cognitive Legibility:** High-density multi-agent simulations quickly produce visual noise under bright, saturated land covers. A dark-field environment ensures that motion vectors, speciation clusters, predator-prey dynamics, and metabolic exchanges remain instantly legible even when zoomed out to several thousand entities.

---

## 3. Visual Vocabulary

The visual system is partitioned into six distinct vocabulary domains:

### 3.1 Substrate & Terrain
The simulation environment is grounded in a deep abyss substrate (`SubstrateStyle`). The six canonical terrain biomes (`TerrainMaterialStyle`) share a coherent blue-violet chromatic identity, differentiating biomes by roughness, specular reflectance, normal strength, and value rather than competing rainbow hues:
1. **Deep Water / Ocean Basin:** Low albedo, high specular reflectance, moderate roughness; establishes oceanic basins.
2. **Shallow Water / Coastal Shelf:** Elevated reflectance with subtle normal distortion; delineates shoreline transitions.
3. **Sand / Beach:** Low specular reflectance, elevated roughness; buffers aquatic and terrestrial boundaries.
4. **Grass / Lowland Plains:** Balanced roughness and reflectance; primary zone for photosynthetic food mote dispersion.
5. **Forest / Canopy:** Darkened, textured surface with mild emissive response representing dense biomass.
6. **Rock / Alpine Ridge:** Highest perceptual roughness, zero emissive gain, maximum normal strength; defines impassable or rugged terrain.

### 3.2 Agent Morphology & Diet Continuum
Agents express genetic traits and metabolic state through structural morphology and controlled emission:
- **Diet Spectrum:** Herbivore tendency spans continuously from electric cyan (`herbivore_srgb`) to hot carnivore magenta (`carnivore_srgb`). Omnivores express intermediate chromatic mixtures.
- **Vitality Attenuation:** Health scales luminance down to `health_luminance_floor`, while age modulates saturation toward `age_luminance_floor`.
- **Anatomical Ornaments:** Spikes extend with high-contrast core coloring (`spike_srgb`) and high emissive gain; wheels reflect locomotion speed on `wheel_srgb`; acoustic ears display `ear_srgb`; eye sclera and pupils provide orientation.
- **Selection & Interaction:** Hovered and focused agents receive boosted emissive multipliers and a dedicated selection rim (`selection_rim_srgb`).

### 3.3 Food Motes
Food exists as bioluminescent phytoplankton motes with two-tone optical depth:
- **Core:** Highly saturated, bright green-cyan mote center (`core_srgb`).
- **Halo:** Softer surrounding halo (`halo_srgb`).
- **Dynamic Growth:** Both radius and HDR emissive gain expand non-linearly with cellular food density, producing distinct luminous motes in rich feeding grounds.

### 3.4 World Events
Dynamic events emit transient two-tone HDR cues with deterministic lifetimes:
- **Combat:** Sharp impact flash mixing hot combat core with warning amber.
- **Birth:** Expansive cool cyan and soft violet bloom.
- **Death:** Warm fading ember decaying rapidly into the dark substrate.
- **Eat:** Subtle fleck adopting food core and halo tones.
- **Reproduction:** Pulsing lavender-cyan wave.
- **Boost:** Cool cyan exhaust trail reading strictly as propulsion, never confused with combat.

### 3.5 Application Chrome & Diagnostics
User interface chrome tokens (`InterfaceStyle`) mirror the dark-field palette with deep surface panels (`surface_srgb`), elevated borders, high-contrast readable text, and standardized telemetry accents (cyan data, magenta alerts, warning amber).

---

## 4. Accessibility & Color Spaces

Accessibility transforms must never compromise scientific simulation fidelity or destroy physical lighting responses:
- **Chroma vs. Luminance Separation:** Emissive gain and HDR bloom thresholds exist as independent scalar multipliers over bounded sRGB color triplets.
- **Final-Boundary Application:** Rendering pipelines composite scenes in linear space and apply [`apply_accessibility_palette`](../crates/scriptbots-core/src/visual.rs) at the presentation boundary. Color vision deficiency filters (Deuteranopia, Protanopia, Tritanopia, High Contrast) remap chromatic axes without altering relative luminance hierarchies or clamping bloom thresholds.

---

## 5. Renderer-Consumption Boundaries & Open Owners

ScriptBots supports multiple rendering backends. Each backend maps the canonical visual model to its specific rendering technology:
- **`scriptbots-world-gfx` (wgpu offscreen / native):** The primary low-level GPU rasterizer. Compiles WGSL shaders directly incorporating `BIOLUMINESCENT_DARK_FIELD_V1` constants, performing instanced quad rendering and post-processing tonemapping/bloom.
- **`scriptbots-bevy` (Bevy 3D ECS):** Translates `MaterialStyle` parameters into Bevy PBR standard materials (`StandardMaterial`) with physical lighting, orbit camera controls, and mesh-based entity representations.
- **`scriptbots-render` (GPUI compositor):** Integrates offscreen GPU readback buffers with GPUI desktop window composition, HUD inspection overlays, and spatial audio hooks.
- **`scriptbots-app` (FrankenTUI):** Projects dark-field semantics into 24-bit ANSI truecolor terminal half-blocks, braille cells, and telemetry widgets.

### Cross-Links to Active Renderer Subsystems
Backend parity and specific subsystem implementations are tracked by dedicated beads:
- **Bevy Material & Terrain Parity:** [`bd-2z0.7.3`](../.beads/issues.jsonl) (agent entities) and [`bd-2z0.7.4`](../.beads/issues.jsonl) (terrain/hydrology).
- **Post-Processing & Bloom v2:** [`bd-2z0.14.1.6`](../.beads/issues.jsonl) (emissive-driven bloom, tonemapping, vignette).
- **Agent Ornament Shader Routing:** [`bd-rl1h`](../.beads/issues.jsonl) (core-to-shader ornament palette) and [`bd-sqji`](../.beads/issues.jsonl) (ear/eye authority).
- **FrankenTUI Dark-Field Styling:** [`bd-2z0.6`](../.beads/issues.jsonl) and [`bd-2z0.14.2`](../.beads/issues.jsonl) (terminal visual polish).

---

## 6. Canonical Reference Capture & Human Review

The canonical reference capture [`bioluminescent_dark_field_v1.png`](rendering_reference/bioluminescent_dark_field_v1.png) was produced by the production `scriptbots-world-gfx::WorldRenderer` pipeline using deterministic fixture parameters (1600x900 viewport, tick 120, seed 424242).

### Human Review Notes
- **Figure/Ground Separation:** Verified. The deep abyss substrate provides a calm, dark backdrop; terrain tiles form subtle structural gradients without overpowering foreground actors.
- **Agent Salience & Diet Contrast:** Verified. Herbivore cyan and carnivore magenta are immediately distinct. Spikes, wheels, and selection rims remain sharply defined against both water and terrestrial substrates.
- **Water vs. Land Biomes:** Verified. Deep water and coastal shelves show distinct reflectivity and darker value profiles compared to vegetative plains, sandy shores, and alpine rock.
- **Food & Event Coherence:** Verified. Phytoplankton food clusters show clear core/halo definition.
- **HUD Exclusion:** Verified. The reference capture strictly represents the simulation world pass; window chrome, diagnostics panels, and HUD elements are excluded from the world texture.
- **Limitations:** This single capture provides definitive visual evidence of style contract adherence and color relationships; it does not claim bit-exact cross-backend parity across disparate rendering engines (e.g. wgpu vs Bevy PBR).
