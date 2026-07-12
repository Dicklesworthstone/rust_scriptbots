# Legacy Renderer Capture Checklist

Purpose: preserve the reproduction procedure and historical record for screenshots from the original GLUT ScriptBots build.

## Repository status

The five PNGs reported as captured on 2025-10-30 are not present in the current
repository or its reachable Git history. They are therefore unavailable
historical evidence, not active visual-test fixtures. Their reported hashes are
preserved here so the record is not lost, but they are intentionally absent from
the active `checksums.txt` manifest until someone explicitly regenerates,
reviews, and commits the corresponding files.

| Unavailable historical file | Reported SHA-256 |
| --- | --- |
| `legacy_default.png` | `8e9e407ad0c9ebef870b879eb0fa92c30fa76034301c2804937f644cb7e30c84` |
| `legacy_selected_agent.png` | `3a5275239e392e518e469c9f07410924b316273b2f695c306c7ffa52dc63e922` |
| `legacy_food_peak.png` | `3f355f970c8712720e3ff58ec74a5d8c7a1a30eff36bb9078962fb121fa671d2` |
| `legacy_food_off.png` | `a9446fd2cd405b60eb7782fcb098f082f974a06d179330e9d325fd014571f7a2` |
| `legacy_zoomed_hud.png` | `d113b13b59556db4f1e215450297370576a6d45bdc34f764262fb95e81103fa8` |

Regeneration is a deliberate review operation. Do not treat the reported
hashes as proof that an output is correct, and do not generate or bless these
assets in CI.

## Prerequisites

- Host with OpenGL + GLUT runtime (Windows or Linux with `freeglut`).
- `cmake`, `make`, and a C++ toolchain capable of compiling the legacy sources in `original_scriptbots_code_for_reference/`.
- Deterministic seed configuration (optional) to align captures with Rust snapshot harness once implemented.

## Build Steps (Linux/macOS)

1. `cd original_scriptbots_code_for_reference`
2. `cmake -S . -B build`
3. `cmake --build build --config Release`
4. Launch: `./build/scriptbots` (ensure window opens at 1600×900).

## Capture Protocol

1. Let simulation stabilize for ~2 seconds (default NUMBOTS=70).
2. Capture the following 1600×900 PNGs:
   - `legacy_default.png` — initial view, no agent selected.
   - `legacy_selected_agent.png` — click an agent to reveal HUD overlays.
   - `legacy_food_peak.png` — toggle `f` off/on to show food overlay at higher densities.
3. Record camera metrics:
   - `scalemult` (should remain 0.2 unless zoomed).
   - `xtranslate`, `ytranslate` (expect 0 if no pan).
4. For each screenshot, log SHA256 in `docs/rendering_reference/checksums.txt`.
5. Note FPS from window title for baseline performance (record min/avg over 10s).

### Automated headless capture (reported as used 2025-10-30)

Run inside the repo root:

```bash
xvfb-run --server-num=111 -s "-screen 0 1600x900x24" bash -lc '
set -euo pipefail
cd original_scriptbots_code_for_reference/build
./scriptbots > /tmp/scriptbots_legacy.log 2>&1 &
PID=$!
WINDOW=""
for attempt in $(seq 1 20); do
  WINDOW=$(xdotool search --name "ScriptBots" || true)
  [ -n "$WINDOW" ] && break
  sleep 0.5
done
[ -n "$WINDOW" ] || { echo "Window not found" >&2; kill $PID; wait $PID || true; exit 1; }
WINDOW=$(echo "$WINDOW" | head -n 1)
sleep 1
import -window "$WINDOW" docs/rendering_reference/legacy_default.png
xdotool mousemove --window "$WINDOW" 800 450 click 1
sleep 1
import -window "$WINDOW" docs/rendering_reference/legacy_selected_agent.png
xdotool key --window "$WINDOW" f
sleep 1
import -window "$WINDOW" docs/rendering_reference/legacy_food_peak.png
kill $PID
wait $PID || true
'
```

This command is the recorded reproduction recipe. The outputs from the reported
2025 capture are unavailable; a new capture must be reviewed before its files or
hashes become active test data.

## Keyboard/Interaction Validation

- Verify controls listed in spec (reset, pause, closed world toggle) still behave as documented. Update spec if deviations observed.
- Confirm follow modes `[s]` and `[o]` recentre camera correctly.

## Data Handoff

- Store reviewed PNGs in `docs/rendering_reference/`.
- Record metrics and the new capture date in `legacy_renderer_spec.md` (World Geometry section).
- Add hashes to the active manifest only in the same reviewed commit that adds the files.
- Notify Camera/Visuals leads once assets are committed so they can reference palette samples.

## Open Items

- Consider automating capture via GLUT script hooks; manual path above is acceptable for initial baseline.
