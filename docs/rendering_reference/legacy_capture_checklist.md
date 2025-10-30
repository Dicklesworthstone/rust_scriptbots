# Legacy Renderer Capture Checklist

Purpose: ensure consistent screenshot/metric capture from the original GLUT ScriptBots build before validating the Rust GPUI renderer.

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

### Automated headless capture (used 2025-10-30)

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

This script reproduces the manual protocol under Xvfb, yielding the PNGs and allowing checksum updates.

## Keyboard/Interaction Validation

- Verify controls listed in spec (reset, pause, closed world toggle) still behave as documented. Update spec if deviations observed.
- Confirm follow modes `[s]` and `[o]` recentre camera correctly.

## Data Handoff

- Store PNGs in `docs/rendering_reference/`.
- Append metrics and capture date to `legacy_renderer_spec.md` (World Geometry section).
- Notify Camera/Visuals leads once assets committed so they can reference palette samples.

## Open Items

- Consider automating capture via GLUT script hooks; manual path above is acceptable for initial baseline.
