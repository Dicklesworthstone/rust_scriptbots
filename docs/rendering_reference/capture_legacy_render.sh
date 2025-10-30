#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 output_dir" >&2
  exit 1
fi
OUTDIR=$1
mkdir -p "$OUTDIR"
LOG="$OUTDIR/legacy_capture.log"

run_capture() {
  local label=$1
  shift
  import -display "$DISPLAY" "$@" "$OUTDIR/${label}.png"
}

cd "$(dirname "$0")/../../original_scriptbots_code_for_reference/build"
./scriptbots >"$LOG" 2>&1 &
PID=$!
cleanup() {
  kill $PID 2>/dev/null || true
  wait $PID 2>/dev/null || true
}
trap cleanup EXIT

WIN=""
for attempt in {1..20}; do
  WIN=$(xdotool search --name "ScriptBots" | head -n1 || true)
  if [ -z "$WIN" ]; then
    WIN=$(xdotool search --class GLUT | head -n1 || true)
  fi
  if [ -n "$WIN" ]; then
    break
  fi
  sleep 1
done
if [ -z "$WIN" ]; then
  echo "ERR: window not found" >&2
  xdotool search --name "ScriptBots" || true
  exit 1
fi

echo "Window id: $WIN" >>"$LOG"

sleep 2
run_capture default -window root

xdotool windowactivate "$WIN"
sleep 0.5
xdotool mousemove --window "$WIN" 800 450
xdotool click --window "$WIN" 1
sleep 1
run_capture selected -window root

xdotool key --window "$WIN" f
sleep 1
run_capture food_off -window root
xdotool key --window "$WIN" f
sleep 1
run_capture food_on -window root

xdotool mousedown --window "$WIN" 2
xdotool mousemove_relative --window "$WIN" 0 120
xdotool mouseup --window "$WIN" 2
sleep 1
run_capture zoomed -window root

exit 0
