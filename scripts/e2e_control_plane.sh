#!/usr/bin/env bash
# e2e_control_plane.sh — wire-level REST+MCP control-plane acceptance probe (bd-6mus).
#
# Boots a real scriptbots-app in long-lived server mode (the only non-TTY
# frontend), then asserts the control surface a remote operator actually
# depends on: REST reads, the two-axis command envelope reaching terminal
# application state, playback semantics (pause freezes, step advances exactly
# one tick, resume unfreezes), SSE streaming, and the MCP JSON-RPC path
# (initialize -> tools/list -> tools/call) over POST /mcp.
#
# Born from the 2026-09-03 reality check, which found MCP /mcp returning
# -32601 "unknown method" for every JSON-RPC method and REST playback
# commands stuck admitted-forever (the bd-w1oi storage latch). This probe
# exists so neither defect can silently return.
#
# Environment overrides:
#   APP_BIN    binary to launch   (default: target/debug/scriptbots-app)
#   REST_ADDR  REST bind address  (default: 127.0.0.1:8188)
#   MCP_ADDR   MCP bind address   (default: 127.0.0.1:8190)
#   TICK_BUDGET_S  seconds to wait for one tick advance (default: 60)
#
# Exit 0 only when every assertion passes. App log tail is printed on failure.

set -u
cd "$(dirname "$0")/.." || exit 1

APP_BIN="${APP_BIN:-target/debug/scriptbots-app}"
REST_ADDR="${REST_ADDR:-127.0.0.1:8188}"
MCP_ADDR="${MCP_ADDR:-127.0.0.1:8190}"
TICK_BUDGET_S="${TICK_BUDGET_S:-60}"
REST="http://$REST_ADDR"
MCP="http://$MCP_ADDR"

PASS=0
FAIL=0
WORKDIR="$(mktemp -d)"
APP_LOG="$WORKDIR/app.log"
trap 'kill "$APP_PID" 2>/dev/null; wait "$APP_PID" 2>/dev/null' EXIT

ok()   { PASS=$((PASS + 1)); printf 'ok   %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$1"; }
check(){ if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (expected [$3] got [$2])"; fi; }
check_ge(){ if [ "$2" -ge "$3" ] 2>/dev/null; then ok "$1"; else bad "$1 (expected >= $3 got [$2])"; fi; }

http() { curl -s --max-time 20 "$@"; }
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$@"; }

tick_now() { http "$REST/api/status" | jq -r '.tick' 2>/dev/null; }

# poll_json URL JQ_EXPR [timeout_s]: poll until jq output is non-empty/non-null.
poll_json() {
    local url="$1" jq_expr="$2" deadline=$((SECONDS + ${3:-60})) out
    while [ "$SECONDS" -lt "$deadline" ]; do
        out="$(http "$url" | jq -r "$jq_expr" 2>/dev/null)"
        [ -n "$out" ] && [ "$out" != "null" ] && { printf '%s' "$out"; return 0; }
        sleep 0.5
    done
    return 1
}

# wait_command_terminal ID [timeout_s]: poll /api/control/status/ID until its
# application_state leaves "admitted"/"pending" (the admitted-forever stall
# this guard exists to catch). Prints the final state.
wait_command_terminal() {
    local id="$1" deadline=$((SECONDS + ${2:-60})) state
    while [ "$SECONDS" -lt "$deadline" ]; do
        state="$(http "$REST/api/control/status/$id" | jq -r '.application_state' 2>/dev/null)"
        case "$state" in
            applied|rejected|failed) printf '%s' "$state"; return 0 ;;
        esac
        sleep 0.5
    done
    printf '%s' "${state:-<no-response>}"
    return 1
}
# ---------------------------------------------------------------- launch ---
[ -x "$APP_BIN" ] || { echo "FATAL: $APP_BIN is not executable (set APP_BIN)"; exit 2; }
SCRIPTBOTS_CONTROL_REST_ENABLED=1 \
SCRIPTBOTS_CONTROL_REST_ADDR="$REST_ADDR" \
SCRIPTBOTS_CONTROL_MCP=http \
SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR="$MCP_ADDR" \
RUST_LOG="${RUST_LOG:-info,fsqlite=off,fsqlite_mvcc=off,fsqlite_vdbe=off,fsqlite_core=off,fsqlite_planner=off}" \
    "$APP_BIN" --mode server --storage memory --threads 2 >"$APP_LOG" 2>&1 &
APP_PID=$!

poll_json "$REST/api/status" '.tick' 90 >/dev/null || {
    bad "app reachable on REST $REST (see log tail below)"; tail -40 "$APP_LOG"; exit 1; }
ok "app reachable on REST $REST"
poll_json "$MCP/health" '.status' 30 >/dev/null && ok "MCP health on $MCP" \
    || bad "MCP health on $MCP"

# ------------------------------------------------------------ REST reads ---
PATHS="$(http "$REST/api-docs/openapi.json" | jq '.paths | length' 2>/dev/null)"
check_ge "openapi publishes >=29 paths" "${PATHS:-0}" 29

KNOBS="$(http "$REST/api/knobs" | jq 'length' 2>/dev/null)"
check_ge "knob roster publishes >=100 knobs" "${KNOBS:-0}" 100

AGENTS="$(poll_json "$REST/api/status" '.agent_count' 30)"
check_ge "world reports a live founding population" "${AGENTS:-0}" 1

# ------------------------------------------- two-axis playback semantics ---
# Pause: the tick loop observes the pause at the NEXT boundary, so judge the
# freeze by comparing two post-pause samples, not the pre-pause tick.
PAUSE_ID="$(http -X POST "$REST/api/control/pause" | jq -r '.command_id')"
PAUSE_STATE="$(wait_command_terminal "$PAUSE_ID")"
check "control/pause reaches a terminal application state" "$PAUSE_STATE" "applied"
sleep 2
F1="$(tick_now)"
sleep 2
F2="$(tick_now)"
check "pause freezes the world" "$F2" "$F1"

# Single step: exactly one tick, still paused afterwards.
STEP_ID="$(http -X POST "$REST/api/control/step" -H 'content-type: application/json' -d '{"count":1}' | jq -r '.command_id')"
STEP_STATE="$(wait_command_terminal "$STEP_ID")"
check "control/step reaches a terminal application state" "$STEP_STATE" "applied"
sleep 2
T_STEP="$(tick_now)"
check "control/step advances exactly one tick" "$T_STEP" "$((F2 + 1))"
sleep 2
check "stepped world stays paused" "$(tick_now)" "$T_STEP"

# Resume must unfreeze: tick advances beyond the frozen boundary.
RESUME_ID="$(http -X POST "$REST/api/control/resume" | jq -r '.command_id')"
RESUME_STATE="$(wait_command_terminal "$RESUME_ID")"
check "control/resume reaches a terminal application state" "$RESUME_STATE" "applied"
poll_json "$REST/api/status" "select(.tick > $T_STEP) | .tick" "$TICK_BUDGET_S" >/dev/null \
    && ok "resume unfreezes the world (tick advanced past $T_STEP)" \
    || bad "resume unfreezes the world (tick stuck at/under $T_STEP within ${TICK_BUDGET_S}s)"

# Negative paths: malformed bodies are rejected without killing the server.
BAD_CODE="$(code -X POST "$REST/api/control/step" -H 'content-type: application/json' -d 'not-json')"
case "$BAD_CODE" in 4*) ok "malformed control/step rejected with HTTP $BAD_CODE";; *) bad "malformed control/step rejected (got HTTP $BAD_CODE)";; esac
UNKNOWN_CODE="$(code "$REST/api/control/status/bd-does-not-exist")"
case "$UNKNOWN_CODE" in 4*) ok "unknown command id answered with typed HTTP $UNKNOWN_CODE";; *) bad "unknown command id typed refusal (got HTTP $UNKNOWN_CODE)";; esac

# ------------------------------------------------------------------- SSE ---
SSE_COUNT="$(curl -s -N --max-time 10 -H 'Accept: text/event-stream' "$REST/api/ticks/stream" \
    | grep -c '^data:' 2>/dev/null)"
check_ge "SSE tick stream yields >=3 summaries in 10s" "${SSE_COUNT:-0}" 3

# ------------------------------------------------------------------- MCP ---
INIT_RESPONSE="$(http -X POST "$MCP/mcp" -H 'content-type: application/json' -H 'accept: application/json, text/event-stream' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"e2e-control-plane","version":"0"}}}')"
check "MCP initialize succeeds (not -32601)" \
    "$(printf '%s' "$INIT_RESPONSE" | jq -r '.error.code // "ok"' 2>/dev/null)" "ok"
check "MCP initialize negotiates a protocol version" \
    "$(printf '%s' "$INIT_RESPONSE" | jq -r '.result.protocolVersion // empty' 2>/dev/null | cut -c1-4)" "2025"

TOOLS="$(http -X POST "$MCP/mcp" -H 'content-type: application/json' -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
    | jq -r '.result.tools[].name' 2>/dev/null | sort | tr '\n' ' ')"
EXPECTED_TOOLS="apply_patch apply_updates apply_preset get_command_status get_config get_status list_knobs list_presets pause resume set_speed shutdown step "
check "MCP tools/list returns the full 13-tool roster" "$TOOLS" "$EXPECTED_TOOLS"

STATUS_CALL="$(http -X POST "$MCP/mcp" -H 'content-type: application/json' \
    -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_status","arguments":{}}}' \
    | jq -r '.result.content[0].text | fromjson | .tick' 2>/dev/null)"
check_ge "MCP tools/call get_status returns a live tick" "${STATUS_CALL:--1}" 0

UNKNOWN_TOOL="$(http -X POST "$MCP/mcp" -H 'content-type: application/json' \
    -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"no_such_tool","arguments":{}}}' \
    | jq -r '.error.code // "no-error"' 2>/dev/null)"
case "$UNKNOWN_TOOL" in no-error|null) bad "unknown MCP tool yields a JSON-RPC error";; *) ok "unknown MCP tool yields a JSON-RPC error ($UNKNOWN_TOOL)";; esac

# ---------------------------------------------------------------- report ---
printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then
    echo "--- app log tail ($APP_LOG) ---"
    tail -40 "$APP_LOG"
    exit 1
fi
exit 0
