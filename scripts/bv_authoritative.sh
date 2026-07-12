#!/usr/bin/env bash
# Run data-bearing BV robot commands against Beads' tracked authoritative export.

set -euo pipefail

fail() {
  printf 'bv-authoritative: %s\n' "$1" >&2
  exit "${2:-65}"
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
authoritative="$repo_root/.beads/issues.jsonl"

for tool in br bv git jq mktemp mkdir ln; do
  command -v "$tool" >/dev/null 2>&1 || fail "required tool is unavailable: $tool" 69
done

[[ -f "$authoritative" ]] || fail "missing authoritative tracker export: $authoritative"
[[ -s "$authoritative" ]] || fail "authoritative tracker export is empty: $authoritative"
(( $# > 0 )) || fail "usage: scripts/bv_authoritative.sh --robot-<command> [robot options]" 64

primary_command=""
primary_count=0
robot_next_seen=0
robot_plan_seen=0
robot_triage_seen=0
triage_grouping_seen=0
label_seen=0
attention_limit_seen=0
graph_format_seen=0

arguments=("$@")
argument_index=0
while (( argument_index < ${#arguments[@]} )); do
  argument="${arguments[argument_index]}"
  case "$argument" in
    --robot-triage-by-track | --robot-triage-by-label)
      triage_grouping_seen=1
      ;;
    --robot-triage | --robot-next | --robot-plan | --robot-priority | --robot-insights | --robot-label-health | --robot-label-flow | --robot-label-attention | --robot-alerts | --robot-suggest | --robot-graph)
      primary_count=$((primary_count + 1))
      primary_command="$argument"
      ;;
    --robot-forecast | --robot-burndown)
      primary_count=$((primary_count + 1))
      primary_command="$argument"
      argument_index=$((argument_index + 1))
      (( argument_index < ${#arguments[@]} )) || fail "$argument requires a value" 64
      [[ "${arguments[argument_index]}" != --* ]] || fail "$argument requires a value" 64
      ;;
    --robot-forecast=* | --robot-burndown=*)
      [[ -n "${argument#*=}" ]] || fail "robot command requires a value: $argument" 64
      primary_count=$((primary_count + 1))
      primary_command="${argument%%=*}"
      ;;
    --label | --recipe | --attention-limit)
      argument_index=$((argument_index + 1))
      (( argument_index < ${#arguments[@]} )) || fail "$argument requires a value" 64
      [[ "${arguments[argument_index]}" != --* ]] || fail "$argument requires a value" 64
      case "$argument" in
        --label) label_seen=1 ;;
        --attention-limit) attention_limit_seen=1 ;;
      esac
      ;;
    --label=* | --recipe=* | --attention-limit=*)
      [[ -n "${argument#*=}" ]] || fail "option requires a value: $argument" 64
      case "$argument" in
        --label=*) label_seen=1 ;;
        --attention-limit=*) attention_limit_seen=1 ;;
      esac
      ;;
    --graph-format)
      argument_index=$((argument_index + 1))
      (( argument_index < ${#arguments[@]} )) || fail "--graph-format requires json" 64
      [[ "${arguments[argument_index]}" == json ]] || fail "only JSON robot output is supported; use --graph-format=json" 64
      graph_format_seen=1
      ;;
    --graph-format=json)
      graph_format_seen=1
      ;;
    *)
      fail "unsupported or potentially mutating BV option: $argument" 64
      ;;
  esac
  argument_index=$((argument_index + 1))
done

(( primary_count == 1 )) || fail "exactly one supported read-only --robot-* command is required" 64
(( triage_grouping_seen == 0 )) || [[ "$primary_command" == --robot-triage ]] ||
  fail "triage grouping requires --robot-triage" 64
(( label_seen == 0 )) || [[ "$primary_command" == --robot-insights || "$primary_command" == --robot-plan || "$primary_command" == --robot-priority ]] ||
  fail "--label is supported only for insights, plan, or priority" 64
(( attention_limit_seen == 0 )) || [[ "$primary_command" == --robot-label-attention ]] ||
  fail "--attention-limit requires --robot-label-attention" 64
(( graph_format_seen == 0 )) || [[ "$primary_command" == --robot-graph ]] ||
  fail "--graph-format requires --robot-graph" 64

[[ "$primary_command" == --robot-next ]] && robot_next_seen=1
[[ "$primary_command" == --robot-plan ]] && robot_plan_seen=1
[[ "$primary_command" == --robot-triage ]] && robot_triage_seen=1

source_hash_before="$(git hash-object "$authoritative")"
view_root="$(mktemp -d "${TMPDIR:-/tmp}/scriptbots-bv-authoritative.XXXXXXXX")"
mkdir "$view_root/.beads"
ln -s "$authoritative" "$view_root/.beads/beads.jsonl"

source_stats="$view_root/source-stats.json"
br_list="$view_root/br-list.json"
br_ready="$view_root/br-ready.json"
verifier="$view_root/bv-triage.json"
planner="$view_root/bv-plan.json"
result="$view_root/bv-result.json"
diagnostics="$view_root/bv-stderr.log"

jq -c -s '
  {
    total: length,
    by_status: (group_by(.status) | map({key: .[0].status, value: length}) | from_entries),
    blocking_edges: ([.[] | .dependencies[]? | select(.type == "blocks")] | length),
    unique_ids: ([.[].id] | unique | length)
  }
' "$authoritative" >"$source_stats"

jq -e '.total > 0 and .total == .unique_ids' "$source_stats" >/dev/null ||
  fail "authoritative export has no issues or contains duplicate issue IDs"

if ! (
  cd "$repo_root"
  env -u BEADS_DB -u BEADS_DIR -u BEADS_JSONL \
    BR_OUTPUT_FORMAT=json br list --json --no-db --all --limit 0 >"$br_list"
); then
  fail "br list could not read the authoritative JSONL export"
fi

if ! (
  cd "$repo_root"
  env -u BEADS_DB -u BEADS_DIR -u BEADS_JSONL \
    BR_OUTPUT_FORMAT=json br ready --json --no-db --limit 0 >"$br_ready" 2>"$view_root/br-ready.log"
); then
  fail "br ready could not read the authoritative JSONL export"
fi

if ! jq -e --slurpfile expected "$source_stats" '
  (length == $expected[0].total)
  and ((group_by(.status) | map({key: .[0].status, value: length}) | from_entries) == $expected[0].by_status)
' "$br_list" >/dev/null; then
  fail "br list counts/statuses disagree with .beads/issues.jsonl"
fi

env -u BEADS_DB -u BEADS_DIR -u BEADS_JSONL \
  BV_OUTPUT_FORMAT=json BV_NO_CACHE=1 \
  bv --no-cache --db "$view_root/.beads" --format json --robot-triage >"$verifier" 2>"$diagnostics" || {
    cat "$diagnostics" >&2
    fail "BV could not analyze the isolated authoritative view"
  }

if ! jq -e --slurpfile expected "$source_stats" --slurpfile ready "$br_ready" '
  (.data_hash | type == "string" and length > 0)
  and (.triage.project_health.counts.total == $expected[0].total)
  and (.triage.project_health.counts.by_status == $expected[0].by_status)
  and (.triage.project_health.graph.node_count == $expected[0].total)
  and (.triage.project_health.graph.edge_count == $expected[0].blocking_edges)
  and (.triage.quick_ref.actionable_count == ($ready[0] | length))
' "$verifier" >/dev/null; then
  fail "BV issue/status/dependency/actionable counts disagree with br and .beads/issues.jsonl"
fi

expected_data_hash="$(jq -r '.data_hash' "$verifier")"

env -u BEADS_DB -u BEADS_DIR -u BEADS_JSONL \
  BV_OUTPUT_FORMAT=json BV_NO_CACHE=1 \
  bv --no-cache --db "$view_root/.beads" --format json --robot-plan >"$planner" 2>"$diagnostics" || {
    cat "$diagnostics" >&2
    fail "BV could not plan the isolated authoritative view"
  }

if ! jq -e --arg expected_hash "$expected_data_hash" --slurpfile ready "$br_ready" '
  ([.plan.tracks[]?.items[]?.id] | sort) as $planned
  | ($ready[0] | map(.id) | sort) as $ready_ids
  | .data_hash == $expected_hash
    and ($planned == ($planned | unique))
    and ($planned == $ready_ids)
    and (.plan.total_actionable == ($ready[0] | length))
' "$planner" >/dev/null; then
  fail "BV actionable issue IDs disagree with br ready; BR remains the claim authority"
fi

set +e
env -u BEADS_DB -u BEADS_DIR -u BEADS_JSONL \
  BV_OUTPUT_FORMAT=json BV_NO_CACHE=1 \
  bv --no-cache --db "$view_root/.beads" --format json "$@" >"$result" 2>"$diagnostics"
bv_status=$?
set -e
cat "$diagnostics" >&2
(( bv_status == 0 )) || exit "$bv_status"

jq -e . "$result" >/dev/null || fail "BV robot command did not produce valid JSON"
actual_data_hash="$(jq -r '.data_hash // empty' "$result")"
[[ -n "$actual_data_hash" ]] || fail "BV robot result omitted its authoritative data_hash"
[[ "$actual_data_hash" == "$expected_data_hash" ]] ||
  fail "BV robot result hash $actual_data_hash disagrees with authoritative hash $expected_data_hash"

if (( robot_next_seen == 1 )); then
  jq -e --slurpfile ready "$br_ready" '
    (.id // "") as $candidate
    | if ($ready[0] | length) == 0 then
        $candidate == ""
      else
        ($candidate != "") and any($ready[0][]; .id == $candidate)
      end
  ' "$result" >/dev/null ||
    fail "BV next result is absent or not present in br ready; refusing an unsafe claim suggestion"
fi

if (( robot_plan_seen == 1 )); then
  jq -e --slurpfile ready "$br_ready" '
    ([.plan.tracks[]?.items[]?.id] | unique) as $planned
    | all($planned[]; . as $candidate | any($ready[0][]; .id == $candidate))
  ' "$result" >/dev/null ||
    fail "BV plan contains an issue that is not present in br ready"
fi

if (( robot_triage_seen == 1 )); then
  jq -e --slurpfile ready "$br_ready" '
    [.triage.quick_ref.top_picks[]?.id] as $picks
    | all($picks[]; . as $candidate | any($ready[0][]; .id == $candidate))
  ' "$result" >/dev/null ||
    fail "BV triage top picks contain an issue that is not present in br ready"
fi

source_hash_after="$(git hash-object "$authoritative")"
[[ "$source_hash_after" == "$source_hash_before" ]] ||
  fail "authoritative export changed during analysis; discard this result and retry"

cat "$result"
