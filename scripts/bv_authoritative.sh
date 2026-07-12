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

robot_command_seen=0
for argument in "$@"; do
  case "$argument" in
    --robot-*)
      robot_command_seen=1
      ;;
    --db | --db=* | --workspace | --workspace=* | --format | --format=* | -f | -f=* | --update | --rollback | --export-* | --pages | --preview-pages | --watch-export)
      fail "option may bypass or mutate the authoritative read-only view: $argument" 64
      ;;
    --graph-format=dot | --graph-format=mermaid)
      fail "only JSON robot output is supported; use --graph-format=json" 64
      ;;
  esac
done
(( robot_command_seen == 1 )) || fail "a --robot-* command is required; interactive BV is forbidden" 64

source_hash_before="$(git hash-object "$authoritative")"
view_root="$(mktemp -d "${TMPDIR:-/tmp}/scriptbots-bv-authoritative.XXXXXXXX")"
mkdir "$view_root/.beads"
ln -s "$authoritative" "$view_root/.beads/beads.jsonl"

source_stats="$view_root/source-stats.json"
br_list="$view_root/br-list.json"
br_ready="$view_root/br-ready.json"
verifier="$view_root/bv-triage.json"
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

source_hash_after="$(git hash-object "$authoritative")"
[[ "$source_hash_after" == "$source_hash_before" ]] ||
  fail "authoritative export changed during analysis; discard this result and retry"

cat "$result"
