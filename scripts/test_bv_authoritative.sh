#!/usr/bin/env bash
# End-to-end fixture for the BR export -> authoritative BV robot integration.

set -euo pipefail

fail() {
  printf 'test-bv-authoritative: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
wrapper="$script_dir/bv_authoritative.sh"

for tool in br bv git jq mktemp mkdir cp; do
  command -v "$tool" >/dev/null 2>&1 || fail "required tool is unavailable: $tool"
done
[[ -x "$wrapper" ]] || fail "wrapper is not executable: $wrapper"

fixture="$(mktemp -d "${TMPDIR:-/tmp}/scriptbots-bv-fixture.XXXXXXXX")"
mkdir "$fixture/scripts"
cp "$wrapper" "$fixture/scripts/bv_authoritative.sh"
chmod +x "$fixture/scripts/bv_authoritative.sh"

(
  cd "$fixture"
  br init --prefix fx --json >/dev/null
  ready_id="$(br create 'Ready foundation' --type task --priority 0 --silent)"
  blocked_id="$(br create 'Blocked dependent' --type bug --priority 1 --silent)"
  in_progress_id="$(br create 'Active independent work' --type task --priority 2 --status in_progress --silent)"
  closed_id="$(br create 'Completed evidence' --type task --priority 3 --status closed --silent)"
  br dep add "$blocked_id" "$ready_id" --type blocks --json >/dev/null
  br dep add "$ready_id" "$closed_id" --type parent-child --json >/dev/null
  br dep add "$in_progress_id" "$closed_id" --type related --json >/dev/null
  br sync --flush-only --json >/dev/null

  [[ ! -e .beads/beads.jsonl ]] || fail "fixture unexpectedly created the stale alternate snapshot"
  jq -c 'select(.id == $id)' --arg id "$closed_id" .beads/issues.jsonl >.beads/beads.jsonl

  issues_hash_before="$(git hash-object .beads/issues.jsonl)"
  stale_hash_before="$(git hash-object .beads/beads.jsonl)"

  br_all="$(br list --json --no-db --all --limit 0)"
  br_ready="$(br ready --json --no-db --limit 0 2>/dev/null)"
  br_blocked="$(br show "$blocked_id" --json --no-db)"
  [[ "$(jq 'length' <<<"$br_all")" == 4 ]] || fail "br list did not return all four exported issues"
  [[ "$(jq 'length' <<<"$br_ready")" == 2 ]] || fail "br ready did not return the ready and in-progress issues"
  [[ "$(jq -s '[.[] | .dependencies[]?] | length' .beads/issues.jsonl)" == 3 ]] ||
    fail "fixture did not retain blocking, hierarchy, and related relationship records"
  jq -e --arg id "$ready_id" '.[0].dependencies == [{id: $id, title: "Ready foundation", status: "open", priority: 0, dependency_type: "blocks"}]' \
    <<<"$br_blocked" >/dev/null || fail "br show did not preserve the blocking dependency"

  stale_result="$(BV_OUTPUT_FORMAT=json bv --no-cache --robot-triage)"
  [[ "$(jq '.triage.project_health.counts.total' <<<"$stale_result")" == 1 ]] ||
    fail "fixture did not prove default BV selected the stale alternate snapshot"

  authoritative_result="$(scripts/bv_authoritative.sh --robot-triage)"
  jq -e '
    .triage.project_health.counts.total == 4
    and .triage.project_health.counts.by_status == {closed: 1, in_progress: 1, open: 2}
    and .triage.project_health.graph.node_count == 4
    and .triage.project_health.graph.edge_count == 1
    and .triage.quick_ref.actionable_count == 2
  ' <<<"$authoritative_result" >/dev/null || fail "authoritative wrapper returned incorrect tracker state"

  authoritative_plan="$(scripts/bv_authoritative.sh --robot-plan)"
  jq -e --argjson ready "$br_ready" '
    ([.plan.tracks[]?.items[]?.id] | sort) == ($ready | map(.id) | sort)
    and .plan.total_actionable == ($ready | length)
  ' <<<"$authoritative_plan" >/dev/null ||
    fail "authoritative wrapper did not preserve the exact br ready ID set"

  authoritative_next="$(scripts/bv_authoritative.sh --robot-next)"
  jq -e --argjson ready "$br_ready" '
    .id as $candidate | any($ready[]; .id == $candidate)
  ' <<<"$authoritative_next" >/dev/null ||
    fail "authoritative wrapper returned a next issue outside br ready"

  authoritative_hash="$(jq -r '.data_hash' <<<"$authoritative_result")"
  stale_hash="$(jq -r '.data_hash' <<<"$stale_result")"
  [[ -n "$authoritative_hash" && "$authoritative_hash" != "$stale_hash" ]] ||
    fail "authoritative and stale BV snapshots were not distinguishable by data_hash"

  if scripts/bv_authoritative.sh --db .beads --robot-triage >/dev/null 2>&1; then
    fail "wrapper accepted a caller-controlled BV source"
  fi

  [[ ! -e .bv ]] || fail "fixture unexpectedly contains BV mutation state"
  feedback_before="$(find . -maxdepth 3 -type f -iname '*feedback*' -print)"
  if scripts/bv_authoritative.sh --robot-drift --save-baseline unsafe >/dev/null 2>&1; then
    fail "wrapper accepted a baseline-writing BV command"
  fi
  if scripts/bv_authoritative.sh --robot-confirm-correlation deadbeef:"$ready_id" >/dev/null 2>&1; then
    fail "wrapper accepted a correlation-feedback mutation"
  fi
  if scripts/bv_authoritative.sh --robot-triage-by-track >/dev/null 2>&1; then
    fail "wrapper accepted a modifier without a primary robot command"
  fi
  if scripts/bv_authoritative.sh --robot-insights --as-of HEAD >/dev/null 2>&1; then
    fail "wrapper accepted a historical source for current-authority analysis"
  fi
  if scripts/bv_authoritative.sh --robot-history >/dev/null 2>&1; then
    fail "wrapper accepted BV history, which does not honor the isolated source"
  fi
  if scripts/bv_authoritative.sh --robot-diff --diff-since HEAD >/dev/null 2>&1; then
    fail "wrapper accepted BV diff, whose historical loader does not honor the isolated source"
  fi
  [[ ! -e .bv ]] || fail "rejected BV mutation options created repository state"
  [[ "$(find . -maxdepth 3 -type f -iname '*feedback*' -print)" == "$feedback_before" ]] ||
    fail "rejected BV mutation options created a feedback file"

  [[ "$(git hash-object .beads/issues.jsonl)" == "$issues_hash_before" ]] ||
    fail "wrapper modified the authoritative issues.jsonl export"
  [[ "$(git hash-object .beads/beads.jsonl)" == "$stale_hash_before" ]] ||
    fail "wrapper modified the stale beads.jsonl snapshot"

  printf 'authoritative fixture: total=4 statuses=2-open/1-in_progress/1-closed stored_relationships=3 blocking_edges=1 actionable_ids_exact=2 hash=%s stale_hash=%s\n' \
    "$authoritative_hash" "$stale_hash"
  printf 'fixture retained for audit: %s\n' "$fixture"
)

missing_fixture="$(mktemp -d "${TMPDIR:-/tmp}/scriptbots-bv-missing.XXXXXXXX")"
mkdir "$missing_fixture/scripts"
cp "$wrapper" "$missing_fixture/scripts/bv_authoritative.sh"
chmod +x "$missing_fixture/scripts/bv_authoritative.sh"
if "$missing_fixture/scripts/bv_authoritative.sh" --robot-triage >/dev/null 2>&1; then
  fail "wrapper accepted a missing authoritative export"
fi

empty_fixture="$(mktemp -d "${TMPDIR:-/tmp}/scriptbots-bv-empty.XXXXXXXX")"
mkdir "$empty_fixture/scripts" "$empty_fixture/.beads"
cp "$wrapper" "$empty_fixture/scripts/bv_authoritative.sh"
chmod +x "$empty_fixture/scripts/bv_authoritative.sh"
: >"$empty_fixture/.beads/issues.jsonl"
if "$empty_fixture/scripts/bv_authoritative.sh" --robot-triage >/dev/null 2>&1; then
  fail "wrapper accepted an empty authoritative export"
fi
printf 'negative gates: source override, mutation flags, modifier-only, historical and mixed-source commands, missing export, and empty export all refused\n'
