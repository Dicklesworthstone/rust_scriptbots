#!/usr/bin/env bash
# Source-pinned DSR analytics evidence. The statistical fixture and actual seeded
# world produce separate manifests; neither substitutes for the other. The real
# planted intervention/control-search acceptance remains explicitly open.
# If physics changes a pinned demographic trajectory, retain the failed probe,
# inspect its actual populations/births/deaths, document the new seed rationale,
# and rerun a fresh DSR profile. Never widen a significance threshold to pass.

set -euo pipefail

fail() { printf 'e2e_analytics: ERROR: %s\n' "$1" >&2; exit 1; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

verify_manifest() {
  python3 - "$1" "$2" "$3" <<'PY'
import hashlib, json, pathlib, re, subprocess, sys

directory, expected, mode = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
assert re.fullmatch(r"[0-9a-f]{40}", expected), "invalid source identity"
fixture = json.loads((directory / "fixture.json").read_text())
real = json.loads((directory / "real-world.json").read_text())
assert fixture["schema"] == "scriptbots.analytics.e2e-manifest.v2"
assert fixture["evidence_class"] == "synthetic_statistical_fixture"
assert fixture["verdict"] == "pass"
assert real["schema"] == "scriptbots.analytics.real-world-manifest.v1"
assert real["evidence_class"] == "seeded_world_reports_cli_and_parquet"
assert real["status"] == "pass"
for manifest in (fixture, real):
    assert manifest["source_commit"] == expected, "source mismatch"
    assert manifest["compiler_identity"], "missing compiler identity"
    assert pathlib.Path(manifest["retained_root"]).is_dir(), "missing retained fixture"
assert fixture["compiler_identity"] == real["compiler_identity"]
assert fixture["run_id"] != real["run_id"], "collapsed fixture/world identity"
assert real["births"] > 0 and real["deaths"] > 0 and real["ticks"] > 0
populations = real["populations"]
assert len(populations) == real["ticks"] and populations
summary = real["ground_truth_summary"]
for key, observed in {
    "tick_count": real["ticks"], "birth_records": real["births"],
    "death_records": real["deaths"], "population_first": populations[0],
    "population_last": populations[-1], "population_min": min(populations),
    "population_max": max(populations),
}.items():
    assert summary[key] == observed, (key, summary[key], observed)
assert abs(summary["population_mean"] - sum(populations) / len(populations)) < 1e-9
stages = {s["stage"]: s for s in fixture["stages"]}
assert len(stages) == len(fixture["stages"])
assert all(s["status"] == "pass" for s in stages.values())
assert set(stages) == {"synthetic_fixture_persistence", "report_suite_execution", "ground_truth_invariants", "frankenpandas_export_and_summary", "graph_exports", "narrative_search_fts"}
declared = stages["report_suite_execution"]["details"]["reports_executed"]
required = stages["report_suite_execution"]["details"]["required_reports"]
assert len(declared) == len(required) and set(declared) == set(required)
reports = {r["name"]: r for r in real["reports"]}
assert len(reports) == len(real["reports"]) and set(reports) == set(declared)
assert {"run-summary", "narrative-timeline", "narrative-validate", "lineage-structure", "interaction-centrality"} <= set(reports)
invariants = {i["invariant"]: i for i in fixture["invariants"]}
assert len(invariants) == len(fixture["invariants"])
assert all(i["status"] == "pass" for i in invariants.values())
assert {"planted_shift_significant_fdr", "null_noise_not_significant", "parquet_sql_row_count_equality", "narrative_search_hit_confirmed"} <= set(invariants)
def digest(path):
    return subprocess.check_output(["b3sum", str(path)], text=True).split()[0]
artifacts = [directory / "fixture.json", directory / "real-world.json", directory / "tests.list.log", directory / "tests.log"]
database = pathlib.Path(real["database"])
assert database.stat().st_size > 0 and digest(database) == real["database_blake3"]
artifacts.append(database)
for name, report in reports.items():
    assert report["exit_code"] == 0 and report["command"]
    path = pathlib.Path(report["json_path"])
    assert digest(path) == report["json_blake3"]
    output = json.loads(path.read_text())
    assert output["schema_version"] == report["schema_version"]
    assert output["report"] == name and output["db_path"] == real["database"]
    assert output["latest_tick"] == real["ticks"] and output["row_count"] == report["row_count"]
    markdown = pathlib.Path(report["markdown_path"])
    assert markdown.stat().st_size > 0
    artifacts.extend([path, markdown])
tables = {t["table"]: t for t in real["tables"]}
assert set(tables) == {"runs", "agents", "lineage_edges", "replay_events", "metrics"}
assert real["export_exit_code"] == 0 and real["export_command"]
for table in tables.values():
    assert table["sql_count"] == table["parquet_count"]
    path = pathlib.Path(table["path"])
    assert digest(path) == table["blake3"]
    artifacts.append(path)
assert tables["metrics"]["sql_count"] > 0 and tables["agents"]["sql_count"] > 0
negatives = {n["case"]: n for n in real["negative_controls"]}
assert set(negatives) == {"missing_database", "empty_database", "corrupt_database", "missing_report"}
for negative in negatives.values():
    assert negative["exit_code"] is not None and negative["exit_code"] != 0
    log = pathlib.Path(negative["stderr_path"])
    assert log.stat().st_size > 0
    artifacts.append(log)
assert real["remaining_acceptance"], "partial evidence must retain its qualification"
declarations = (directory / "tests.list.log").read_text()
declared_tests = re.search(r"^([0-9]+) tests, 0 benchmarks$", declarations, re.MULTILINE)
assert declared_tests and int(declared_tests[1]) > 0, "missing executed-test declaration"
for test in ("test_analytics_e2e_full_pipeline_and_invariants", "report_suite_on_a_real_seeded_simulation_matches_the_simulation_ground_truth"):
    assert re.search(rf"^{test}: test$", declarations, re.MULTILINE), "missing named pipeline test"
assert re.search(rf"^test result: ok\. {declared_tests[1]} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;", (directory / "tests.log").read_text(), re.MULTILINE), "not every declared pipeline test ran successfully"
hashes = "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n" for path in artifacts)
if mode == "record":
    (directory / "artifacts.sha256").write_text(hashes)
else:
    assert mode == "verify" and (directory / "artifacts.sha256").read_text() == hashes, "changed retained artifacts"
PY
}

if [[ ${1:-} == --verify-evidence ]]; then
  [[ $# == 3 ]] || fail "usage: --verify-evidence DIRECTORY EXPECTED_COMMIT"
  jq -e --arg source "$3" '.schema == "scriptbots.analytics-proof.v1" and .status == "pass" and .source == $source and .executed_tests > 0' "$2/verdict.json" >/dev/null || fail "missing source-bound analytics verdict"
  verify_manifest "$2" "$3" verify
  exit 0
fi
if [[ ${1:-} != --inside-dsr ]]; then
  [[ $# == 2 ]] || fail "usage: PROFILE UNIQUE_VERSION (pinned DSR configuration required)"
  exec bash scripts/dsr_verify.sh --run "$1" "$2"
fi
[[ $# == 2 && $2 = /* ]] || fail "missing external analytics proof directory"
[[ ${RCH_DISABLED:-} == 1 && ${RCH_CARGO_WRAPPER_BYPASS:-} == 1 ]] || fail "DSR native profile required"
[[ ${SCRIPTBOTS_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || fail "missing pinned source"
[[ $(git rev-parse HEAD) == "$SCRIPTBOTS_EXPECTED_COMMIT" ]] || fail "source mismatch"
[[ -z $(git status --porcelain --untracked-files=all) ]] || fail "dirty source"
case "$2/" in "$repo_root/"*) fail "proof directory must be external" ;; esac
out_dir=$2
mkdir "$out_dir" || fail "analytics proof directory reused or unavailable"
export SCRIPTBOTS_E2E_ANALYTICS_FIXTURE_MANIFEST="$out_dir/fixture.json"
export SCRIPTBOTS_E2E_ANALYTICS_REAL_MANIFEST="$out_dir/real-world.json"
command -v b3sum >/dev/null || fail "missing independent BLAKE3 reader"
cargo test --locked -p scriptbots-analytics --test e2e_pipeline -- --list 2>&1 | tee "$out_dir/tests.list.log"
for name in test_analytics_e2e_full_pipeline_and_invariants report_suite_on_a_real_seeded_simulation_matches_the_simulation_ground_truth; do
  rg -q "^$name: test$" "$out_dir/tests.list.log" || fail "missing named pipeline test: $name"
done
declared_tests=$(sed -nE 's/^([0-9]+) tests, 0 benchmarks$/\1/p' "$out_dir/tests.list.log")
[[ "$declared_tests" =~ ^[0-9]+$ && "$declared_tests" -gt 0 ]] || fail "missing test declaration count"
cargo test --locked -p scriptbots-analytics --test e2e_pipeline -- --nocapture --test-threads=1 2>&1 | tee "$out_dir/tests.log"
rg -q "^test result: ok\. $declared_tests passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;" "$out_dir/tests.log" || fail "not every declared test passed"
verify_manifest "$out_dir" "$SCRIPTBOTS_EXPECTED_COMMIT" record
sha256sum --check "$out_dir/artifacts.sha256"
jq -n --arg source "$SCRIPTBOTS_EXPECTED_COMMIT" --argjson tests "$declared_tests" \
  '{schema:"scriptbots.analytics-proof.v1",status:"pass",source:$source,executed_tests:$tests,scope:"synthetic statistical fixture plus real seeded-world reports, analyzer CLI and verified Parquet; original remaining acceptance is recorded in real-world.json"}' \
  > "$out_dir/verdict.json"
printf 'e2e_analytics: observed %s passing tests; retained evidence at %s\n' "$declared_tests" "$out_dir"
