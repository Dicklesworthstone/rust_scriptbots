#!/usr/bin/env bash
# Real CLI acceptance inside a pinned DSR tournament-smoke or tournament-full lane.
# Preserve every execution, observation and negative fixture outside the checkout.

set -euo pipefail

fail() {
  printf 'e2e_tournament_leaderboard: ERROR: %s\n' "$1" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"
cd "$repo_root"

[[ ${RCH_DISABLED:-} == 1 && ${RCH_CARGO_WRAPPER_BYPASS:-} == 1 ]] || fail "invoke through a pinned native DSR profile"
[[ ${SCRIPTBOTS_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || fail "missing expected source identity"
[[ $(git rev-parse HEAD) == "$SCRIPTBOTS_EXPECTED_COMMIT" && -z $(git status --porcelain --untracked-files=all) ]] || fail "source is dirty or mismatched"
proof=${SCRIPTBOTS_TOURNAMENT_PROOF_DIR:-}
[[ "$proof" = /* && ! -e "$proof" ]] || fail "require a fresh external proof directory"
case "$proof/" in "$repo_root/"*) fail "proof directory must be outside the source" ;; esac
mkdir "$proof"
mode=${1:-}
case "$mode" in
  smoke) protocol_args=(--smoke --ticks 2000) ;;
  full) protocol_args=() ;;
  *) fail "expected smoke or full mode" ;;
esac
export RUST_LOG="scriptbots::tournament=info,info"
cargo build --locked -p scriptbots-app --bin scriptbots-app --message-format=json | tee "$proof/build.jsonl"
binary=$(jq -e -r -s '
  [.[] | select(.reason == "compiler-artifact" and .target.name == "scriptbots-app"
     and (.target.kind | index("bin")) != null and .executable != null) | .executable]
  | unique
  | if length == 1 then .[0] else error("expected one Cargo-reported tournament executable") end
' "$proof/build.jsonl")
[[ -x "$binary" ]] || fail "missing built tournament executable"
"$binary" tournament "${protocol_args[@]}" --jobs 4 --out "$proof/first" 2>&1 | tee "$proof/first.log"
"$binary" tournament "${protocol_args[@]}" --jobs 4 --out "$proof/second" \
  --check "$proof/first/leaderboard.md" 2>&1 | tee "$proof/second.log"
for name in tournament_results.jsonl ratings.json leaderboard.md; do
  [[ -s "$proof/first/$name" && -s "$proof/second/$name" ]] || fail "missing $name"
  cmp "$proof/first/$name" "$proof/second/$name" || fail "repeated $name bytes differ"
done

python3 - "$proof" "$repo_root/tournament/spec.toml" "$mode" "$SCRIPTBOTS_EXPECTED_COMMIT" <<'PY'
import itertools, json, math, pathlib, sys, tomllib
proof, spec_path, mode, source = sys.argv[1:]
proof = pathlib.Path(proof)
spec = tomllib.loads(pathlib.Path(spec_path).read_text())
protocol = spec['smoke'] if mode == 'smoke' else spec['tournament']
families, seeds = protocol['families'], protocol['seeds']
cohort = protocol['cohort_size']
ticks = 2000 if mode == 'smoke' else protocol['ticks']
if protocol['order_policy'] == 'both_assignments':
    orders = [families, families[::-1]]
elif protocol['order_policy'] == 'balanced_latin_square':
    orders = [families[i:] + families[:i] for i in range(len(families))]
else:
    orders = list(map(list, itertools.permutations(families)))
expected = set(itertools.product(seeds, range(len(orders)), families))
rows = [json.loads(line) for line in (proof/'first/tournament_results.jsonl').read_text().splitlines()]
cells = [(row['seed'], row['spawn_order_index'], row['family']) for row in rows]
assert len(cells) == len(set(cells)) == len(expected) and set(cells) == expected
for row in rows:
    assert row['ticks_run'] == ticks and row['spawn_order'] == orders[row['spawn_order_index']]
    assert row['source_revision'] == source and row['source_tree_clean'] is True
    assert row['source_provenance_complete'] is True and row['reproducible'] is True
    assert row['sense_backend'] == 'cpu_simd' and row['sense_determinism'] == 'exact'
    assert row['initial_brain_source_scalars'] == spec['parameters']['nodes_per_family']
    assert len(row['initial_mutation_rates']) == cohort // len(families)
    assert all(math.isfinite(rates['primary']) and rates['primary'] >= 0 and math.isfinite(rates['secondary']) and rates['secondary'] >= 0 for rates in row['initial_mutation_rates'])
assert len({row['protocol_digest'] for row in rows}) == 1
assert len({row['config_digest'] for row in rows}) == 1
matches = len({row['match_id'] for row in rows})
assert matches == len(seeds) * len(orders)
ratings = json.loads((proof/'first/ratings.json').read_text())
for axis in ratings['axes'].values():
    if 'Rated' in axis:
        assert axis['Rated']['n_matches'] == matches and axis['Rated']['n_seeds'] == len(seeds)
        assert set(axis['Rated']['ratings']) == set(families)
document = (proof/'first/leaderboard.md').read_text()
assert f'| Tick Budget | {ticks} ticks per match |' in document
assert f'| Observed Matrix | {matches} matches, {len(rows)} family rows |' in document
assert ('Smoke Report' in document) == (mode == 'smoke')
for name in ('first.log', 'second.log'):
    log = (proof/name).read_text()
    for observation in ('starting tournament leaderboard execution', 'match outcome row',
                        'reproducibility gate passed', 'leaderboard execution complete'):
        assert observation in log, (name, observation)
print(json.dumps({'source': source, 'mode': mode, 'matches': matches, 'rows': len(rows),
                  'seeds': len(seeds), 'ticks_per_match': ticks}, sort_keys=True))
PY

if [[ "$mode" == smoke ]]; then
  for negative in document config rows ratings; do
    mkdir "$proof/$negative"
    cp "$proof/first/tournament_results.jsonl" "$proof/first/ratings.json" "$proof/first/leaderboard.md" "$proof/$negative/"
    case "$negative" in
      document) printf '\nDeliberate document mutation.\n' >> "$proof/$negative/leaderboard.md" ;;
      config) sed -E "s/Effective Config Digest \| \`[a-f0-9]+\`/Effective Config Digest | \`stale-config\`/" \
        "$proof/first/leaderboard.md" > "$proof/$negative/leaderboard.md" ;;
      rows) jq -c 'if .family == "mlp" then .survival_share = 0.123456 else . end' \
        "$proof/first/tournament_results.jsonl" > "$proof/$negative/tournament_results.jsonl" ;;
      ratings) jq '.warnings += ["deliberate rating mutation"]' \
        "$proof/first/ratings.json" > "$proof/$negative/ratings.json" ;;
    esac
    if "$binary" tournament "${protocol_args[@]}" --jobs 4 --check "$proof/$negative/leaderboard.md" \
      > "$proof/$negative.log" 2>&1; then
      fail "accepted $negative mutation"
    fi
    case "$negative" in
      config) rg -q 'config digest drift detected' "$proof/$negative.log" || fail "missing config refusal" ;;
      document) rg -q 'leaderboard document drift detected' "$proof/$negative.log" || fail "missing document refusal" ;;
      rows|ratings) rg -q 'tournament artifact drift at' "$proof/$negative.log" || fail "missing artifact refusal" ;;
    esac
    printf 'Observed rejection: %s\n' "$negative"
  done
fi
sha256sum "$proof/first/tournament_results.jsonl" "$proof/first/ratings.json" "$proof/first/leaderboard.md" \
  "$proof/second/tournament_results.jsonl" "$proof/second/ratings.json" "$proof/second/leaderboard.md" > "$proof/artifacts.sha256"
jq -n --arg source "$SCRIPTBOTS_EXPECTED_COMMIT" --arg mode "$mode" \
  '{schema:"scriptbots.tournament-proof.v1",status:"pass",source:$source,mode:$mode,identical_invocations:2}' > "$proof/verdict.json"
printf 'Observed identical %s CLI artifacts from two invocations; proof: %s\n' "$mode" "$proof"
