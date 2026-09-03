#!/usr/bin/env bash
# e2e_architecture_recipes.sh — End-to-end verification for architecture guide contracts & recipes (bd-bsuh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Running architecture contract & extension recipe verification ==="
cd "${REPO_ROOT}"

OUTPUT=$(rch exec -- cargo test -p scriptbots-app --test architecture_contract -- --nocapture 2>&1)

EVIDENCE=$(python3 -c "
import sys, json
text = sys.stdin.read()
results = []
for line in text.splitlines():
    idx = line.find('{\"schema\":\"scriptbots.architecture-recipe.evidence.v1\"')
    if idx != -1:
        json_str = line[idx:].strip()
        try:
            data = json.loads(json_str)
            results.append(json.dumps(data))
        except Exception:
            pass
if results:
    print('\n'.join(results))
" <<< "${OUTPUT}")

if [ -z "${EVIDENCE}" ]; then
    echo "ERROR: Expected JSONL evidence not found in test output" >&2
    exit 1
fi

echo "${EVIDENCE}"

# Verify each recipe ID is present with none failure disposition
for recipe_id in "brain_family_extension" "scenario_extension" "frontend_extension"; do
    if ! echo "${OUTPUT}" | grep -q "\"recipe_id\":\"${recipe_id}\".*\"failure_disposition\":\"none\""; then
        echo "ERROR: Missing clean evidence for recipe ${recipe_id}" >&2
        exit 1
    fi
done

echo "=== All architecture contracts and extension recipes verified successfully! ==="
exit 0
