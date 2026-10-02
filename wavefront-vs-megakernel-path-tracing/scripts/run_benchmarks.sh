#!/bin/bash
# Runs the fixed benchmark protocol (bench/protocol.json) with wfbench and stores one raw result file
# per scene under results/. wfbench exits 0 for every valid result (the verdict is recorded in the JSON,
# a weakened verdict is a result, not an error) and 2 on an error; only an error stops this script.
set -euo pipefail

BUILD_DIR="${WF_BUILD_DIR:-build}"

quick_mode=0
for arg in "$@"; do
    if [[ "$arg" == "--quick" ]]; then
        quick_mode=1
    else
        echo "Usage: $0 [--quick] - unknown flag: $arg" >&2
        exit 2
    fi
done

if [[ ! -f "$BUILD_DIR/wfbench" ]]; then
    echo "Error: $BUILD_DIR/wfbench not found" >&2
    exit 1
fi

protocol="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/bench/protocol.json"
if [[ ! -f "$protocol" ]]; then
    echo "Error: bench/protocol.json not found" >&2
    exit 1
fi

# Protocol values come only from the fixed protocol file: line 1 the scenes, line 2 the wfbench arguments.
parsed="$(python3 - "$protocol" "$quick_mode" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
q = p["quick"] if sys.argv[2] == "1" else p
g = lambda k: q.get(k, p[k])
print(" ".join(g("scenes")))
print(f'--width {g("width")} --height {g("height")} --spp {g("spp")} --reps {g("reps")} --warmup {g("warmup")} '
      f'--max-depth {p["max_depth"]} --seed {p["seed"]} --cpu-spp {g("cpu_spp")}')
PY
)"
scene_list="$(printf '%s\n' "$parsed" | sed -n 1p)"
arg_list="$(printf '%s\n' "$parsed" | sed -n 2p)"

mkdir -p results
for scene in $scene_list; do
    echo "run_benchmarks: $scene ($arg_list)"
    rc=0
    # shellcheck disable=SC2086
    "$BUILD_DIR/wfbench" --scene "$scene" $arg_list --out "results/bench-$scene.json" || rc=$?
    if [[ $rc -ge 2 ]]; then
        echo "Error: wfbench failed for scene $scene (exit $rc)" >&2
        exit 1
    fi
done
exit 0
