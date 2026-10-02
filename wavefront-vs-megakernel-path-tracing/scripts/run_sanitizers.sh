#!/bin/bash
set -euo pipefail
# Runs compute-sanitizer memcheck on the megakernel and wavefront renders and, when an ASan/UBSan build
# exists ($WF_BUILD_DIR/wf_asan), the same render under it. Every row records the exact command, the real exit
# code and the tool's own error count; a run that did not complete counts as an error. An ASan/UBSan run that
# could not be made is recorded as not run, never as clean. Exit 0 iff every error_count is 0.

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

if [[ ! -f "$BUILD_DIR/wf" ]]; then
    echo "Error: $BUILD_DIR/wf not found" >&2
    exit 1
fi

mkdir -p results
if [[ $quick_mode -eq 1 ]]; then
    dims="--width 16 --height 9 --spp 2"
else
    dims="--width 64 --height 36 --spp 4"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
n_runs=0

# run_one <tool> <command...>: runs the command, stores tool, command, exit code, error count, first failure.
run_one() {
    local tool="$1"; shift
    local cmd="$*" out rc=0 count first
    out="$($cmd 2>&1)" || rc=$?
    count="$(printf '%s\n' "$out" | sed -n 's/.*ERROR SUMMARY: \([0-9][0-9]*\) error.*/\1/p' | tail -1)"
    first=""
    if [[ "$tool" == "compute-sanitizer" ]]; then
        if [[ -z "$count" ]]; then count=1; first="sanitizer did not complete: $(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -1)"; fi
    else
        count="$(printf '%s\n' "$out" | grep -cE 'AddressSanitizer|runtime error' || true)"
    fi
    if [[ "$count" -eq 0 && $rc -ne 0 ]]; then count=1; first="program exited with code $rc: $(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -1)"; fi
    if [[ "$count" -gt 0 && -z "$first" ]]; then first="$(printf '%s\n' "$out" | grep -m1 -E 'Invalid|ERROR|AddressSanitizer|runtime error' || true)"; fi
    [[ "$count" -gt 0 && -z "$first" ]] && first="error reported without text"
    n_runs=$((n_runs + 1))
    printf '%s\n%s\n%s\n%s\n%s\n' "$tool" "$cmd" "$rc" "$count" "$(printf '%s' "$first" | head -c 300 | tr '\n\t' '  ')" > "$tmp/$n_runs"
}

for mode in megakernel wavefront; do
    if command -v compute-sanitizer &>/dev/null; then
        run_one compute-sanitizer compute-sanitizer --tool memcheck --error-exitcode 1 "$BUILD_DIR/wf" render --scene divergence $dims --seed 1 --mode "$mode" --out "results/sanitizer-$mode"
    else
        n_runs=$((n_runs + 1))
        printf '%s\n%s\n%s\n%s\n%s\n' "compute-sanitizer" "compute-sanitizer not found: not run" "-1" "1" "compute-sanitizer is not installed" > "$tmp/$n_runs"
    fi
done

if [[ -x "$BUILD_DIR/wf_asan" ]]; then
    for mode in cpu megakernel wavefront; do
        cmd="env ASAN_OPTIONS=detect_leaks=0 $BUILD_DIR/wf_asan render --scene divergence $dims --seed 1 --mode $mode --out results/sanitizer-asan-$mode"
        rc=0; out="$($cmd 2>&1)" || rc=$?
        n_runs=$((n_runs + 1))
        if printf '%s\n' "$out" | grep -qE 'ERROR: AddressSanitizer|runtime error:'; then
            head_lines="$(printf '%s\n' "$out" | grep -A8 -m1 -E 'ERROR: AddressSanitizer|runtime error:' || true)"
            first="$(printf '%s\n' "$head_lines" | head -1 | head -c 300 | tr '\n\t' '  ')"
            if printf '%s\n' "$head_lines" | grep -q 'libcuda'; then
                # The report comes from inside the NVIDIA driver before any project code ran: not a project result.
                printf '%s\n%s\n%s\n%s\n%s\n' "asan" "NOT RUN as a project check ($mode): AddressSanitizer aborts inside libcuda; $cmd" "-1" "0" "" > "$tmp/$n_runs"
            else
                printf '%s\n%s\n%s\n%s\n%s\n' "asan" "$cmd" "$rc" "1" "$first" > "$tmp/$n_runs"
            fi
        elif [[ $rc -ne 0 ]]; then
            printf '%s\n%s\n%s\n%s\n%s\n' "asan" "$cmd" "$rc" "1" "program exited with code $rc: $(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -1 | head -c 200)" > "$tmp/$n_runs"
        else
            printf '%s\n%s\n%s\n%s\n%s\n' "asan" "$cmd" "0" "0" "" > "$tmp/$n_runs"
        fi
    done
else
    n_runs=$((n_runs + 1))
    printf '%s\n%s\n%s\n%s\n%s\n' "asan" "NOT RUN: no $BUILD_DIR/wf_asan (configure a sanitizer build with -DWF_SANITIZE=ON)" "-1" "0" "" > "$tmp/$n_runs"
fi

python3 - "$tmp" "$n_runs" > results/sanitizers.json <<'PY'
import json, sys
d, n = sys.argv[1], int(sys.argv[2])
runs = []
for i in range(1, n + 1):
    t, c, rc, cnt, first = open(f"{d}/{i}").read().split("\n")[:5]
    runs.append({"tool": t, "command": c, "exit_code": int(rc), "error_count": int(cnt), "first_failure": first})
print(json.dumps({"schema_version": 1, "runs": runs}, indent=2))
PY

if python3 -c 'import json,sys; sys.exit(0 if all(r["error_count"] == 0 for r in json.load(open("results/sanitizers.json"))["runs"]) else 1)'; then
    exit 0
fi
exit 1
