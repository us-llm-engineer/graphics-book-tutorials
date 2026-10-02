#!/bin/bash
# Profiles the megakernel and wavefront modes with Nsight Compute. When counters cannot be read the exact
# command that was run and the tool's own output are stored as the evidence (never a guessed cause).
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

if [[ ! -f "$BUILD_DIR/wf" ]]; then
    echo "Error: $BUILD_DIR/wf not found" >&2
    exit 1
fi

mkdir -p results/profiles
rm -f results/profiles/*.error.txt

if [[ $quick_mode -eq 1 ]]; then
    dims="--width 64 --height 36 --spp 1"
else
    dims="--width 1280 --height 720 --spp 1"
fi

# One-line, JSON-safe copy of a text (backslashes, quotes, tabs and newlines escaped).
json_text() {
    printf '%s' "$1" | head -c 600 | sed ':a;N;$!ba;s/\\/\\\\/g;s/"/\\"/g;s/\t/ /g;s/\n/\\n/g'
}

available=true
summary_command=""
summary_message=""
failed_command=""
failed_output=""

# Raw metric ids: occupancy, warp execution efficiency, registers, DRAM and SM throughput, duration,
# issue activity, eligible warps and the main warp-stall reasons.
METRICS=sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,launch__registers_per_thread,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,smsp__average_warps_issue_stalled_membar_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio

for mode in megakernel wavefront; do
    cmd="ncu --csv --target-processes all --metrics $METRICS $BUILD_DIR/wf render --scene divergence $dims --seed 1234 --mode $mode --out results/profiles/$mode-render"
    summary_command="$cmd"
    if ! out=$($cmd 2>&1); then
        available=false
        failed_command="$cmd"
        failed_output="$out"
        break
    fi
    printf '%s\n' "$out" > "results/profiles/$mode.ncu-output.txt"
    # Keep only the CSV rows (quoted fields); the tool's progress lines stay in the .ncu-output.txt copy.
    printf '%s\n' "$out" | grep '^"' > "results/profiles/$mode.csv" || true
done

if ! $available; then
    summary_command="$failed_command"
    summary_message="$failed_output"
    for mode in megakernel wavefront; do
        {
            echo "$failed_command"
            echo ""
            echo "$failed_output"
        } > "results/profiles/$mode.error.txt"
    done
fi

{
    echo "{"
    echo "  \"schema_version\": 1,"
    if $available; then
        echo "  \"available\": true,"
    else
        echo "  \"available\": false,"
    fi
    echo "  \"command\": \"$(json_text "$summary_command")\","
    echo "  \"message\": \"$(json_text "$summary_message")\""
    echo "}"
} > results/profiles/summary.json

exit 0
