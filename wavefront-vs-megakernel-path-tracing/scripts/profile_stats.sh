#!/bin/bash
# Nsight Compute counters for every scene and both GPU arms: raw metric ids, base units, one CSV per run.
# Output: results/profiles/stats/<scene>-<mode>.csv and the exact command in <scene>-<mode>.command.txt.
set -euo pipefail
BUILD_DIR="${WF_BUILD_DIR:-build}"
out=results/profiles/stats
mkdir -p "$out"
METRICS=device__attribute_multiprocessor_count,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__thread_inst_executed_per_inst_executed.ratio,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum
for scene in cornell divergence trianglehv; do
    for mode in megakernel wavefront; do
        cmd="ncu --csv --print-units base --target-processes all --metrics $METRICS $BUILD_DIR/wf render --scene $scene --width 1280 --height 720 --spp 1 --seed 1234 --mode $mode --out /tmp/ncu-stats-render"
        echo "$cmd" > "$out/$scene-$mode.command.txt"
        $cmd 2>&1 | grep '^"' > "$out/$scene-$mode.csv" || true
        echo "$scene $mode: $(grep -c . "$out/$scene-$mode.csv") rows"
    done
done
rm -f /tmp/ncu-stats-render*
