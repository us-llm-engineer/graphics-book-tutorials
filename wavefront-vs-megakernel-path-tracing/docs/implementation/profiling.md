# Profiling and sanitizers

## Nsight Compute

[`scripts/profile_ncu.sh`](../../scripts/profile_ncu.sh) profiles one pass per arm on the divergence scene and stores the exact command and the tool's own output; if counters are unavailable, the failing command and message are stored instead of a guess. [`scripts/profile_stats.sh`](../../scripts/profile_stats.sh) profiles all three scenes and both arms with raw metric ids and `--print-units base`, one CSV per run. [`tools/ncusum.cpp`](../../tools/ncusum.cpp) turns a CSV into per-kernel means.

Metrics collected: SM count, SM clock, kernel duration, DRAM bytes and percent of peak, FADD, FMUL and FFMA thread instructions, `smsp__thread_inst_executed_per_inst_executed.ratio` (active lanes per warp instruction), achieved occupancy, executed instructions, and the DRAM peak (`dram__bytes.sum.peak_sustained` times the DRAM clock).

## Roofline derivation (Figure 6)

- FLOP = FADD + FMUL + 2 FFMA thread instructions (contraction is disabled in this build, so FFMA is rare).
- Arithmetic intensity = FLOP / DRAM bytes; achieved GFLOP/s = FLOP / kernel duration.
- `P_peak = SMs x 64 x 2 x SM clock` (64 FP32 lanes per SM on Turing); `B = 32 bytes/cycle x DRAM clock`; ridge point `P_peak / B`. Both roofs come from counters read on the same machine.
- Marker area is the kernel's share of kernel time; the megakernel is one kernel.

## Lane efficiency (Figure 5)

Per kernel the profiler reports the mean number of active lanes per warp instruction. The figure weights each kernel by executed warp instructions times launches and divides by 32. The model it is compared with uses the material mix of primary hits (`tools/wfstats.cpp`, section `divergence`).

## Sanitizers

[`scripts/run_sanitizers.sh`](../../scripts/run_sanitizers.sh) runs `compute-sanitizer --tool memcheck` on both GPU arms and the ASan/UBSan build in CPU mode, recording each command, the real exit code and the tool's error count in `results/sanitizers.json`. ASan on the GPU modes aborts inside the CUDA driver under WSL; those rows are recorded as not run, never as clean.
