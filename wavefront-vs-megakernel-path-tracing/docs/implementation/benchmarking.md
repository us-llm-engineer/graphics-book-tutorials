# Benchmarking

Sources: [`include/wf/bench.hpp`](../../include/wf/bench.hpp), [`src/bench.cpp`](../../src/bench.cpp), [`tools/wfbench.cpp`](../../tools/wfbench.cpp), [`scripts/run_benchmarks.sh`](../../scripts/run_benchmarks.sh), [`bench/protocol.json`](../../bench/protocol.json).

## Protocol

The parameters live in `bench/protocol.json` and nowhere else: 1280x720, 64 samples per pixel, maximum depth 8, seed 1234, 2 warm-up runs, 5 timed runs, CPU compared at 4 samples per pixel. Timing uses CUDA events around the whole render (all passes). Throughput numbers use a separate untimed counting run for rays (extension plus shadow rays), so ray counting never contaminates timings. Per-stage wavefront times are collected in a separate instrumented run (`results/stage-times.run.txt`) and never mixed with benchmark timings.

## Verdict rule

Defined once in `bench.hpp` and applied by `apply_verdict`:

- speedup = median(megakernel) / median(wavefront) must be at least `kSpeedupTarget = 1.30`;
- every quality check must pass: megakernel against wavefront equal (bitwise), CPU against each GPU arm within three combined standard errors on at least 99.5% of pixels;
- peak device memory of every condition at most `kVramBudgetBytes = 3.5e9`;
- no queue overflows, invalid hits or non-finite pixels in any condition.

Anything else gives `hypothesis_weakened` with the reasons listed. A weakened verdict is a valid result: `wfbench` exits 0 and records it in the JSON; only an error exits with 2.

## Report schema (version 1)

`results/bench-<scene>.json` contains: `schema_version`, scene name and hash, width, height, spp, max depth, seed, warm-up and repetition counts, device name, driver and CUDA runtime versions, `conditions` (one object per mode with `run_ms`, `min`, `median`, `p10`, `p90`, `max`, `spread_rel`, `samples_per_second`, `rays_per_second`, `peak_vram_bytes`, `queue_high_water`, `queue_overflows`, `invalid_hits`, `nonfinite_pixels`), `quality` checks (name, value, threshold, pass, detail), `speedup`, `speedup_meets_target`, `vram_within_budget`, `verdict` and `reasons`. The writer and a validator are in `src/bench.cpp`; the tests reject corrupted files.

## Reading the numbers

Percentiles are linear interpolation between order statistics. With five runs, p10 and p90 are close to the minimum and maximum; `spread_rel` is the relative spread of the runs. Clocks were not locked, so run-to-run spread of several percent is expected on a laptop GPU.
