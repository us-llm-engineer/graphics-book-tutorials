# Technical report: wavefront versus megakernel on a GTX 1650 Ti

## Question
Is a queue-based wavefront path tracer at least 1.30x faster than a megakernel on a divergent scene, with
peak GPU memory at or below 3.5 GB? Both GPU arms and the CPU reference use the same shading code
(`include/wf/path_core.hpp`) and a counter-based RNG, so they can be compared pixel for pixel.

## Setup
GTX 1650 Ti, CUDA 12.9. Scenes: `cornell`, `divergence`, `trianglehv`. 1280x720, 64 spp, depth 8, seed 1234,
2 warmup and 5 timed runs. Compiled with `--fmad=false`. Raw results: `results/bench-<scene>.json`.

## Result
| Scene | Megakernel median (ms) | Wavefront median (ms) | Speedup | Verdict |
|---|---|---|---|---|
| cornell | 447.3 | 965.4 | 0.46 | below 1.30 |
| divergence | 1060.6 | 1585.3 | 0.67 | below 1.30 |
| trianglehv | 2141.5 | 3118.9 | 0.69 | below 1.30 |

Quality gates pass (CPU against both GPU arms within 3 standard errors on 99.9% of pixels; megakernel and
wavefront agree bitwise). No queue overflow, invalid hit or non-finite pixel was recorded. Peak memory was
42 to 54 MB for the megakernel and about 490 to 504 MB for the wavefront, well inside the budget. A repeat of
`trianglehv` gave 0.78, so run-to-run spread of several percent is real.

Label: **direct** (measured here): the speedup and memory numbers above on this GPU and this implementation.

## Why the wavefront lost
An instrumented run at 4 spp (`results/stage-times.run.txt`) splits wavefront time by stage. On `cornell`,
traversal is about 10% and bookkeeping (classify, compaction, counter updates) about half. On `trianglehv`
traversal is about 55%. The implementation synchronises with the host after each bounce and moves per-path
state through global memory, so on small scenes the fixed cost per bounce outweighs the divergence it removes.

Label: **transferable** (likely applies to similar designs): per-bounce synchronisation and state traffic
set a floor that divergence reduction must beat. Label: **unsupported** (not tested): that a tuned wavefront
(device-side scheduling, persistent threads, fewer passes) would pass 1.30x.

## Profiler evidence (Nsight Compute, `divergence`, 1280x720, 1 spp)
Raw rows: `results/profiles/{megakernel,wavefront}.csv`; summaries by `ncusum`. Command recorded in `results/profiles/summary.json`.

| Kernel | Time (ms) | Occupancy | Warp execution efficiency (of 32) | SM throughput | DRAM throughput |
|---|---|---|---|---|---|
| megakernel (1 launch) | 16.1 | 55% | 8.7 | 56% | 1% |
| wavefront: extend (9 launches) | 7.5 | 70% | 10.2 | 49% | 7% |
| wavefront: classify | 3.0 | 84% | 22.3 | 8% | 56% |
| wavefront: shade lambert / dielectric / metal | 2.4 / 1.7 / 1.1 | 53 to 78% | 30 to 31 | 3 to 6% | 41% |
| wavefront: shadow | 1.5 | 66% | 9.6 | 45% | 9% |
| wavefront, all kernels | 19.5 | 74% (mean) | 26 (mean) | 15% | 39% |

Label: **direct**. The wavefront does what it is designed to do for shading: warp execution efficiency rises from 8.7
to about 30 of 32 lanes. But traversal (`extend`, 38% of its kernel time) and shadow rays stay as divergent as the
megakernel (about 10 of 32), and the classify and shading kernels become memory-bound (41 to 56% DRAM throughput,
long-scoreboard stalls) because every bounce reads and writes per-path state in global memory. The megakernel keeps
state in registers (1% DRAM). Summed kernel time at 1 spp is already 19.5 ms against 16.1 ms, and the timed runs
add host synchronisation after each bounce on top, which matches the measured 0.46 to 0.69.

Label: **unsupported** (not tested): that removing the state traffic, for example with fewer, fused passes, would close the gap.

## Limits
- One GPU, one driver, three scenes; a negative result for this implementation, not for the paradigm.
- The profile is one frame at 1 sample per pixel on `divergence`; stall reasons were sampled per kernel with serialized replay, so absolute times differ from the timed runs.
- Stage times include synchronisation and are for proportions only.
