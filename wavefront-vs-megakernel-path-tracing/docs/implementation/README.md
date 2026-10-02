# Implementation notes

In-depth notes on how the code works, with pointers into the source. The measurements behind the claims are in [`../../results`](../../results/README.md).

| Note | Subject |
|---|---|
| [`design.md`](design.md) | the original design and evaluation contract |
| [`estimator.md`](estimator.md) | the shared path-tracing estimator and its numerical choices |
| [`wavefront-engine.md`](wavefront-engine.md) | path state, queues, scan and compaction, failure handling |
| [`bvh.md`](bvh.md) | binned-SAH build, skip links, stackless traversal |
| [`rng.md`](rng.md) | the counter-based random number generator |
| [`benchmarking.md`](benchmarking.md) | protocol, verdict rule and report schema |
| [`profiling.md`](profiling.md) | Nsight Compute counters, roofline derivation, sanitizers |
| [`viewer.md`](viewer.md) | the Vulkan viewer |
| [`analysis.md`](analysis.md) | the nine statistical studies and how to read the figures |

The short technical report is [`../REPORT.md`](../REPORT.md).
