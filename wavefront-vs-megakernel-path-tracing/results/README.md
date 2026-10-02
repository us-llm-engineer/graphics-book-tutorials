# Results

Raw measurements behind every number in the READMEs. Nothing here is edited by hand.

| Path | Produced by | Contents |
|---|---|---|
| `bench-<scene>.json`, `bench-<scene>.run.txt` | `scripts/run_benchmarks.sh` | benchmark JSON for the three scenes and the command that produced each |
| `stage-times.run.txt` | an instrumented wavefront run (`RenderParams::stage_timing`) | per-stage times, spp 4, for the stage chart |
| `stats/stats.json` | `build/wfstats` | convergence, furnace, variance, survival, divergence, scaling, BVH and RNG data for Figures 1 to 9 |
| `profiles/*.csv`, `profiles/*summary.json` | `scripts/profile_ncu.sh`, `tools/ncusum` | first profiling pass (divergence scene, 1 spp) |
| `profiles/stats/*.csv`, `*.summary.json`, `*.command.txt` | `scripts/profile_stats.sh`, `tools/ncusum` | counters for three scenes and both arms, plus the device peaks (`device.*`); the command is stored beside each CSV |
| `sanitizers.json`, `sanitizers*.run.txt` | `scripts/run_sanitizers.sh` | memcheck and ASan rows with exit codes; the ASan log of a GPU mode shows the driver abort |

Reading the benchmark JSON: [`../docs/implementation/benchmarking.md`](../docs/implementation/benchmarking.md). The Nsight CSVs use raw metric ids and base units (nanoseconds, bytes, hertz).
