# Tools

All tools are built by CMake next to the renderer. Exit status is 0 on success, 2 on an error (a message on stderr names the problem and no partial output is written).

| Tool | Usage | Purpose |
|---|---|---|
| `wfbench` | `wfbench --scene S [--width --height --spp --reps --warmup --max-depth --seed --cpu-spp] --out result.json` | runs CPU, megakernel and wavefront, checks quality, applies the verdict rule, writes the benchmark JSON (exit 0 also for a weakened verdict) |
| `imgdiff` | `imgdiff a.pfm b.pfm [--tol T]` | compares two images and reports the maximum difference; exit status reflects the tolerance |
| `ncusum` | `ncusum profile.csv [--out summary.json]` | summarises an Nsight Compute CSV: per-kernel metric means (text-valued metrics are skipped, `n/a` becomes null) and a summary object |
| `wfstats` | `wfstats out.json [section ...]` | measures convergence, furnace bias, variance concentration, path survival, material divergence, scaling, BVH and RNG statistics from real renders |
| `svgstats` | `svgstats out-dir stats.json ncu-dir` | draws the nine analysis figures |
| `svgplot` | `svgplot out-dir bench-a.json bench-b.json bench-c.json [stage-times.txt]` | draws the three benchmark charts; refuses results that are not the full protocol |
| `wfgif` | `wfgif out.gif [--width --height --mode --final-png-prefix P]` | renders progressive frames of the three scenes and writes an animated GIF (own LZW encoder, fixed 6x7x6 colour cube with error diffusion) |

The figures use CSS variables for light and dark themes, direct labels and hover titles on every mark.
