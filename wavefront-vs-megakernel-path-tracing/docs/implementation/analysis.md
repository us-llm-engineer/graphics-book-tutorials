# The nine analyses

[`tools/wfstats.cpp`](../../tools/wfstats.cpp) measures, [`tools/svgstats.cpp`](../../tools/svgstats.cpp) draws; all data are written to `results/stats/stats.json` and the figures to `docs/figures/analysis/`. Figure 7 is the scaling law; the three benchmark charts are drawn by `tools/svgplot.cpp`.

| # | Question | Method | How to read it |
|---|---|---|---|
| 1 Convergence | Does error fall as $N^{-1/2}$? | luminance RMSE of 6 seeds per N against a 16384-sample reference, reference noise removed; log-log fit | slope near -0.5 from N >= 16; the predicted line uses only the renderer's own variance estimate |
| 2 Bias | Is the estimator unbiased through Russian roulette and the GPU arms? | furnace scene, depth 1 to 16, three renderers against $L_e\sum_{k\le D}\rho^k$ | residuals inside the 95% band; the arms share random numbers so residuals coincide |
| 3 Variance | Where does the noise live? | Lorenz curve, Gini coefficient and kurtosis of per-pixel variance | Gini near 1: a few pixels carry most of the variance |
| 4 Survival | How quickly do paths die? | extension-ray counts at depth caps 0 to 16; $S_k=(E_k-E_{k-1})/N$; geometric tail fit | tail ratio per scene; points with fewer than 30 counted rays are dropped |
| 5 SIMT efficiency | Does the material mix explain lane efficiency? | iid model and 16x2-tile model of primary-hit materials against Nsight lane efficiency | model is an upper bound; traversal divergence explains the rest |
| 6 Roofline | Compute- or memory-limited? | FLOP, bytes and time per kernel against counter-derived roofs | position relative to the ridge point |
| 7 Scaling | What does resolution do to the speedup? | render time at 11 sizes, linear fit $\alpha+\beta P$, speedup limit | limits below 1 mean more pixels cannot rescue the wavefront |
| 8 BVH | Are the SAH splits good? | leaf depth histogram, child-to-parent area ratio per depth, SAH cost | ratio near 0.5; depth a few levels above $\log_2$ of the leaf count |
| 9 RNG | Is the generator good enough? | avalanche matrix, serial chi-square over 32 seeds, autocorrelation, KS | all within chance, worst avalanche cell borderline |

## Reproduce

```bash
build/wfstats results/stats/stats.json            # all sections; pass section names to run a subset
bash scripts/profile_stats.sh                     # Nsight Compute counters for Figures 5 and 6
for f in results/profiles/stats/*.csv; do build/ncusum "$f" --out "${f%.csv}.summary.json"; done
build/svgstats docs/figures/analysis results/stats/stats.json results/profiles/stats
```

Keep the GPU otherwise idle while `wfstats` runs: the scaling section measures wall-clock time.

## Caveats

Primary-hit materials only (secondary bounces decohere further); one frame at 1 sample per pixel for the counters; stage times include synchronisation; the convergence reference is itself a Monte Carlo estimate (its noise floor is subtracted).
