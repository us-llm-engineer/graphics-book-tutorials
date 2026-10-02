# Counter-based random numbers

Source: [`include/wf/rng.hpp`](../../include/wf/rng.hpp).

## Design

Every draw is a pure function of five inputs: `rng(key, depth, dim)` with `key = make_path_key(seed, pixel, sample)`. Nothing depends on how many numbers a thread consumed before, so the megakernel, the wavefront and the CPU reference draw identical values for identical paths regardless of execution order. This follows the counter-based generator idea of Salmon et al. [9].

## Construction

- `hash_u32` is the "lowbias32" integer mixer (two multiply-xorshift rounds).
- `make_path_key` runs two independent hash chains over (seed, pixel, sample) and returns two 32-bit words `k0`, `k1`, so a key carries about 64 bits and distinct paths virtually never share a stream.
- `rng(key, depth, dim)` hashes `depth * 16 + dim` together with the key and keeps the top 24 bits, giving a float in `[0, 1)` with 24-bit resolution. The estimator uses dimensions for pixel jitter, light choice, light point, BSDF sample, roulette and dielectric choice, at most 16 per bounce.

## Quality tests

Figure 9 and `results/stats/stats.json` (section `rng`):

- **Avalanche:** flipping any of the 96 input bits flips each of the 64 output bits with probability close to 1/2 (mean 0.5000; worst cell 4.5 standard errors over 6144 cells, Bonferroni p about 0.05, borderline and consistent with chance).
- **Uniformity:** Kolmogorov-Smirnov D of 2^20 draws is 0.59e-3 against a 5% critical value of 1.33e-3.
- **Serial test:** a 16x16 chi-square on consecutive-pixel pairs stays inside the 95% band of chi-square(255) for all 32 seeds tested.
- **Autocorrelation:** lags 1 to 64 stay near the 95% band of plus or minus 1.96/sqrt(N); the largest magnitude is 0.0022 against a band of 0.0019, as expected for 64 lags.
