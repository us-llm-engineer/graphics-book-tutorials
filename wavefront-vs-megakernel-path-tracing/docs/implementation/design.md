# Design: wavefront versus megakernel path tracing

This document is binding for implementation and evaluation. It fixes the contracts before any measurement.

## 1. Question and falsifier
On a GTX 1650 Ti (CC 7.5, 4 GB) at 1280x720, fixed scene, seed, samples per pixel (spp), max depth and compiler flags, is the queue-based wavefront tracer at least 30% faster (median of five warmed runs) than the megakernel on the material-divergence scene, with equal image quality and at most 3.5 GB of device memory? Anything else is reported as a negative result.

## 2. One estimator, three arms
`include/wf/path_core.hpp` holds every estimator step as small inline functions compiled for host and device. The CPU renderer, the megakernel and the wavefront stages all call these functions; no arm contains its own copy of shading logic. Consequences:
- All random numbers come from `wf::rng(key, depth, dim)`, a pure function of (seed, pixel, sample, depth, dimension). Order of execution cannot change a path.
- CUDA is compiled with `--fmad=false`, the host with `-ffp-contract=off`, so the megakernel and wavefront execute the same float operations per path and must produce identical pixels.
- A pass renders exactly one sample per pixel; pixel `i` accumulates sample `s` into `accum[i]` only after pass `s-1` finished. There is no atomic floating-point addition anywhere.

## 3. Scene model (`scene.hpp`)
- Spheres and indexed triangles in one primitive space: id `p < n_spheres` is a sphere, otherwise triangle `p - n_spheres`.
- Materials: Lambertian, metal (tinted perfect mirror), dielectric (exact Fresnel, Snell, TIR), emissive (pure emitter, absorbs). Any material may also carry `emission`; a Lambertian with emission is allowed (used by the furnace oracle).
- Lights are all primitives whose emission is non-zero; light selection is uniform over lights, a point is sampled uniformly by area.
- Constant environment radiance `env` (zero by default), counted with weight 1 when a ray escapes.
- Scenes at 1280x720: `cornell`, `divergence`, `trianglehv` (triangle-heavy, 100k+ triangles from a procedural heightfield and spheres), plus `furnace` for the oracle and OBJ files via `--scene path.obj`.
- `divergence`: a grid of spheres and a ground, with at least four materials interleaved in space (Lambertian in 4 albedos, metal, dielectric, emissive) so neighbouring pixels hit different materials.

## 4. Estimator (`path_core.hpp`)
Per path: `throughput`, `L`, `depth`, `specular` flag, `prev_pdf` (solid-angle pdf of the last BSDF sample, 0 for delta), `prev_p`.
Per bounce, in this order:
1. Find vertex (`intersect`). Miss: `L += throughput * env`, terminate.
2. Emission: `Le` at the hit. Weight 1 if `depth == 0` or `specular`; else power heuristic (beta 2) of `prev_pdf` against the light pdf for that hit. `L += throughput * Le * w`.
3. Terminate if the material is purely emissive or `depth == max_depth`.
4. For Lambertian only, next-event estimation: pick a light, sample a point, shadow ray; contribution `throughput * (albedo/pi) * cos * Le * w_light / p_light`, `w_light` the power heuristic of light pdf against cosine pdf. Added to `L` only if unoccluded.
5. Sample the BSDF: Lambertian cosine hemisphere (`throughput *= albedo`), metal mirror, dielectric reflect-or-refract by Fresnel probability. Set `specular`, `prev_pdf`, `prev_p`.
6. Russian roulette for `depth >= 3`: `q = clamp(max component of throughput, 0.05, 0.95)`, survive if `u < q`, `throughput /= q`.
7. Offset the new ray origin by `1e-4` along the facing normal; `tmin = 1e-4`.

## 5. BVH and traversal
- CPU-built binary BVH (binned SAH, leaf size at most 4) stored in preorder: the left child of node `N` is `N+1`; each node stores `skip`, the node reached on a miss or after a leaf; the last skip is `-1`. Traversal is the stackless loop of the stackless-BVH paper's Algorithm 1 (visit node, descend left on hit, otherwise jump to `skip`).
- Brute force over all primitives is the second oracle; `intersect_bruteforce` and `intersect` must agree on `prim` and `t` within 1e-5 relative for 100000 random rays on every test scene.

## 6. Wavefront pipeline (`wavefront.cu`)
One pass = one sample for all `W*H` pixels. Path state is structure-of-arrays (about 150 bytes per path, about 140 MB at 1280x720). Per bounce:
1. `extend`: traverse for every active path, store `HitRecord` (t, prim, u, v).
2. `classify`: handle miss and emission, terminate or push the path index to the Lambertian, metal or dielectric queue (warp-aggregated atomic reservation).
3. `shade_lambertian`, `shade_metal`, `shade_dielectric`: run on their queue; write the next ray; Lambertian also writes a shadow ray plus contribution and reserves a slot in the shadow queue.
4. `shadow`: any-hit traversal on the shadow queue; unoccluded rays add their contribution to the owning path's `L`.
5. `compact`: from per-path alive flags, an exclusive prefix scan (block scan plus scan of block sums, written from scratch) builds the next active list in stable order.
Queue capacity equals the path count, so a queue cannot overflow; `high_water` per queue is recorded. After the last bounce, `finish` adds each path's `L` to `accum[pixel]`.
Counting-sort by material is performed by the three material queues; there is no separate radix sort and no CUB.

## 7. Megakernel (`megakernel.cu`)
One kernel per pass, one thread per pixel, loop over bounces calling the same `path_core` functions with the shadow ray traced immediately. Same pass and accumulation structure as the wavefront.

## 8. Image quality and tolerance (fixed before measurement)
- Film: per pixel, accumulate `sum(L)` and `sum(lum(L)^2)` over samples; the image is `sum(L)/spp` followed by exposure and tone mapping for PNG; `.pfm` stores the linear HDR mean.
- Pre-registered pass criteria:
  - megakernel vs wavefront (same device, same seed): maximum per-pixel absolute difference `<= 1e-5` relative to the pixel value (identical expected).
  - CPU vs GPU: at least 99.5% of pixels within three combined standard errors of the pixel mean (`se` from per-pixel luminance variance over spp), and image-mean relative difference `<= 0.5%`.
  - furnace oracle: image mean within three standard errors of `Le / (1 - rho)`.
  - zero non-finite pixels, zero queue overflows, zero invalid intersections (hit `t` finite, `prim` in range).
- A failure of any criterion is reported as a failure; the criteria are not changed after measurement.

## 9. Evaluation protocol
- Modes: `cpu` (reduced spp for the CPU arm, stated in the report), `megakernel`, `wavefront`.
- Scenes: `cornell`, `divergence`, `trianglehv`, 1280x720, spp 64, max depth 8, seed 1234 (spp and seed fixed in `bench/protocol.json`).
- Per condition: 2 untimed warm-up runs, then 5 timed runs; timing with CUDA events around the whole render (all passes); report min, median, p10, p90, max, samples per second and rays per second (rays = extension plus shadow rays traced, counted by a device counter in a separate, untimed counting run).
- Peak device memory from `cudaMemGetInfo` before allocation and after allocation (peak sampled at end of each pass); must be at most 3.5 GB.
- Speedup = median(megakernel) / median(wavefront); the hypothesis requires `>= 1.30`.
- Per-stage wavefront timings (events per stage summed over a run) are collected in a separate instrumented run and never mixed with the benchmark timings.
- GPU clocks and power state are recorded (`nvidia-smi` sample before and after); clock locking is attempted and its success or failure reported.

## 10. Profiling
Nsight Compute on one representative pass for both modes: occupancy, registers per thread, warp execution efficiency, issue-slot utilisation, stall reasons, DRAM and L2 throughput. Raw `.ncu-rep` files or `--csv` exports are stored under `results/profiles/`. If counters are unavailable in WSL, the exact failing command and message are stored instead.

## 11. Module map and ownership
| Module | Files | Contract |
|---|---|---|
| math, rng | `include/wf/math.hpp`, `rng.hpp` | header-only, host and device |
| scene, camera, materials | `scene.hpp`, `src/scenes.cpp`, `src/obj_loader.cpp` | procedural scenes, OBJ loader, validation errors |
| BVH | `bvh.hpp`, `src/bvh_build.cpp` | build, skip links, brute-force oracle |
| estimator | `path_core.hpp` | the only shading logic |
| film, I/O | `film.hpp`, `src/image_io.cpp` | tone map, PNG (stored deflate), PFM, FNV-1a hash |
| config, report | `config.hpp`, `report.hpp`, `src/config.cpp`, `src/report.cpp` | CLI parsing, versioned `report.json` |
| CPU renderer | `renderer.hpp`, `src/cpu_renderer.cpp` | multi-threaded, same estimator |
| GPU queues | `src/gpu/queues.cu` | scan, compaction, aggregated reservation |
| megakernel, wavefront | `src/gpu/megakernel.cu`, `wavefront.cu` | as above |
| CLI | `src/main.cpp` | `render` command |
| benchmark | `tools/wfbench.cpp`, `tools/imgdiff.cpp`, `tools/svgplot.cpp` | protocol runner, image comparison, charts |
| viewer | `src/viewer/viewer.cpp` | GLFW and Vulkan, host-staged texture |

## 12. Falsifying experiments
- Queue failure: shrink queue capacity in a test build; the run must report overflow (counter) rather than corrupt data.
- Invalid output: poison hit records; validators must flag them.
- Divergence control: the `cornell` scene (few materials) is expected to show a smaller wavefront gain; a larger gain there than on `divergence` would contradict the divergence explanation and is reported.
