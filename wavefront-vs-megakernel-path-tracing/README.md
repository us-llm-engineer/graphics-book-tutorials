# Wavefront versus megakernel path tracing

A C++20/CUDA study of two ways to structure a GPU path tracer, built around one shared estimator so that the comparison is about execution structure only. This README is the project reference; the repository [README](../README.md) shows the results and figures, and [`docs/implementation`](docs/implementation/README.md) holds the in-depth notes.

## The question

On a GTX 1650 Ti (compute capability 7.5, 4 GB) at 1280x720 with fixed scene, seed, samples per pixel (64), maximum depth (8) and compiler flags: is a queue-based wavefront tracer at least 30% faster (median of five warmed-up runs) than a megakernel on the material-divergent scene, with equal image quality and at most 3.5 GB of device memory? The rule is encoded in [`include/wf/bench.hpp`](include/wf/bench.hpp) and applied by the benchmark driver, so a negative outcome is reported as one.

## Three arms, one estimator

| Arm | Entry point | Structure |
|---|---|---|
| CPU reference | `render_cpu` in `src/cpu_renderer.cpp` | multi-threaded, one path per task |
| Megakernel | `src/gpu/megakernel.cu` | one kernel per pass, one thread per pixel, whole path in a loop, shadow ray traced in place |
| Wavefront | `src/gpu/wavefront.cu` | per-bounce stages: extend, classify, shade per material, shadow, compact |

All three call the inline functions in [`include/wf/path_core.hpp`](include/wf/path_core.hpp); random numbers come from a counter-based generator, so a draw depends only on (seed, pixel, sample, depth, dimension). The GPU code is compiled with `--fmad=false` and the host code with `-ffp-contract=off`; with that, the megakernel and the wavefront produce **bitwise identical images**, which the benchmark verifies on every scene.

## Layout

| Folder | Contents | Notes |
|---|---|---|
| [`include/wf`](include/wf) | estimator, math, RNG, scene model, BVH, renderer and report interfaces, shared overlay helpers | header-only where it must run on the device |
| [`src`](src/README.md) | scene and BVH build, CPU renderer, CLI, benchmark driver, CUDA arms, viewer | |
| [`tools`](tools/README.md) | `wfbench`, `imgdiff`, `ncusum`, `wfstats`, `svgstats`, `svgplot`, `wfgif` | command-line utilities |
| [`scripts`](scripts/README.md) | benchmark, profiling, sanitizer and viewer wrappers | bash |
| [`tests`](tests/README.md) | unit tests and Catch2 suites | |
| [`results`](results/README.md) | raw benchmark JSON, Nsight Compute exports, statistics, sanitizer logs | evidence for every number in the READMEs |
| [`docs`](docs) | implementation notes, figures, media, technical report | |
| [`bench`](bench) | `protocol.json`, the fixed benchmark parameters | |

## Build

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=<catch2-install-prefix>
cmake --build build -j
```

Options: `-DWF_SANITIZE=ON` builds host code with AddressSanitizer and UBSan (CPU mode only; the GPU modes abort inside some drivers), `-DWF_BUILD_VIEWER=OFF` skips the Vulkan viewer. CUDA architecture 75 is set in `CMakeLists.txt`; change `CMAKE_CUDA_ARCHITECTURES` for another GPU. The viewer needs GLFW and the Vulkan loader and is skipped with a message if either is missing; the Catch2 suites are skipped if Catch2 3 is not found.

## Command line

```
wf render --scene {cornell|divergence|trianglehv|furnace|<file.obj>} --width W --height H --spp N
          --seed S --mode {cpu|megakernel|wavefront} [--max-depth D] [--threads T]
          [--warmup N --reps N] [--out PREFIX] [--report report.json]
```

`--out PREFIX` writes `PREFIX.png` (tone mapped) and `PREFIX.pfm` (linear HDR). `--report` writes a versioned `report.json` with the scene hash, device and driver versions, timings, samples and rays per second, peak VRAM, queue high-water marks and output hashes. Invalid arguments fail with a message naming the flag and write nothing.

`wfbench` runs the full comparison for one scene and writes the benchmark JSON; `scripts/run_benchmarks.sh` runs it for every scene in `bench/protocol.json`.

## Tests

`ctest --test-dir build` runs every suite as its own process. The suites cover: math and RNG, ray-primitive intersection, BVH structure and traversal against a brute-force oracle, scene and OBJ validation, CLI parsing and report schema, the CPU estimator (furnace oracle, naive-path-tracer cross-check, MIS), GPU scan, compaction and queue primitives, both GPU arms (bitwise agreement, validity counters, overflow behaviour, memory), the benchmark rule and statistics, `ncusum` parsing, and the scripts. GPU suites need a CUDA device. See [`tests/README.md`](tests/README.md).

## Evidence map

Every quantitative statement in the READMEs comes from a file under `results/`:

| Statement | File |
|---|---|
| speedups, medians, quality gates, memory, queue high-water marks | `results/bench-<scene>.json` |
| per-stage wavefront times | `results/stage-times.run.txt` |
| convergence, bias, variance, survival, scaling, BVH, RNG statistics | `results/stats/stats.json` |
| kernel counters (occupancy, efficiency, throughput, FLOP and byte counts) | `results/profiles/stats/*.csv` and `*.summary.json` |
| first profiling pass (divergence scene) | `results/profiles/` |
| sanitizer runs | `results/sanitizers.json`, `results/sanitizers*.run.txt` |

## Limitations

See the repository README. In short: one GPU and driver, three scenes, clocks not locked, host-synchronised wavefront design, GPU modes not covered by ASan/UBSan.
