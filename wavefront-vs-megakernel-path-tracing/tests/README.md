# Tests

`ctest --test-dir build --output-on-failure` runs every executable as its own process. Run from the project directory: the suites use relative paths.

## Executables

| Group | Executables | Covers |
|---|---|---|
| Unit tests (own minimal framework, `test_framework.hpp`) | `test_math_rng`, `test_intersect`, `test_bvh`, `test_scene`, `test_obj`, `test_config`, `test_film`, `test_image_io`, `test_report`, `test_path_core`, `test_cpu_render`, `test_gpu_queues`, `test_gpu_render` | math and RNG, ray-primitive tests, BVH against a brute-force oracle, scenes, OBJ loading, CLI parsing, film and tone mapping, image I/O, report schema, estimator steps, CPU renderer, GPU scan and queues, GPU renderer |
| Catch2 suites (`tests/catch`) | `catch_assets_loading`, `catch_scene_bvh`, `catch_cli_artifacts`, `catch_cpu_render`, `catch_gpu_megakernel`, `catch_gpu_primitives`, `catch_wavefront`, `catch_wavefront_failure`, `catch_bench_run`, `catch_bench_stats`, `catch_ncusum`, `catch_scripts` | invalid input and failure mapping, estimator oracles (furnace, naive cross-check), megakernel and wavefront correctness and validity, queue overflow behaviour and memory, the benchmark rule and JSON schema, `ncusum` parsing, the scripts |

GPU suites need a CUDA device and are skipped or fail loudly without one. The Catch2 suites are only built when Catch2 3 is found (`-DCMAKE_PREFIX_PATH=<prefix>`).

## Environment

Suites that start other executables read `WF_BIN`, `WFBENCH_BIN`, `IMGDIFF_BIN`, `NCUSUM_BIN` and `WF_BUILD_DIR`; defaults point into `build/`.

## Caveats

`catch_scripts` runs the benchmark, profiling and sanitizer scripts in quick mode and **rewrites files under `results/`** (it clears `results/profiles/`). Commit or back up `results/` before running it. The sanitizer scripts expect a CUDA device; the ASan row is recorded as not run when no ASan build exists.
