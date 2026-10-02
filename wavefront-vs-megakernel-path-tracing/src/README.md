# Sources

| File | Role |
|---|---|
| `main.cpp` | the `wf render` command: parse arguments, build the scene, run the selected arm with warm-up and repetitions, write images and `report.json` |
| `config.cpp` | command-line parsing and validation; every invalid input names the flag |
| `scene.cpp`, `scenes.cpp`, `obj_loader.cpp` | scene validation and finalisation (lights, BVH, camera rig), the procedural scenes, the OBJ loader with file and line in every error |
| `bvh_build.cpp` | binned-SAH BVH construction and structural validation |
| `cpu_renderer.cpp`, `film.cpp` | multi-threaded CPU renderer, film accumulation, tone mapping, comparison statistics |
| `image_io.cpp` | PNG writer (stored deflate), PFM reader and writer, FNV-1a hashing |
| `json.cpp`, `report.cpp` | a small JSON parser and the versioned `report.json` writer and validator |
| `bench.cpp` | the benchmark driver: runs the three arms, fills the result, applies the verdict rule, serialises and validates the JSON |
| `gpu/gpu_renderer.cu` | `GpuRenderer`: device buffers, scene upload, progressive API (`reset`, `run_passes`, `download_mean`) |
| `gpu/megakernel.cu` | the megakernel |
| `gpu/wavefront.cu`, `gpu/queues.cu` | the wavefront engine and the queue, scan and compaction primitives |
| `gpu/engine.cuh`, `gpu/queues.cuh` | internal interfaces between the renderer and the two engines |
| `viewer/viewer.cpp` | the Vulkan viewer |

Headers are in `../include/wf`. The estimator (`path_core.hpp`), math, RNG, scene and BVH traversal are header-only because they run on both host and device. See [`../docs/implementation`](../docs/implementation/README.md).
