# Wavefront engine

Sources: [`src/gpu/wavefront.cu`](../../src/gpu/wavefront.cu), [`queues.cu`](../../src/gpu/queues.cu), [`queues.cuh`](../../src/gpu/queues.cuh), [`engine.cuh`](../../src/gpu/engine.cuh).

## One pass

One pass renders one sample for all `W*H` pixels. Path state is stored as structure-of-arrays (about 150 bytes per path by design, about 140 MB at 1280x720); the measured peak device memory of the whole renderer is about 490 to 504 MB. Per bounce, the host launches:

| Kernel | Work |
|---|---|
| `kernel_extend` | traverse the BVH for every active path, store a hit record (t, primitive, barycentrics) |
| `kernel_classify` | handle miss and emission, terminate paths, push the survivors onto the Lambertian, metal or dielectric queue |
| `kernel_shade_lambert`, `kernel_shade_metal`, `kernel_shade_dielectric` | run on the matching queue; write the next ray; Lambertian also writes a shadow ray and reserves a slot in the shadow queue |
| `kernel_shadow` | any-hit traversal on the shadow queue; unoccluded rays add their contribution to the owning path |
| `kernel_mark_alive`, `make_indicator_kernel`, `scan_pass1/2/3_kernel`, `compact_kernel` | build the next active list from per-path alive flags |
| `kernel_update_counters` | per-queue high-water marks and overflow counts |

Before the first bounce `kernel_init_pixel_indices_impl` and `kernel_init_paths` set up the path pool; after the last, `kernel_finish` adds each path's radiance to the film.

## Queues

A queue is an index array plus a counter. `queue_push` is warp-aggregated: the lanes that want a slot ballot, one lane performs a single atomic add for the whole group and each lane derives its position from the prefix of the ballot, so contention is one atomic per warp per queue. The classify kernel calls `queue_push` once per queue with a per-lane predicate (`material == q`); calling it with different queues across lanes in one call would read out of bounds and is avoided by construction.

Capacity equals the path count by default, so overflow cannot happen. A test hook (`wavefront_set_test_queue_capacity`) shrinks the capacity: pushes beyond it are dropped, counted in `overflows`, and the affected path is ended instead of corrupting memory. The result is reported as an overflow, never as a silent error.

## Scan and compaction

`scan_exclusive_u32` and `compact_u32` are written from scratch (no CUB). The scan has three passes: (1) each block scans a tile exclusively with full-mask warp shuffles and writes its tile total, (2) one block scans the tile totals, looping over 1024-entry tiles with a running carry, (3) each block adds its offset. `compact_u32` scans the indicator `(flag != 0)`, writes the selected indices in stable order, and returns the count in a device scalar. `n == 0` succeeds with count 0 and `n` above the workspace capacity is rejected with `cudaErrorInvalidValue`.

## Costs that the profiler shows

The host reads the active count back after every bounce, so each bounce includes a host synchronisation; per-path state is read and written through global memory by every stage. In the divergence scene the classify and shading kernels run at 41% to 56% DRAM throughput while the megakernel keeps state in registers (1%); traversal (`kernel_extend`) is 38% of the wavefront's kernel time and is as divergent as in the megakernel. See [`profiling.md`](profiling.md) and Figures 5 to 7.
