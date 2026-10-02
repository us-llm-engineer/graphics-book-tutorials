// Megakernel GPU renderer: one thread per pixel, traces complete path per sample.
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "engine.cuh"
#include "wf/path_core.hpp"

namespace wf::gpu {

// One thread per pixel: traces one complete path (sample) for that pixel.
__global__ void megakernel_kernel(DeviceScene scene, DeviceFilm film,
                                   DeviceCounters* counters, uint32_t sample, uint32_t seed,
                                   int max_depth, bool count_rays) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= scene.view.width || py >= scene.view.height) return;

    int pixel = py * scene.view.width + px;

    // Initialize path state for this pixel and sample.
    wf::PathState ps = wf::init_path(scene.view, px, py, sample, seed);

    // Trace the path.
    for (;;) {
        // Trace ray to find hit.
        wf::HitRecord h;
        wf::intersect(scene.view, ps.ray, &h);

        // Count extension ray if requested.
        if (count_rays) atomicAdd((unsigned long long*)&counters->extension_rays, 1ull);

        // Validate the hit.
        if (!wf::gpu::hit_is_valid(scene.view, h)) {
            atomicAdd(&counters->invalid_hits, 1u);
            // Continue with miss (treat as invalid hit == miss).
            ps.L += ps.throughput * scene.view.env;
            break;
        }

        // Classify the hit: process emission, check termination, get material type.
        wf::Surface f;
        int mat = wf::classify(scene.view, ps, h, max_depth, &f);
        if (mat < 0) break;  // Path terminated.

        // Shade based on material type.
        bool alive = false;
        if (mat == wf::kLambert) {
            // Lambertian: next-event estimation + scatter.
            wf::Ray shadow_ray;
            wf::Vec3 shadow_contrib;
            int has_shadow = 0;

            alive = wf::shade_lambert(scene.view, ps, f, &shadow_ray, &shadow_contrib, &has_shadow);

            // Trace shadow ray if present.
            if (has_shadow) {
                wf::HitRecord shadow_hit;
                bool shadow_blocked = wf::intersect(scene.view, shadow_ray, &shadow_hit, true);

                if (count_rays) atomicAdd((unsigned long long*)&counters->shadow_rays, 1ull);

                // If shadow ray doesn't hit anything (or hits beyond the light), add contribution.
                if (!shadow_blocked) {
                    ps.L += shadow_contrib;
                }
            }
        } else if (mat == wf::kMetal) {
            alive = wf::shade_metal(scene.view, ps, f);
        } else if (mat == wf::kDielectric) {
            alive = wf::shade_dielectric(scene.view, ps, f);
        }

        if (!alive) break;
    }

    // Accumulate result into film.
    wf::gpu::film_add(film, pixel, ps.L);
}

void run_megakernel_pass(const DeviceScene& scene, const DeviceFilm& film, DeviceCounters* counters,
                         uint32_t sample, uint32_t seed, int max_depth, bool count_rays,
                         cudaStream_t stream) {
    // Launch one thread per pixel.
    // Use 2D block layout for better cache locality.
    const int block_x = 16, block_y = 16;
    int grid_x = (scene.view.width + block_x - 1) / block_x;
    int grid_y = (scene.view.height + block_y - 1) / block_y;

    megakernel_kernel<<<dim3(grid_x, grid_y), dim3(block_x, block_y), 0, stream>>>
        (scene, film, counters, sample, seed, max_depth, count_rays);

    // Check for kernel launch errors.
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("megakernel launch failed: ") + cudaGetErrorString(e));
    }

    // Synchronize to catch device-side errors and ensure kernel completes.
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("megakernel sync failed: ") + cudaGetErrorString(e));
    }
}

}  // namespace wf::gpu
