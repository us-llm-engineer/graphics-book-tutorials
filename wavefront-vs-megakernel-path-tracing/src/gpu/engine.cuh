// Interface between GpuRenderer (gpu_renderer.cu) and the two engines.
// Both engines accumulate through film_add in the same per-pixel sample order, so their images
// are bitwise identical when the per-path math is identical.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "wf/path_core.hpp"
#include "wf/renderer.hpp"

namespace wf::gpu {

struct DeviceScene {
    SceneView view;          // all pointers are device pointers
    uint64_t bytes = 0;      // device bytes owned by the scene upload
};

// Per-pixel sums over samples. One path owns a pixel in a pass; there are no atomics.
struct DeviceFilm {
    float* rgb = nullptr;    // 3 * pixels
    float* lum = nullptr;    // pixels  (sum of luminance)
    float* lum2 = nullptr;   // pixels  (sum of squared luminance)
    int pixels = 0;
};

__device__ __forceinline__ void film_add(const DeviceFilm& f, int pixel, Vec3 L) {
    f.rgb[3 * pixel + 0] += L.x;
    f.rgb[3 * pixel + 1] += L.y;
    f.rgb[3 * pixel + 2] += L.z;
    float l = luminance(L);
    f.lum[pixel] += l;
    f.lum2[pixel] += l * l;
}

// Counters shared by both engines; zeroed by GpuRenderer before each run.
struct DeviceCounters {
    unsigned long long extension_rays;   // only incremented when count_rays
    unsigned long long shadow_rays;      // only incremented when count_rays
    uint32_t invalid_hits;               // hit with prim out of range or non-finite t
    uint32_t overflows;                  // dropped queue pushes
    uint32_t high_water[5];              // lambert, metal, dielectric, shadow, active (wavefront)
};

// Device-side check used by both engines after every intersect(): returns true if valid.
__device__ __forceinline__ bool hit_is_valid(const SceneView& s, const HitRecord& h) {
    if (h.prim < 0) return true;
    int total = s.n_spheres + s.n_triangles;
    return h.prim < total && is_finite1(h.t) && h.t > 0.0f;
}

// One pass = one sample for every pixel.
void run_megakernel_pass(const DeviceScene& scene, const DeviceFilm& film, DeviceCounters* counters,
                         uint32_t sample, uint32_t seed, int max_depth, bool count_rays,
                         cudaStream_t stream);

class WavefrontEngine {
public:
    WavefrontEngine(const DeviceScene& scene, int width, int height);   // allocates path pool
    ~WavefrontEngine();
    WavefrontEngine(const WavefrontEngine&) = delete;
    WavefrontEngine& operator=(const WavefrontEngine&) = delete;

    // Renders sample `sample` for all pixels into `film`. If stage_timing, per-stage events are
    // summed into stages (synchronising); counters->high_water is max-updated each pass.
    void run_pass(const DeviceFilm& film, DeviceCounters* counters, uint32_t sample, uint32_t seed,
                  int max_depth, bool count_rays, bool stage_timing, StageTimes* stages,
                  cudaStream_t stream);
    uint64_t bytes() const;
private:
    struct Impl;
    Impl* impl_;
};

}  // namespace wf::gpu
