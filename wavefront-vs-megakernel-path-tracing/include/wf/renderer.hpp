// Renderer entry points for the three arms. All use wf::trace_path / the path_core stages.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "wf/film.hpp"
#include "wf/scene.hpp"

namespace wf {

struct RenderParams {
    int spp = 64;
    uint32_t seed = 1234;
    int max_depth = 8;
    int threads = 0;              // CPU only
    bool count_rays = false;      // GPU: also count extension and shadow rays (adds overhead)
    bool stage_timing = false;    // wavefront: time each stage with events (adds overhead)
    int queue_capacity = 0;       // wavefront test hook: 0 = path count (cannot overflow); a smaller
                                  // value must make overflows COUNTED (stats.queue_overflows) and
                                  // never write outside the queues
};

struct StageTimes {  // milliseconds summed over a render, wavefront only
    double extend = 0, classify = 0, shade_lambert = 0, shade_metal = 0, shade_dielectric = 0,
           shadow = 0, compact = 0, other = 0;
};

struct RenderStats {
    double render_ms = 0.0;   // GPU: events around all passes; CPU: wall clock
    double total_ms = 0.0;    // render_ms + result download
    uint64_t extension_rays = 0, shadow_rays = 0;   // only if count_rays
    long long queue_high_water[5] = {0, 0, 0, 0, 0};  // lambert, metal, dielectric, shadow, active
    long long queue_overflows = 0;
    long long invalid_hits = 0;       // hit with non-finite t, or prim out of range
    uint64_t peak_vram_bytes = 0;     // device memory used by this renderer (cudaMemGetInfo delta)
    StageTimes stages;
};

RenderResult render_cpu(const Scene& scene, const RenderParams& p, RenderStats* stats);

enum class GpuMode { Megakernel, Wavefront };

// Owns device buffers for one scene. The scene is uploaded in the constructor.
class GpuRenderer {
public:
    GpuRenderer(const Scene& scene, GpuMode mode);   // throws std::runtime_error on CUDA error
    ~GpuRenderer();
    GpuRenderer(const GpuRenderer&) = delete;
    GpuRenderer& operator=(const GpuRenderer&) = delete;

    // reset(); run_passes(0, spp); download(spp). Result averaged over spp samples.
    RenderResult render(const RenderParams& p, RenderStats* stats);

    // Progressive API used by the viewer: zero the accumulators, then render samples
    // [first_sample, first_sample+count) and add them. No atomics: one path per pixel per pass.
    void reset();
    void run_passes(int first_sample, int count, const RenderParams& p, RenderStats* stats);
    RenderResult download(int spp_done);
    // Mean radiance only (no variance), cheaper; used by the viewer.
    void download_mean(int spp_done, std::vector<float>* rgb);

    uint64_t device_bytes() const;   // bytes allocated by this object
    const std::string& device_name() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// CUDA runtime/driver/device facts for report.json.
struct CudaDeviceFacts {
    std::string name, driver_version, runtime_version, driver_api_version, compute_capability;
    uint64_t total_memory_bytes = 0, free_memory_bytes = 0;
};
CudaDeviceFacts query_cuda_device();   // throws std::runtime_error if no device

}  // namespace wf
