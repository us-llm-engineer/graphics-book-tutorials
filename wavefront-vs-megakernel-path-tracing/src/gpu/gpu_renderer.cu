// GPU renderer: manages device memory, uploads scenes, and runs megakernel/wavefront.
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "engine.cuh"
#include "wf/renderer.hpp"

namespace wf {

// Query CUDA device facts.
CudaDeviceFacts query_cuda_device() {
    int device = 0;
    cudaError_t e = cudaGetDevice(&device);
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(e));
    }

    cudaDeviceProp props;
    e = cudaGetDeviceProperties(&props, device);
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(e));
    }

    size_t free_bytes, total_bytes;
    e = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(e));
    }

    // Get driver and runtime versions.
    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    // Format compute capability.
    std::string cc = std::to_string(props.major) + "." + std::to_string(props.minor);

    // Format versions.
    std::string driver_str = std::to_string(driver_version / 1000) + "." +
                             std::to_string((driver_version % 1000) / 10);
    std::string runtime_str = std::to_string(runtime_version / 1000) + "." +
                              std::to_string((runtime_version % 1000) / 10);

    return CudaDeviceFacts{
        props.name,
        driver_str,
        runtime_str,
        std::to_string(driver_version),  // driver_api_version
        cc,
        total_bytes,
        free_bytes
    };
}

// ============================================================================
// GpuRenderer implementation
// ============================================================================

namespace gpu { void wavefront_set_test_queue_capacity(int capacity); }

struct GpuRenderer::Impl {
    gpu::DeviceScene scene{};  // Value-initialize to avoid garbage pointers.
    gpu::DeviceFilm film{};    // Value-initialize to avoid garbage pointers.
    gpu::DeviceCounters* d_counters = nullptr;
    GpuMode mode = GpuMode::Megakernel;
    gpu::WavefrontEngine* engine = nullptr;

    int width = 0, height = 0;
    int pixels = 0;

    size_t peak_vram_bytes = 0;

    // For timing.
    cudaEvent_t event_start = nullptr, event_end = nullptr;

    Impl() {}

    ~Impl() {
        if (engine) delete engine;
        if (d_counters) cudaFree(d_counters);
        if (film.rgb) cudaFree(film.rgb);
        if (film.lum) cudaFree(film.lum);
        if (film.lum2) cudaFree(film.lum2);

        if (scene.view.spheres) cudaFree((void*)scene.view.spheres);
        if (scene.view.vertices) cudaFree((void*)scene.view.vertices);
        if (scene.view.triangles) cudaFree((void*)scene.view.triangles);
        if (scene.view.materials) cudaFree((void*)scene.view.materials);
        if (scene.view.nodes) cudaFree((void*)scene.view.nodes);
        if (scene.view.prim_order) cudaFree((void*)scene.view.prim_order);
        if (scene.view.lights) cudaFree((void*)scene.view.lights);

        if (event_start) cudaEventDestroy(event_start);
        if (event_end) cudaEventDestroy(event_end);
    }
};

GpuRenderer::GpuRenderer(const Scene& scene, GpuMode mode) : impl_(std::make_unique<Impl>()) {
    // Check scene is finalized.
    if (scene.nodes.empty()) {
        throw std::runtime_error("Scene not finalized (no BVH nodes)");
    }

    // Store mode and create appropriate engine for wavefront.
    impl_->mode = mode;

    // Get device info to track peak VRAM.
    size_t free_before = 0, total = 0;
    cudaError_t e = cudaMemGetInfo(&free_before, &total);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemGetInfo: ") + cudaGetErrorString(e));

    // Upload scene data.
    uint64_t scene_bytes = 0;

    // Spheres
    if (!scene.spheres.empty()) {
        e = cudaMalloc((void**)&impl_->scene.view.spheres, scene.spheres.size() * sizeof(Sphere));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc spheres: ") + cudaGetErrorString(e));
        e = cudaMemcpy((void*)impl_->scene.view.spheres, scene.spheres.data(),
                      scene.spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy spheres: ") + cudaGetErrorString(e));
        scene_bytes += scene.spheres.size() * sizeof(Sphere);
    }

    // Vertices
    if (!scene.vertices.empty()) {
        e = cudaMalloc((void**)&impl_->scene.view.vertices, scene.vertices.size() * sizeof(Vec3));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc vertices: ") + cudaGetErrorString(e));
        e = cudaMemcpy((void*)impl_->scene.view.vertices, scene.vertices.data(),
                      scene.vertices.size() * sizeof(Vec3), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy vertices: ") + cudaGetErrorString(e));
        scene_bytes += scene.vertices.size() * sizeof(Vec3);
    }

    // Triangles
    if (!scene.triangles.empty()) {
        e = cudaMalloc((void**)&impl_->scene.view.triangles, scene.triangles.size() * sizeof(Triangle));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc triangles: ") + cudaGetErrorString(e));
        e = cudaMemcpy((void*)impl_->scene.view.triangles, scene.triangles.data(),
                      scene.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy triangles: ") + cudaGetErrorString(e));
        scene_bytes += scene.triangles.size() * sizeof(Triangle);
    }

    // Materials
    if (!scene.materials.empty()) {
        e = cudaMalloc((void**)&impl_->scene.view.materials, scene.materials.size() * sizeof(Material));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc materials: ") + cudaGetErrorString(e));
        e = cudaMemcpy((void*)impl_->scene.view.materials, scene.materials.data(),
                      scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy materials: ") + cudaGetErrorString(e));
        scene_bytes += scene.materials.size() * sizeof(Material);
    }

    // BVH nodes
    if (!scene.nodes.empty()) {
        e = cudaMalloc((void**)&impl_->scene.view.nodes, scene.nodes.size() * sizeof(BvhNode));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc nodes: ") + cudaGetErrorString(e));
        e = cudaMemcpy((void*)impl_->scene.view.nodes, scene.nodes.data(),
                      scene.nodes.size() * sizeof(BvhNode), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy nodes: ") + cudaGetErrorString(e));
        scene_bytes += scene.nodes.size() * sizeof(BvhNode);
    }

    // Primitive order
    if (!scene.prim_order.empty()) {
        e = cudaMalloc((void**)&impl_->scene.view.prim_order, scene.prim_order.size() * sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc prim_order: ") + cudaGetErrorString(e));
        e = cudaMemcpy((void*)impl_->scene.view.prim_order, scene.prim_order.data(),
                      scene.prim_order.size() * sizeof(int), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy prim_order: ") + cudaGetErrorString(e));
        scene_bytes += scene.prim_order.size() * sizeof(int);
    }

    // Lights
    if (!scene.lights.empty()) {
        e = cudaMalloc((void**)&impl_->scene.view.lights, scene.lights.size() * sizeof(Light));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc lights: ") + cudaGetErrorString(e));
        e = cudaMemcpy((void*)impl_->scene.view.lights, scene.lights.data(),
                      scene.lights.size() * sizeof(Light), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy lights: ") + cudaGetErrorString(e));
        scene_bytes += scene.lights.size() * sizeof(Light);
    }

    // Fill the view.
    impl_->scene.view.n_spheres = (int)scene.spheres.size();
    impl_->scene.view.n_triangles = (int)scene.triangles.size();
    impl_->scene.view.n_materials = (int)scene.materials.size();
    impl_->scene.view.n_nodes = (int)scene.nodes.size();
    impl_->scene.view.n_lights = (int)scene.lights.size();
    impl_->scene.view.env = scene.env;
    impl_->scene.view.camera = make_camera_rig(scene.camera, scene.width, scene.height);
    impl_->scene.view.width = scene.width;
    impl_->scene.view.height = scene.height;

    impl_->scene.bytes = scene_bytes;
    impl_->width = scene.width;
    impl_->height = scene.height;
    impl_->pixels = scene.width * scene.height;

    // Allocate film buffers.
    size_t film_pixels = (size_t)impl_->pixels;
    e = cudaMalloc(&impl_->film.rgb, 3 * film_pixels * sizeof(float));
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc film.rgb: ") + cudaGetErrorString(e));
    e = cudaMalloc(&impl_->film.lum, film_pixels * sizeof(float));
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc film.lum: ") + cudaGetErrorString(e));
    e = cudaMalloc(&impl_->film.lum2, film_pixels * sizeof(float));
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc film.lum2: ") + cudaGetErrorString(e));

    impl_->film.pixels = impl_->pixels;

    size_t film_bytes = 3 * film_pixels * sizeof(float) + 2 * film_pixels * sizeof(float);

    // Allocate counters.
    e = cudaMalloc(&impl_->d_counters, sizeof(gpu::DeviceCounters));
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc counters: ") + cudaGetErrorString(e));

    // Create CUDA events for timing.
    e = cudaEventCreate(&impl_->event_start);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaEventCreate start: ") + cudaGetErrorString(e));
    e = cudaEventCreate(&impl_->event_end);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaEventCreate end: ") + cudaGetErrorString(e));

    // Create engine if wavefront.
    uint64_t engine_bytes = 0;
    if (mode == GpuMode::Wavefront) {
        impl_->engine = new gpu::WavefrontEngine(impl_->scene, scene.width, scene.height);
        engine_bytes = impl_->engine->bytes();
    }

    // Measure peak VRAM.
    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total);
    size_t allocated = free_before - free_after;
    impl_->peak_vram_bytes = allocated + scene_bytes + film_bytes + sizeof(gpu::DeviceCounters) + engine_bytes;
}

GpuRenderer::~GpuRenderer() {}

void GpuRenderer::reset() {
    if (!impl_) return;

    // Zero film buffers.
    cudaError_t e = cudaMemset(impl_->film.rgb, 0, 3 * impl_->pixels * sizeof(float));
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemset rgb: ") + cudaGetErrorString(e));
    e = cudaMemset(impl_->film.lum, 0, impl_->pixels * sizeof(float));
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemset lum: ") + cudaGetErrorString(e));
    e = cudaMemset(impl_->film.lum2, 0, impl_->pixels * sizeof(float));
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemset lum2: ") + cudaGetErrorString(e));
}

void GpuRenderer::run_passes(int first_sample, int count, const RenderParams& p, RenderStats* stats) {
    if (!impl_) return;

    // Record start time.
    cudaEventRecord(impl_->event_start, 0);

    for (int i = 0; i < count; ++i) {
        int sample = first_sample + i;

        // Zero counters.
        gpu::DeviceCounters zero_counters{0};
        cudaError_t e = cudaMemcpy(impl_->d_counters, &zero_counters, sizeof(gpu::DeviceCounters), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy zero counters: ") + cudaGetErrorString(e));

        // Run appropriate engine.
        if (impl_->mode == GpuMode::Wavefront) {
            gpu::wavefront_set_test_queue_capacity(p.queue_capacity);
            impl_->engine->run_pass(impl_->film, impl_->d_counters, sample, p.seed, p.max_depth, p.count_rays, p.stage_timing, stats ? &stats->stages : nullptr, 0);
        } else {
            gpu::run_megakernel_pass(impl_->scene, impl_->film, impl_->d_counters,
                                     sample, p.seed, p.max_depth, p.count_rays, 0);
        }

        // Optionally copy back stats for this pass.
        if (stats) {
            gpu::DeviceCounters counters;
            cudaError_t e = cudaMemcpy(&counters, impl_->d_counters, sizeof(gpu::DeviceCounters), cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy get counters: ") + cudaGetErrorString(e));
            stats->extension_rays += counters.extension_rays;
            stats->shadow_rays += counters.shadow_rays;
            stats->invalid_hits += counters.invalid_hits;
            stats->queue_overflows += counters.overflows;

            // Update high-water marks.
            for (int j = 0; j < 5; ++j) {
                stats->queue_high_water[j] = std::max(stats->queue_high_water[j], (long long)counters.high_water[j]);
            }
        }
    }

    // Record end time.
    cudaEventRecord(impl_->event_end, 0);
    cudaEventSynchronize(impl_->event_end);

    if (stats) {
        float ms = 0;
        cudaEventElapsedTime(&ms, impl_->event_start, impl_->event_end);
        stats->render_ms = (double)ms;
        stats->peak_vram_bytes = impl_->peak_vram_bytes;
    }
}

RenderResult GpuRenderer::download(int spp_done) {
    if (!impl_) return RenderResult();

    RenderResult result;
    result.width = impl_->width;
    result.height = impl_->height;
    result.spp = spp_done;

    // Download RGB.
    result.rgb.resize(3 * impl_->pixels);
    cudaError_t e = cudaMemcpy(result.rgb.data(), impl_->film.rgb, 3 * impl_->pixels * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy download rgb: ") + cudaGetErrorString(e));

    // Divide RGB by spp.
    float inv_spp = 1.0f / (float)spp_done;
    for (auto& v : result.rgb) v *= inv_spp;

    // Download luminance variance.
    result.lum_var.resize(impl_->pixels);
    std::vector<float> lum(impl_->pixels), lum2(impl_->pixels);
    e = cudaMemcpy(lum.data(), impl_->film.lum, impl_->pixels * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy download lum: ") + cudaGetErrorString(e));
    e = cudaMemcpy(lum2.data(), impl_->film.lum2, impl_->pixels * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy download lum2: ") + cudaGetErrorString(e));

    // Compute variance: E[X^2] - E[X]^2. Use double for better precision.
    // Clamp negative variances to 0 to handle floating-point precision issues.
    for (int i = 0; i < impl_->pixels; ++i) {
        double mean_lum = (double)lum[i] / (double)spp_done;
        double mean_lum2 = (double)lum2[i] / (double)spp_done;
        double var = mean_lum2 - mean_lum * mean_lum;
        result.lum_var[i] = (float)std::max(0.0, var);
    }

    return result;
}

void GpuRenderer::download_mean(int spp_done, std::vector<float>* rgb) {
    if (!impl_ || !rgb) return;

    rgb->resize(3 * impl_->pixels);
    cudaError_t e = cudaMemcpy(rgb->data(), impl_->film.rgb, 3 * impl_->pixels * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy download_mean: ") + cudaGetErrorString(e));

    // Divide by spp.
    float inv_spp = 1.0f / (float)spp_done;
    for (auto& v : *rgb) v *= inv_spp;
}

RenderResult GpuRenderer::render(const RenderParams& p, RenderStats* stats) {
    reset();

    RenderStats local_stats{};
    run_passes(0, p.spp, p, &local_stats);

    if (stats) *stats = local_stats;

    return download(p.spp);
}

uint64_t GpuRenderer::device_bytes() const {
    if (!impl_) return 0;

    uint64_t total = impl_->scene.bytes;
    total += 3ull * impl_->pixels * sizeof(float);  // film.rgb
    total += impl_->pixels * sizeof(float);         // film.lum
    total += impl_->pixels * sizeof(float);         // film.lum2
    total += sizeof(gpu::DeviceCounters);
    if (impl_->engine) total += impl_->engine->bytes();

    return total;
}

const std::string& GpuRenderer::device_name() const {
    static std::string name;
    if (name.empty()) {
        try {
            auto facts = query_cuda_device();
            name = facts.name;
        } catch (...) {
            name = "unknown";
        }
    }
    return name;
}

}  // namespace wf
