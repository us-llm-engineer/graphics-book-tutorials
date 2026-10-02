// Wavefront path tracer: queue-based stage execution for material divergence handling.
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "engine.cuh"
#include "wf/path_core.hpp"
#include "queues.cuh"

namespace wf::gpu {

struct PathPoolArrays {
    wf::Ray* rays;
    wf::Vec3* throughputs;
    wf::Vec3* L_accum;
    int* depths;
    int* specular_flags;
    float* prev_pdfs;
    wf::PathKey* keys;
    int* alive_flags;
    uint32_t* pixel_indices;
    wf::Ray* shadow_rays;
    wf::Vec3* shadow_contribs;
    uint32_t* shadow_paths;
    wf::HitRecord* hits;
    wf::Surface* surfaces;
};

// Max-update the high-water marks and fold the per-bounce overflow count into the pass counters.
// One thread: the kernel is a sequential bookkeeping step between stages, so no atomics are needed.
__global__ void kernel_update_counters(
    const uint32_t* queue_counters, const uint32_t* overflow, DeviceCounters* counters,
    uint32_t active_count, uint32_t capacity)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < 4; ++i) {
        uint32_t v = queue_counters[i] < capacity ? queue_counters[i] : capacity;
        if (v > counters->high_water[i]) counters->high_water[i] = v;
    }
    if (active_count > counters->high_water[4]) counters->high_water[4] = active_count;
    counters->overflows += *overflow;
}

__global__ void kernel_init_pixel_indices(uint32_t* pixel_indices, int pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pixels) pixel_indices[i] = i;
}

__global__ void kernel_init_paths(
    const SceneView scene, PathPoolArrays paths, uint32_t* active_list,
    uint32_t sample, uint32_t seed, int pixels)
{
    int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_idx >= pixels) return;
    
    int px = path_idx % scene.width;
    int py = path_idx / scene.width;
    
    wf::PathState ps = wf::init_path(scene, px, py, sample, seed);
    
    paths.rays[path_idx] = ps.ray;
    paths.throughputs[path_idx] = ps.throughput;
    paths.L_accum[path_idx] = ps.L;
    paths.depths[path_idx] = ps.depth;
    paths.specular_flags[path_idx] = ps.specular;
    paths.prev_pdfs[path_idx] = ps.prev_pdf;
    paths.keys[path_idx] = ps.key;
    paths.alive_flags[path_idx] = 1;
    paths.pixel_indices[path_idx] = path_idx;
    active_list[path_idx] = path_idx;
}

__global__ void kernel_extend(
    const SceneView scene, PathPoolArrays paths, const uint32_t* active_list, uint32_t active_count,
    wf::HitRecord* hits, DeviceCounters* counters, bool count_rays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;
    
    uint32_t path_idx = active_list[idx];
    wf::Ray ray = paths.rays[path_idx];
    wf::HitRecord h;
    wf::intersect(scene, ray, &h);
    paths.hits[path_idx] = h;
    
    if (count_rays) atomicAdd((unsigned long long*)&counters->extension_rays, 1ull);
}

__global__ void kernel_classify(
    const SceneView scene, PathPoolArrays paths, const uint32_t* active_list, uint32_t active_count,
    uint32_t** queue_items, uint32_t* queue_counters, uint32_t queue_capacity,
    uint32_t* overflow, int max_depth, DeviceCounters* counters)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = idx < active_count;
    uint32_t path_idx = 0;
    int mat = -1;

    if (valid) {
        path_idx = active_list[idx];
        wf::HitRecord h = paths.hits[path_idx];

        if (!hit_is_valid(scene, h)) {
            atomicAdd(&counters->invalid_hits, 1u);
            h.prim = -1;
        }

        wf::PathState ps;
        ps.ray = paths.rays[path_idx];
        ps.throughput = paths.throughputs[path_idx];
        ps.L = paths.L_accum[path_idx];
        ps.depth = paths.depths[path_idx];
        ps.specular = paths.specular_flags[path_idx];
        ps.prev_pdf = paths.prev_pdfs[path_idx];
        ps.key = paths.keys[path_idx];

        wf::Surface f;
        mat = wf::classify(scene, ps, h, max_depth, &f);
        paths.L_accum[path_idx] = ps.L;
        paths.surfaces[path_idx] = f;
        paths.alive_flags[path_idx] = (mat >= 0) ? 1 : 0;
    }

    // queue_push reserves slots for the whole calling warp in ONE queue, so each queue gets its own
    // call in which only the lanes of that material set pred; every lane of the warp calls all three.
    for (int q = 0; q < 3; ++q) {
        const bool mine = valid && mat == q;
        uint32_t slot = queue_push(queue_items[q], &queue_counters[q], queue_capacity, path_idx, mine, overflow);
        if (mine && slot == 0xFFFFFFFFu) paths.alive_flags[path_idx] = 0;   // dropped on overflow: the path ends
    }
}

__global__ void kernel_shade_lambert(
    const SceneView scene, PathPoolArrays paths, const uint32_t* lambert_queue, uint32_t lambert_count,
    uint32_t* shadow_items, uint32_t* shadow_counter, uint32_t queue_capacity,
    uint32_t* overflow, DeviceCounters* counters, bool count_rays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= lambert_count) return;
    
    uint32_t path_idx = lambert_queue[idx];
    wf::Surface f = paths.surfaces[path_idx];
    
    wf::PathState ps;
    ps.ray = paths.rays[path_idx];
    ps.throughput = paths.throughputs[path_idx];
    ps.L = paths.L_accum[path_idx];
    ps.depth = paths.depths[path_idx];
    ps.specular = paths.specular_flags[path_idx];
    ps.prev_pdf = paths.prev_pdfs[path_idx];
    ps.key = paths.keys[path_idx];
    
    wf::Ray shadow_ray;
    wf::Vec3 contrib;
    int has_shadow;
    bool alive = wf::shade_lambert(scene, ps, f, &shadow_ray, &contrib, &has_shadow);
    
    paths.rays[path_idx] = ps.ray;
    paths.throughputs[path_idx] = ps.throughput;
    paths.depths[path_idx] = ps.depth;
    paths.specular_flags[path_idx] = ps.specular;
    paths.prev_pdfs[path_idx] = ps.prev_pdf;
    paths.L_accum[path_idx] = ps.L;
    paths.alive_flags[path_idx] = alive ? 1 : 0;
    
    if (has_shadow) {
        uint32_t slot = queue_push(shadow_items, shadow_counter, queue_capacity, path_idx, true, overflow);
        if (slot != 0xFFFFFFFFu) {
            paths.shadow_rays[slot] = shadow_ray;
            paths.shadow_contribs[slot] = contrib;
            paths.shadow_paths[slot] = path_idx;
        }
    }
}

__global__ void kernel_shade_metal(const SceneView scene, PathPoolArrays paths,
    const uint32_t* metal_queue, uint32_t metal_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= metal_count) return;
    
    uint32_t path_idx = metal_queue[idx];
    wf::Surface f = paths.surfaces[path_idx];
    
    wf::PathState ps;
    ps.ray = paths.rays[path_idx];
    ps.throughput = paths.throughputs[path_idx];
    ps.L = paths.L_accum[path_idx];
    ps.depth = paths.depths[path_idx];
    ps.specular = paths.specular_flags[path_idx];
    ps.prev_pdf = paths.prev_pdfs[path_idx];
    ps.key = paths.keys[path_idx];
    
    bool alive = wf::shade_metal(scene, ps, f);
    
    paths.rays[path_idx] = ps.ray;
    paths.throughputs[path_idx] = ps.throughput;
    paths.depths[path_idx] = ps.depth;
    paths.specular_flags[path_idx] = ps.specular;
    paths.prev_pdfs[path_idx] = ps.prev_pdf;
    paths.alive_flags[path_idx] = alive ? 1 : 0;
}

__global__ void kernel_shade_dielectric(const SceneView scene, PathPoolArrays paths,
    const uint32_t* diel_queue, uint32_t diel_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= diel_count) return;
    
    uint32_t path_idx = diel_queue[idx];
    wf::Surface f = paths.surfaces[path_idx];
    
    wf::PathState ps;
    ps.ray = paths.rays[path_idx];
    ps.throughput = paths.throughputs[path_idx];
    ps.L = paths.L_accum[path_idx];
    ps.depth = paths.depths[path_idx];
    ps.specular = paths.specular_flags[path_idx];
    ps.prev_pdf = paths.prev_pdfs[path_idx];
    ps.key = paths.keys[path_idx];
    
    bool alive = wf::shade_dielectric(scene, ps, f);
    
    paths.rays[path_idx] = ps.ray;
    paths.throughputs[path_idx] = ps.throughput;
    paths.depths[path_idx] = ps.depth;
    paths.specular_flags[path_idx] = ps.specular;
    paths.prev_pdfs[path_idx] = ps.prev_pdf;
    paths.alive_flags[path_idx] = alive ? 1 : 0;
}

__global__ void kernel_shadow(const SceneView scene, PathPoolArrays paths,
    const uint32_t* shadow_queue, uint32_t shadow_count, DeviceCounters* counters, bool count_rays)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= shadow_count) return;
    
    uint32_t path_idx = paths.shadow_paths[idx];
    wf::Ray shadow_ray = paths.shadow_rays[idx];
    wf::Vec3 contrib = paths.shadow_contribs[idx];
    
    wf::HitRecord sh_hit;
    bool blocked = wf::intersect(scene, shadow_ray, &sh_hit, true);
    
    if (count_rays) atomicAdd((unsigned long long*)&counters->shadow_rays, 1ull);
    
    if (!blocked) {
        paths.L_accum[path_idx].x += contrib.x;
        paths.L_accum[path_idx].y += contrib.y;
        paths.L_accum[path_idx].z += contrib.z;
    }
}

__global__ void kernel_mark_alive(const uint32_t* active_list, uint32_t active_count,
    const int* alive_flags, uint32_t* alive_marks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= active_count) return;
    uint32_t path_idx = active_list[idx];
    alive_marks[idx] = alive_flags[path_idx] ? 1 : 0;   // flag by list position, as compact_u32 reads it
}

// Finish: accumulate L to film for ALL pixels, not just active ones
__global__ void kernel_finish(const DeviceFilm film, const PathPoolArrays paths, int pixels)
{
    int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_idx < pixels) {
        uint32_t pixel = paths.pixel_indices[path_idx];
        wf::gpu::film_add(film, pixel, paths.L_accum[path_idx]);
    }
}

}  // namespace wf::gpu

// Define WavefrontEngine::Impl as the actual implementation class
namespace wf::gpu {

// Test hook: queue capacity below the path count (0 = path count). The engine interface has no
// capacity parameter, so GpuRenderer sets it through this function before each pass.
static int g_test_queue_capacity = 0;
void wavefront_set_test_queue_capacity(int capacity) { g_test_queue_capacity = capacity; }

struct WavefrontEngine::Impl {
    DeviceScene scene;
    int width, height, pixels;
    PathPoolArrays paths = {};
    uint32_t* active_list;
    uint32_t* next_active_list;
    uint32_t active_count;
    uint32_t* queue_counters;
    uint32_t* queue_items[5];
    ScanWorkspace scan_ws;
    uint32_t* scratch_u32;
    uint32_t* overflow_counter;
    uint32_t* hw_device;
    uint32_t** queue_items_device;
    cudaEvent_t stage_events[10];
    
    Impl(const DeviceScene& scene, int width, int height);
    ~Impl();
    void run_pass(const DeviceFilm& film, DeviceCounters* counters, uint32_t sample,
                  uint32_t seed, int max_depth, bool count_rays, bool stage_timing,
                  StageTimes* stages, cudaStream_t stream);
    uint64_t bytes() const;
    
private:
    void cleanup();
    void init_pixel_indices();
};

__global__ void kernel_init_pixel_indices_impl(uint32_t* pixel_indices, int pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pixels) pixel_indices[i] = i;
}

WavefrontEngine::Impl::Impl(const DeviceScene& scene, int width, int height)
    : scene(scene), width(width), height(height), pixels(width * height),
      active_list(nullptr), next_active_list(nullptr), active_count(pixels),
      queue_counters(nullptr), scratch_u32(nullptr), overflow_counter(nullptr),
      hw_device(nullptr)
{
    for (int i = 0; i < 5; ++i) queue_items[i] = nullptr;
    for (int i = 0; i < 10; ++i) stage_events[i] = nullptr;
    
    try {
        cudaError_t e;
        
        e = cudaMalloc(&paths.rays, pixels * sizeof(wf::Ray));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc rays: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.throughputs, pixels * sizeof(wf::Vec3));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc throughputs: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.L_accum, pixels * sizeof(wf::Vec3));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc L_accum: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.depths, pixels * sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc depths: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.specular_flags, pixels * sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc specular_flags: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.prev_pdfs, pixels * sizeof(float));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc prev_pdfs: ") + cudaGetErrorString(e));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc prev_ps: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.keys, pixels * sizeof(wf::PathKey));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc keys: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.alive_flags, pixels * sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc alive_flags: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.pixel_indices, pixels * sizeof(uint32_t));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc pixel_indices: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.shadow_rays, pixels * sizeof(wf::Ray));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc shadow_rays: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.shadow_contribs, pixels * sizeof(wf::Vec3));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc shadow_contribs: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.shadow_paths, pixels * sizeof(uint32_t));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc shadow_paths: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.hits, pixels * sizeof(wf::HitRecord));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc hits: ") + cudaGetErrorString(e));
        e = cudaMalloc(&paths.surfaces, pixels * sizeof(wf::Surface));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc surfaces: ") + cudaGetErrorString(e));
        
        e = cudaMalloc(&active_list, pixels * sizeof(uint32_t));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc active_list: ") + cudaGetErrorString(e));
        e = cudaMalloc(&next_active_list, pixels * sizeof(uint32_t));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc next_active_list: ") + cudaGetErrorString(e));
        e = cudaMalloc(&queue_counters, 5 * sizeof(uint32_t));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc queue_counters: ") + cudaGetErrorString(e));
        
        for (int i = 0; i < 5; ++i) {
            e = cudaMalloc(&queue_items[i], pixels * sizeof(uint32_t));
            if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc queue_items: ") + cudaGetErrorString(e));
        }
        
        e = cudaMalloc(&queue_items_device, 5 * sizeof(uint32_t*));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc queue_items_device: ") + cudaGetErrorString(e));
        e = cudaMemcpy(queue_items_device, queue_items, 5 * sizeof(uint32_t*), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMemcpy queue_items_device: ") + cudaGetErrorString(e));
        
        e = cudaMalloc(&scratch_u32, (2 * (size_t)pixels + 1) * sizeof(uint32_t));  // flags | scratch | device count
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc scratch_u32: ") + cudaGetErrorString(e));
        e = cudaMalloc(&overflow_counter, sizeof(uint32_t));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc overflow_counter: ") + cudaGetErrorString(e));
        e = cudaMalloc(&hw_device, 5 * sizeof(uint32_t));
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc hw_device: ") + cudaGetErrorString(e));
        
        e = scan_workspace_create(&scan_ws, pixels);
        if (e != cudaSuccess) throw std::runtime_error(std::string("scan_workspace_create: ") + cudaGetErrorString(e));
        
        for (int i = 0; i < 10; ++i) {
            e = cudaEventCreate(&stage_events[i]);
            if (e != cudaSuccess) throw std::runtime_error(std::string("cudaEventCreate: ") + cudaGetErrorString(e));
        }
        
        init_pixel_indices();
    } catch (...) {
        cleanup();
        throw;
    }
}

void WavefrontEngine::Impl::cleanup() {
    cudaFree(paths.rays);
    cudaFree(paths.throughputs);
    cudaFree(paths.L_accum);
    cudaFree(paths.depths);
    cudaFree(paths.specular_flags);
    cudaFree(paths.prev_pdfs);
    cudaFree(paths.keys);
    cudaFree(paths.alive_flags);
    cudaFree(paths.pixel_indices);
    cudaFree(paths.shadow_rays);
    cudaFree(paths.shadow_contribs);
    cudaFree(paths.shadow_paths);
    cudaFree(paths.hits);
    cudaFree(paths.surfaces);
    cudaFree(active_list);
    cudaFree(next_active_list);
    cudaFree(queue_counters);
    for (int i = 0; i < 5; ++i) cudaFree(queue_items[i]);
    cudaFree(scratch_u32);
    cudaFree(overflow_counter);
    cudaFree(hw_device);
    cudaFree(queue_items_device);
    scan_workspace_destroy(&scan_ws);
    for (int i = 0; i < 10; ++i) if (stage_events[i]) cudaEventDestroy(stage_events[i]);
}

WavefrontEngine::Impl::~Impl() { cleanup(); }

void WavefrontEngine::Impl::init_pixel_indices() {
    const int threads = 256;
    int blocks = (pixels + threads - 1) / threads;
    kernel_init_pixel_indices_impl<<<blocks, threads>>>(paths.pixel_indices, pixels);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("init_pixel_indices: ") + cudaGetErrorString(e));
}

void WavefrontEngine::Impl::run_pass(const DeviceFilm& film, DeviceCounters* counters,
    uint32_t sample, uint32_t seed, int max_depth, bool count_rays, bool stage_timing,
    StageTimes* stages, cudaStream_t stream)
{
    const int threads = 256;
    const uint32_t cap = (g_test_queue_capacity > 0 && g_test_queue_capacity < pixels)
                             ? (uint32_t)g_test_queue_capacity : (uint32_t)pixels;

    // Stage timing: each mark closes the stage that just ran (events, synchronising; observation only).
    const bool timing = stage_timing && stages != nullptr;
    cudaEvent_t ev_prev = stage_events[0], ev_cur = stage_events[1];
    if (timing) cudaEventRecord(ev_prev, stream);
    auto mark = [&](double StageTimes::*field) {
        if (!timing) return;
        cudaEventRecord(ev_cur, stream);
        cudaEventSynchronize(ev_cur);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_prev, ev_cur);
        stages->*field += (double)ms;
        std::swap(ev_prev, ev_cur);
    };

    {
        int blocks = (pixels + threads - 1) / threads;
        kernel_init_paths<<<blocks, threads, 0, stream>>>(
            scene.view, paths, active_list, sample, seed, pixels);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("init_paths: ") + cudaGetErrorString(e));
    }
    
    active_count = pixels;
    int bounce = 0;
    const int max_bounces = max_depth + 2;
    
    while (active_count > 0 && bounce < max_bounces) {
        uint32_t zero_qc[5] = {0, 0, 0, 0, 0};
        uint32_t zero_hw[5] = {0, 0, 0, 0, 0};
        uint32_t zero_overflow = 0;
        
        cudaError_t e = cudaMemcpyAsync(queue_counters, zero_qc, 5 * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("memcpy zero queue_counters: ") + cudaGetErrorString(e));
        
        e = cudaMemcpyAsync(hw_device, zero_hw, 5 * sizeof(uint32_t),
                           cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("memcpy zero hw: ") + cudaGetErrorString(e));
        
        e = cudaMemcpyAsync(overflow_counter, &zero_overflow, sizeof(uint32_t),
                           cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("memcpy zero overflow: ") + cudaGetErrorString(e));
        
        mark(&StageTimes::other);
        // Extend
        {
            int blocks = (active_count + threads - 1) / threads;
            kernel_extend<<<blocks, threads, 0, stream>>>(
                scene.view, paths, active_list, active_count, paths.hits, counters, count_rays);
            e = cudaGetLastError();
            if (e != cudaSuccess) throw std::runtime_error(std::string("extend: ") + cudaGetErrorString(e));
        }
        
        // Classify
        mark(&StageTimes::extend);
        {
            int blocks = (active_count + threads - 1) / threads;
            kernel_classify<<<blocks, threads, 0, stream>>>(
                scene.view, paths, active_list, active_count,
                queue_items_device, queue_counters, cap, overflow_counter, max_depth, counters);
            e = cudaGetLastError();
            if (e != cudaSuccess) throw std::runtime_error(std::string("classify: ") + cudaGetErrorString(e));
        }
        
        uint32_t queue_counts[5];
        e = cudaMemcpyAsync(queue_counts, queue_counters, 5 * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("memcpy queue_counts: ") + cudaGetErrorString(e));
        cudaStreamSynchronize(stream);
        // queue_push counts dropped pushes too: never launch over more entries than were stored.
        for (int i = 0; i < 4; ++i) if (queue_counts[i] > cap) queue_counts[i] = cap;
        
        mark(&StageTimes::classify);
        // Shade Lambert
        if (queue_counts[0] > 0) {
            int blocks = (queue_counts[0] + threads - 1) / threads;
            kernel_shade_lambert<<<blocks, threads, 0, stream>>>(
                scene.view, paths, queue_items[0], queue_counts[0],
                queue_items[3], &queue_counters[3], cap, overflow_counter, counters, count_rays);
            e = cudaGetLastError();
            if (e != cudaSuccess) throw std::runtime_error(std::string("shade_lambert: ") + cudaGetErrorString(e));
        }
        
        mark(&StageTimes::shade_lambert);
        // Shade metal
        if (queue_counts[1] > 0) {
            int blocks = (queue_counts[1] + threads - 1) / threads;
            kernel_shade_metal<<<blocks, threads, 0, stream>>>(
                scene.view, paths, queue_items[1], queue_counts[1]);
            e = cudaGetLastError();
            if (e != cudaSuccess) throw std::runtime_error(std::string("shade_metal: ") + cudaGetErrorString(e));
        }
        
        mark(&StageTimes::shade_metal);
        // Shade dielectric
        if (queue_counts[2] > 0) {
            int blocks = (queue_counts[2] + threads - 1) / threads;
            kernel_shade_dielectric<<<blocks, threads, 0, stream>>>(
                scene.view, paths, queue_items[2], queue_counts[2]);
            e = cudaGetLastError();
            if (e != cudaSuccess) throw std::runtime_error(std::string("shade_dielectric: ") + cudaGetErrorString(e));
        }
        
        mark(&StageTimes::shade_dielectric);
        kernel_update_counters<<<1, 1, 0, stream>>>(queue_counters, overflow_counter, counters, active_count, cap);
        e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("update_counters: ") + cudaGetErrorString(e));

        // Shadow
        e = cudaMemcpyAsync(queue_counts, queue_counters, 5 * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("memcpy shadow count: ") + cudaGetErrorString(e));
        cudaStreamSynchronize(stream);
        // queue_push counts dropped pushes too: never launch over more entries than were stored.
        for (int i = 0; i < 4; ++i) if (queue_counts[i] > cap) queue_counts[i] = cap;
        
        if (queue_counts[3] > 0) {
            int blocks = (queue_counts[3] + threads - 1) / threads;
            kernel_shadow<<<blocks, threads, 0, stream>>>(
                scene.view, paths, queue_items[3], queue_counts[3], counters, count_rays);
            e = cudaGetLastError();
            if (e != cudaSuccess) throw std::runtime_error(std::string("shadow: ") + cudaGetErrorString(e));
        }
        
        mark(&StageTimes::shadow);
        // Mark alive and compact
        {
            int blocks = (active_count + threads - 1) / threads;
            kernel_mark_alive<<<blocks, threads, 0, stream>>>(
                active_list, active_count, paths.alive_flags, scratch_u32);
            e = cudaGetLastError();
            if (e != cudaSuccess) throw std::runtime_error(std::string("mark_alive: ") + cudaGetErrorString(e));
        }
        
        uint32_t* d_count = scratch_u32 + 2 * (size_t)pixels;       // device scalar for compact_u32
        e = compact_u32(active_list, scratch_u32, next_active_list, d_count,
                        scratch_u32 + pixels, (int)active_count, scan_ws, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("compact_u32: ") + cudaGetErrorString(e));
        uint32_t new_active_count = 0;
        e = cudaMemcpyAsync(&new_active_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) throw std::runtime_error(std::string("memcpy active count: ") + cudaGetErrorString(e));
        cudaStreamSynchronize(stream);

        active_count = new_active_count;
        std::swap(active_list, next_active_list);
        mark(&StageTimes::compact);
        bounce++;
    }
    
    // Finish: accumulate ALL pixels to film
    {
        int blocks = (pixels + threads - 1) / threads;
        kernel_finish<<<blocks, threads, 0, stream>>>(film, paths, pixels);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) throw std::runtime_error(std::string("finish: ") + cudaGetErrorString(e));
        mark(&StageTimes::other);
    }
    
    // Sync stream
    cudaStreamSynchronize(stream);
}

uint64_t WavefrontEngine::Impl::bytes() const {
    uint64_t total = 0;
    total += (uint64_t)pixels * (sizeof(wf::Ray) + sizeof(wf::Vec3) + sizeof(wf::Vec3) + sizeof(int) + sizeof(int) + sizeof(float) + sizeof(wf::Vec3) + sizeof(wf::PathKey) + sizeof(int) + sizeof(uint32_t));
    total += (uint64_t)pixels * (sizeof(wf::Ray) + sizeof(wf::Vec3) + sizeof(uint32_t) + sizeof(wf::HitRecord) + sizeof(wf::Surface));
    total += (uint64_t)pixels * 2 * sizeof(uint32_t);
    total += 5 * sizeof(uint32_t);
    total += (uint64_t)pixels * 5 * sizeof(uint32_t);
    total += (uint64_t)pixels * sizeof(uint32_t);
    total += 6 * sizeof(uint32_t);
    return total;
}

}  // namespace wf::gpu

namespace wf::gpu {

WavefrontEngine::WavefrontEngine(const DeviceScene& scene, int width, int height)
    : impl_(new Impl(scene, width, height)) {}

WavefrontEngine::~WavefrontEngine() {
    delete static_cast<Impl*>(impl_);
    impl_ = nullptr;
}

void WavefrontEngine::run_pass(const DeviceFilm& film, DeviceCounters* counters,
                               uint32_t sample, uint32_t seed, int max_depth, bool count_rays,
                               bool stage_timing, StageTimes* stages, cudaStream_t stream) {
    if (!impl_) return;
    static_cast<Impl*>(impl_)->run_pass(
        film, counters, sample, seed, max_depth, count_rays, stage_timing, stages, stream);
}

uint64_t WavefrontEngine::bytes() const {
    if (!impl_) return 0;
    return static_cast<Impl*>(impl_)->bytes();
}

}  // namespace wf::gpu
