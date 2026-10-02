// CPU renderer: multi-threaded path tracer using the shared estimator.
#include "wf/renderer.hpp"
#include "wf/path_core.hpp"
#include "wf/math.hpp"
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace wf {

RenderResult render_cpu(const Scene& scene, const RenderParams& p, RenderStats* stats) {
    // Record total start time (including setup)
    auto total_start = std::chrono::high_resolution_clock::now();

    SceneView view = scene.view();

    int width = scene.width;
    int height = scene.height;
    int spp = p.spp;
    uint32_t seed = p.seed;
    int max_depth = p.max_depth;

    // Determine number of threads
    int threads = p.threads;
    if (threads <= 0) {
        threads = (int)std::thread::hardware_concurrency();
        if (threads <= 0) threads = 1;
    }

    // Allocate result buffers
    int n_pixels = width * height;
    std::vector<float> rgb_sum(3 * n_pixels, 0.0f);
    std::vector<float> lum_sum(n_pixels, 0.0f);
    std::vector<float> lum_sum2(n_pixels, 0.0f);

    // Initialize stats if provided
    if (stats) {
        stats->render_ms = 0.0;
        stats->total_ms = 0.0;
        stats->extension_rays = 0;
        stats->shadow_rays = 0;
        stats->queue_overflows = 0;
        stats->invalid_hits = 0;
        stats->peak_vram_bytes = 0;
        for (int i = 0; i < 5; ++i) stats->queue_high_water[i] = 0;
    }

    // Record render start time (before path tracing)
    auto render_start = std::chrono::high_resolution_clock::now();

    // Render samples in a multi-threaded loop
    // Each thread processes a contiguous range of rows
    auto render_rows = [&](int start_row, int end_row) {
        for (int py = start_row; py < end_row; ++py) {
            for (int px = 0; px < width; ++px) {
                int pixel_idx = py * width + px;
                for (int s = 0; s < spp; ++s) {
                    Vec3 L = trace_path(view, px, py, (uint32_t)s, seed, max_depth);

                    // Accumulate RGB into sum
                    rgb_sum[3 * pixel_idx] += L.x;
                    rgb_sum[3 * pixel_idx + 1] += L.y;
                    rgb_sum[3 * pixel_idx + 2] += L.z;

                    // Accumulate luminance statistics
                    float lum = luminance(L);
                    lum_sum[pixel_idx] += lum;
                    lum_sum2[pixel_idx] += lum * lum;
                }
            }
        }
    };

    // Launch threads
    std::vector<std::thread> thread_pool;
    int rows_per_thread = (height + threads - 1) / threads;

    for (int t = 0; t < threads; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = std::min((t + 1) * rows_per_thread, height);
        if (start_row < height) {
            thread_pool.emplace_back([&, start_row, end_row]() {
                render_rows(start_row, end_row);
            });
        }
    }

    // Wait for all threads to finish
    for (auto& t : thread_pool) {
        t.join();
    }

    // Record render end time
    auto render_end = std::chrono::high_resolution_clock::now();

    // Compute final result
    RenderResult result;
    result.width = width;
    result.height = height;
    result.spp = spp;
    result.rgb.resize(3 * n_pixels);
    result.lum_var.resize(n_pixels);

    for (int i = 0; i < n_pixels; ++i) {
        // Compute mean RGB
        result.rgb[3 * i] = rgb_sum[3 * i] / spp;
        result.rgb[3 * i + 1] = rgb_sum[3 * i + 1] / spp;
        result.rgb[3 * i + 2] = rgb_sum[3 * i + 2] / spp;

        // Compute variance of per-sample luminance
        float mean_lum = lum_sum[i] / spp;
        if (spp > 1) {
            float sum_sq_diff = lum_sum2[i] - spp * mean_lum * mean_lum;
            result.lum_var[i] = std::max(0.0f, sum_sq_diff / (spp - 1));
        } else {
            result.lum_var[i] = 0.0f;
        }
    }

    // Record total end time
    auto total_end = std::chrono::high_resolution_clock::now();

    // Fill in timing statistics if provided
    if (stats) {
        // render_ms: wall clock time of actual rendering (excluding setup)
        auto render_duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(render_end - render_start);
        stats->render_ms = render_duration.count();

        // total_ms: total execution time (including setup)
        auto total_duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(total_end - total_start);
        stats->total_ms = total_duration.count();
    }

    return result;
}

}  // namespace wf
