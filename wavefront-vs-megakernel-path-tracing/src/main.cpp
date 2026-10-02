// Command-line interface: `render` command for CPU and GPU path tracing.
#include "wf/config.hpp"
#include "wf/scene.hpp"
#include "wf/renderer.hpp"
#include "wf/film.hpp"
#include "wf/image_io.hpp"
#include "wf/report.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>

#ifdef __GNUC__
#include <unistd.h>
#endif

using namespace wf;

std::string get_git_revision() {
    FILE* p = popen("git rev-parse HEAD 2>/dev/null", "r");
    if (!p) return "";
    char buf[64] = {0};
    size_t n = fread(buf, 1, sizeof(buf) - 1, p);
    pclose(p);
    std::string rev(buf, n);
    // Trim whitespace
    while (!rev.empty() && (rev.back() == '\n' || rev.back() == '\r')) {
        rev.pop_back();
    }
    return rev;
}

int main(int argc, const char* argv[]) {
    try {
        // Parse command-line arguments
        Config config = parse_args(argc, argv);

        // Load or create scene
        Scene scene;
        std::string scene_identifier = config.scene;
        if (scene_identifier.size() >= 4 &&
            scene_identifier.substr(scene_identifier.size() - 4) == ".obj") {
            // Load from OBJ file
            scene = load_obj(scene_identifier, config.width, config.height);
        } else {
            // Create procedural scene
            scene = make_scene(scene_identifier, config.width, config.height);
        }

        // Set up render parameters
        RenderParams render_params;
        render_params.spp = config.spp;
        render_params.seed = config.seed;
        render_params.max_depth = config.max_depth;
        render_params.threads = config.threads;
        render_params.count_rays = false;  // Only enable if needed
        render_params.stage_timing = false;

        RenderResult result;
        RenderStats stats;
        std::vector<double> timing_runs;

        // Handle warm-up renders
        for (int w = 0; w < config.warmup; ++w) {
            if (config.mode == Mode::Cpu) {
                render_cpu(scene, render_params, &stats);
            } else {
                // GPU modes: wavefront will throw on round 1
                try {
                    GpuRenderer renderer(scene, config.mode == Mode::Megakernel ? GpuMode::Megakernel : GpuMode::Wavefront);
                    renderer.render(render_params, &stats);
                } catch (const std::runtime_error& e) {
                    std::cerr << e.what() << std::endl;
                    return 1;
                }
            }
        }

        // Render with timing
        for (int r = 0; r < config.reps; ++r) {
            stats = RenderStats{};   // counters of the last timed run only
            if (config.mode == Mode::Cpu) {
                result = render_cpu(scene, render_params, &stats);
                timing_runs.push_back(stats.render_ms);
            } else {
                // GPU modes
                try {
                    GpuRenderer renderer(scene, config.mode == Mode::Megakernel ? GpuMode::Megakernel : GpuMode::Wavefront);
                    result = renderer.render(render_params, &stats);
                    timing_runs.push_back(stats.render_ms);
                } catch (const std::runtime_error& e) {
                    std::cerr << e.what() << std::endl;
                    return 1;
                }
            }
        }

        // Write output images if prefix is provided
        if (!config.out_prefix.empty()) {
            // Tone map and write PNG
            std::vector<uint8_t> rgb8 = tonemap_to_srgb8(result.rgb);
            std::string png_path = config.out_prefix + ".png";
            write_png(png_path, result.width, result.height, rgb8);

            // Write PFM
            std::string pfm_path = config.out_prefix + ".pfm";
            write_pfm(pfm_path, result.width, result.height, result.rgb);
        }

        // Compute PFM hash
        uint64_t pfm_hash = fnv1a64(result.rgb.data(), result.rgb.size() * sizeof(float));

        // Write report if requested
        if (!config.report_path.empty()) {
            Report report;
            report.git_revision = get_git_revision();
            report.scene_name = config.scene;
            report.scene_hash = scene_hash(scene);
            report.width = config.width;
            report.height = config.height;
            report.spp = config.spp;
            report.max_depth = config.max_depth;
            report.seed = config.seed;
            report.mode = mode_name(config.mode);
            report.device.name = config.mode == Mode::Cpu ? "cpu" : "gpu";
            report.image_png = config.out_prefix.empty() ? "" : config.out_prefix + ".png";
            report.image_pfm = config.out_prefix.empty() ? "" : config.out_prefix + ".pfm";
            report.pfm_hash = pfm_hash;
            report.nonfinite_pixels = count_nonfinite(result.rgb) / 3;
            report.run_ms = timing_runs;

            // Compute samples per second from median run time
            if (!timing_runs.empty()) {
                double median_ms = percentile(timing_runs, 0.5);
                if (median_ms > 0.0) {
                    double samples = (double)config.width * config.height * config.spp;
                    report.samples_per_second = samples / (median_ms / 1000.0);
                }
            }

            // Validity counters and queue high-water marks of the last timed run.
            for (int i = 0; i < 5; ++i) report.queue_high_water[i] = stats.queue_high_water[i];
            report.queue_overflows = stats.queue_overflows;
            report.invalid_hits = stats.invalid_hits;

            // Rays per second from one separate, untimed counting run (counting never touches the timings).
            if (config.mode != Mode::Cpu && !timing_runs.empty()) {
                const double median_ms = percentile(timing_runs, 0.5);
                try {
                    RenderParams counting = render_params;
                    counting.count_rays = true;
                    RenderStats cstats;
                    GpuRenderer counter(scene, config.mode == Mode::Megakernel ? GpuMode::Megakernel : GpuMode::Wavefront);
                    counter.render(counting, &cstats);
                    const double rays = (double)cstats.extension_rays + (double)cstats.shadow_rays;
                    if (median_ms > 0.0 && rays > 0.0) report.rays_per_second = rays / (median_ms / 1000.0);
                } catch (const std::runtime_error& e) {
                    std::cerr << "warning: ray counting run failed: " << e.what() << std::endl;
                }
            }

            // GPU-specific fields
            if (config.mode != Mode::Cpu) {
                try {
                    CudaDeviceFacts facts = query_cuda_device();
                    report.device.name = facts.name;
                    report.device.driver_version = facts.driver_version;
                    report.device.cuda_runtime_version = facts.runtime_version;
                    report.device.cuda_driver_version = facts.driver_api_version;
                    report.device.compute_capability = facts.compute_capability;
                    report.device.total_memory_bytes = facts.total_memory_bytes;
                    report.peak_vram_bytes = stats.peak_vram_bytes;
                } catch (const std::runtime_error&) {
                    // If CUDA device query fails, leave default values
                }
            }

            // Write report.json
            std::string json_text = report_to_json(report);
            std::ofstream report_file(config.report_path);
            if (!report_file) {
                std::cerr << "Error: cannot open report file for writing: " << config.report_path << std::endl;
                return 1;
            }
            report_file << json_text;
            if (!report_file) {
                std::cerr << "Error: failed to write report file" << std::endl;
                return 1;
            }
        }

        return 0;

    } catch (const ConfigError& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
