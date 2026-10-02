// Benchmark protocol (DESIGN.md section 9) and verdict rule, fixed before any measurement.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "wf/renderer.hpp"
#include "wf/scene.hpp"

namespace wf {

constexpr int kBenchSchemaVersion = 1;
constexpr double kSpeedupTarget = 1.30;            // wavefront must be >= 30% faster
constexpr double kVramBudgetBytes = 3.5e9;         // operational device memory ceiling
constexpr double kSePixelFraction = 0.995;         // CPU vs GPU: pixels within 3 combined SE
constexpr double kMeanRelTolerance = 0.005;        // CPU vs GPU: image mean relative difference
constexpr double kBitwiseRelTolerance = 1e-5;      // megakernel vs wavefront max relative difference

struct BenchConfig {
    int reps = 5;                 // timed repetitions per condition
    int warmup = 2;               // untimed warm-up renders per condition
    int cpu_spp = 0;              // 0 = skip the CPU quality reference; else render_cpu at this spp
    int cpu_threads = 0;
    bool count_rays = true;       // one extra untimed run per GPU condition
    bool stage_timing = true;     // one extra untimed instrumented wavefront run
    RenderParams params;          // spp, seed, max_depth shared by all conditions
};

struct ConditionStats {
    std::string mode;             // "megakernel" | "wavefront"
    std::vector<double> run_ms;   // timed repetitions in run order
    double min = 0, median = 0, p10 = 0, p90 = 0, max = 0;
    double spread_rel = 0;        // (p90 - p10) / median
    double samples_per_second = 0;   // width*height*spp / median seconds
    double rays_per_second = 0;      // 0 if not counted
    uint64_t peak_vram_bytes = 0;
    long long queue_high_water[5] = {0, 0, 0, 0, 0};
    long long queue_overflows = 0, invalid_hits = 0, nonfinite_pixels = 0;
    StageTimes stages;               // instrumented run only (wavefront), else zeros
};

struct QualityCheck {
    std::string name;     // "megakernel_vs_wavefront_bitwise", "cpu_vs_megakernel", "cpu_vs_wavefront"
    bool pass = false;
    double value = 0.0;       // the measured quantity (max_abs_rel, frac_within_3se, ...)
    double threshold = 0.0;   // the pre-registered threshold it was compared with
    std::string detail;
};

struct BenchResult {
    std::string scene_name;
    uint64_t scene_hash = 0;
    int width = 0, height = 0, spp = 0, max_depth = 0, reps = 0, warmup = 0;
    unsigned seed = 0;
    std::string device_name, driver_version, cuda_runtime_version;
    std::vector<ConditionStats> conditions;   // megakernel first, wavefront second
    std::vector<QualityCheck> quality;
    double speedup = 0.0;          // median(megakernel) / median(wavefront); 0 if a mode is missing
    bool speedup_meets_target = false;   // speedup >= kSpeedupTarget
    bool vram_within_budget = false;     // every condition peak_vram_bytes <= kVramBudgetBytes
    std::string verdict;           // "hypothesis_supported" | "hypothesis_weakened"
    std::vector<std::string> reasons;    // one entry per violated criterion (empty if supported)
};

// Linear-interpolation percentile statistics for one condition (fills min..max, spread_rel and
// samples_per_second from run_ms and the image geometry). Throws std::invalid_argument if empty.
void finalize_condition(ConditionStats* c, int width, int height, int spp);

// Verdict rule (fixed before measurement): "hypothesis_supported" iff speedup >= kSpeedupTarget AND
// every quality check passes AND vram_within_budget AND no condition has queue_overflows,
// invalid_hits or nonfinite_pixels; otherwise "hypothesis_weakened" with one reason per violation.
// Fills speedup, speedup_meets_target, vram_within_budget, verdict, reasons from conditions+quality.
void apply_verdict(BenchResult* r);

// Runs the protocol on a finalized scene: for each of megakernel and wavefront: warm-up renders,
// `reps` timed renders (stats.render_ms), one untimed counting run, wavefront stage-timing run;
// quality checks per DESIGN.md section 8 (bitwise between the GPU arms; CPU vs each GPU arm when
// cpu_spp > 0, comparing at cpu_spp for all three); then apply_verdict. Throws std::runtime_error
// on CUDA failure. Seeds, spp and compiler settings are identical across conditions.
BenchResult run_benchmark(const Scene& scene, const BenchConfig& cfg);

// schema_version 1. JSON keys are exactly the struct field names (top level: schema_version, scene_name,
// scene_hash (16 hex digits string), width, height, spp, max_depth, reps, warmup, seed, device_name,
// driver_version, cuda_runtime_version, conditions[], quality[], speedup, speedup_meets_target,
// vram_within_budget, verdict, reasons[]; condition and quality objects use their field names;
// queue_high_water is an array of 5 numbers; uint64 values that can exceed 2^53 are strings).
// Every key is required. For the CPU checks QualityCheck.threshold is kSePixelFraction.
std::string bench_to_json(const BenchResult& r);
// Empty string if valid JSON and schema-valid (verdict consistent with apply_verdict's rule applied
// to the parsed values), else the first problem.
std::string validate_bench_json(const std::string& json_text);

}  // namespace wf
