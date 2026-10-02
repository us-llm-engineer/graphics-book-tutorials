// Versioned report.json writer and validator.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace wf {

constexpr int kReportSchemaVersion = 1;

struct DeviceInfo {
    std::string name = "cpu";
    std::string driver_version;        // empty for cpu
    std::string cuda_runtime_version;  // empty for cpu
    std::string cuda_driver_version;   // empty for cpu
    std::string compute_capability;    // "7.5" or empty
    uint64_t total_memory_bytes = 0;
};

struct Report {
    std::string git_revision;          // empty -> written as null
    std::string scene_name;
    uint64_t scene_hash = 0;
    int width = 0, height = 0, spp = 0, max_depth = 0;
    unsigned seed = 0;
    std::string mode;                  // cpu|megakernel|wavefront
    DeviceInfo device;
    std::vector<double> run_ms;        // timed repetitions
    double samples_per_second = 0.0;   // width*height*spp / median seconds
    double rays_per_second = 0.0;      // 0 if ray counting was not run (written as null)
    uint64_t peak_vram_bytes = 0;      // 0 for cpu
    long long queue_high_water[5] = {0, 0, 0, 0, 0};  // lambert, metal, dielectric, shadow, active
    std::string image_png, image_pfm;  // paths, may be empty
    uint64_t pfm_hash = 0;             // fnv1a64 of the float RGB buffer bytes
    long long nonfinite_pixels = 0, queue_overflows = 0, invalid_hits = 0;
};

// Median, p10, p90 use linear interpolation between order statistics.
double percentile(std::vector<double> v, double p);  // p in [0,1]; throws on empty input

// Serialises to the schema below (schema_version = 1). Keys:
// schema_version, git_revision, scene{name,hash}, config{width,height,spp,seed,max_depth,mode},
// device{name,driver_version,cuda_runtime_version,cuda_driver_version,compute_capability,
// total_memory_bytes}, timings_ms{runs[],min,median,p10,p90,max},
// throughput{samples_per_second,rays_per_second}, peak_vram_bytes,
// queue_high_water{lambert,metal,dielectric,shadow,active}, outputs{png,pfm,pfm_hash},
// validity{nonfinite_pixels,queue_overflows,invalid_hits}. Hashes are 16 hex digit strings.
// Empty strings in `device` for a cpu report are written as empty JSON strings.
std::string report_to_json(const Report& r);

// Returns an empty string if `json_text` is valid JSON and satisfies the schema (all keys
// present with the right types, schema_version == kReportSchemaVersion, runs non-empty,
// min <= median <= max, hashes 16 hex digits), otherwise the first problem found.
std::string validate_report_json(const std::string& json_text);

}  // namespace wf
