// Command-line configuration for `render`.
#include "wf/config.hpp"
#include <sstream>
#include <stdexcept>
#include <charconv>
#include <cstring>

namespace wf {

std::string usage() {
    return R"(Usage: render --scene NAME --width W --height H --spp N --seed S --max-depth D
       --mode {cpu,megakernel,wavefront} --report PATH --out PREFIX --threads N --reps N --warmup N

All flags are optional and have defaults:
  --scene NAME: "cornell" (or path to .obj file)
  --width W: 1280
  --height H: 720
  --spp N: 64
  --seed S: 1234
  --max-depth D: 8
  --mode: wavefront
  --report PATH: (no report)
  --out PREFIX: (no output files)
  --threads N: 0 (hardware concurrency)
  --reps N: 1
  --warmup N: 0
)";
}

std::string mode_name(Mode m) {
    switch (m) {
        case Mode::Cpu: return "cpu";
        case Mode::Megakernel: return "megakernel";
        case Mode::Wavefront: return "wavefront";
    }
    return "unknown";
}

// Strict integer parsing: whole token must be consumed
static bool parse_int_strict(const char* s, int& out) {
    if (!s || !s[0]) return false;
    int value = 0;
    auto result = std::from_chars(s, s + std::strlen(s), value);
    if (result.ec != std::errc() || result.ptr != s + std::strlen(s)) {
        return false;  // parse error or unconsumed characters
    }
    out = value;
    return true;  // success
}

static bool parse_uint_strict(const char* s, unsigned& out) {
    if (!s || !s[0]) return false;
    unsigned value = 0;
    auto result = std::from_chars(s, s + std::strlen(s), value);
    if (result.ec != std::errc() || result.ptr != s + std::strlen(s)) {
        return false;  // parse error or unconsumed characters
    }
    out = value;
    return true;  // success
}

Config parse_args(int argc, const char* const* argv) {
    Config config;

    // argv[0] is the program name, argv[1] must be "render"
    if (argc < 2 || std::string(argv[1]) != "render") {
        throw ConfigError("argv[1] must be \"render\"");
    }

    // Parse flags starting from argv[2]
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];

        if (flag == "--scene") {
            if (i + 1 >= argc) throw ConfigError("--scene: missing value");
            config.scene = argv[++i];
        } else if (flag == "--width") {
            if (i + 1 >= argc) throw ConfigError("--width: missing value");
            int w;
            if (!parse_int_strict(argv[++i], w)) {
                throw ConfigError("--width: non-numeric value");
            }
            config.width = w;
            if (config.width < 1 || config.width > 8192) {
                throw ConfigError("--width: value outside 1..8192");
            }
        } else if (flag == "--height") {
            if (i + 1 >= argc) throw ConfigError("--height: missing value");
            int h;
            if (!parse_int_strict(argv[++i], h)) {
                throw ConfigError("--height: non-numeric value");
            }
            config.height = h;
            if (config.height < 1 || config.height > 8192) {
                throw ConfigError("--height: value outside 1..8192");
            }
        } else if (flag == "--spp") {
            if (i + 1 >= argc) throw ConfigError("--spp: missing value");
            int s;
            if (!parse_int_strict(argv[++i], s)) {
                throw ConfigError("--spp: non-numeric value");
            }
            config.spp = s;
            if (config.spp < 1) {
                throw ConfigError("--spp: value < 1");
            }
        } else if (flag == "--seed") {
            if (i + 1 >= argc) throw ConfigError("--seed: missing value");
            unsigned sd;
            if (!parse_uint_strict(argv[++i], sd)) {
                throw ConfigError("--seed: non-numeric value");
            }
            config.seed = sd;
        } else if (flag == "--max-depth") {
            if (i + 1 >= argc) throw ConfigError("--max-depth: missing value");
            int md;
            if (!parse_int_strict(argv[++i], md)) {
                throw ConfigError("--max-depth: non-numeric value");
            }
            config.max_depth = md;
            if (config.max_depth < 1 || config.max_depth > 64) {
                throw ConfigError("--max-depth: value outside 1..64");
            }
        } else if (flag == "--mode") {
            if (i + 1 >= argc) throw ConfigError("--mode: missing value");
            std::string mode_str = argv[++i];
            if (mode_str == "cpu") {
                config.mode = Mode::Cpu;
            } else if (mode_str == "megakernel") {
                config.mode = Mode::Megakernel;
            } else if (mode_str == "wavefront") {
                config.mode = Mode::Wavefront;
            } else {
                throw ConfigError("--mode: unknown value \"" + mode_str + "\"");
            }
        } else if (flag == "--report") {
            if (i + 1 >= argc) throw ConfigError("--report: missing value");
            config.report_path = argv[++i];
        } else if (flag == "--out") {
            if (i + 1 >= argc) throw ConfigError("--out: missing value");
            config.out_prefix = argv[++i];
        } else if (flag == "--threads") {
            if (i + 1 >= argc) throw ConfigError("--threads: missing value");
            int t;
            if (!parse_int_strict(argv[++i], t)) {
                throw ConfigError("--threads: non-numeric value");
            }
            config.threads = t;
        } else if (flag == "--reps") {
            if (i + 1 >= argc) throw ConfigError("--reps: missing value");
            int r;
            if (!parse_int_strict(argv[++i], r)) {
                throw ConfigError("--reps: non-numeric value");
            }
            config.reps = r;
            if (config.reps < 1) {
                throw ConfigError("--reps: value < 1");
            }
        } else if (flag == "--warmup") {
            if (i + 1 >= argc) throw ConfigError("--warmup: missing value");
            int w;
            if (!parse_int_strict(argv[++i], w)) {
                throw ConfigError("--warmup: non-numeric value");
            }
            config.warmup = w;
            if (config.warmup < 0) {
                throw ConfigError("--warmup: value < 0");
            }
        } else {
            throw ConfigError("unknown flag: " + flag);
        }
    }

    return config;
}

}  // namespace wf
