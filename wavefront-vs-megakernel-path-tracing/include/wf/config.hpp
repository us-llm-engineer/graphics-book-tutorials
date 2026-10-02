// Command-line configuration for `render`.
#pragma once
#include <stdexcept>
#include <string>

namespace wf {

enum class Mode { Cpu, Megakernel, Wavefront };

struct Config {
    std::string scene = "cornell";   // procedural name or path to an .obj file
    int width = 1280, height = 720;
    int spp = 64;
    unsigned seed = 1234;
    int max_depth = 8;
    Mode mode = Mode::Wavefront;
    std::string report_path;         // --report <path>; empty = do not write
    std::string out_prefix;          // --out <prefix>: writes <prefix>.png and <prefix>.pfm
    int threads = 0;                 // CPU arm; 0 = hardware_concurrency
    int reps = 1;                    // timed repetitions (GPU modes)
    int warmup = 0;                  // untimed warm-up renders
};

struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Parses: render --scene S --width N --height N --spp N --seed N --max-depth N
//         --mode {cpu,megakernel,wavefront} --report PATH --out PREFIX --threads N --reps N --warmup N
// argv[0] is the program name, argv[1] must be "render". Throws ConfigError with a message that
// names the offending flag for: unknown flag, missing value, non-numeric value, width/height
// outside 1..8192, spp < 1, max-depth outside 1..64, reps < 1, warmup < 0, unknown mode.
Config parse_args(int argc, const char* const* argv);

std::string usage();
std::string mode_name(Mode m);  // "cpu" | "megakernel" | "wavefront"

}  // namespace wf
