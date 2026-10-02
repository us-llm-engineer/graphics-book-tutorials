#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <stdexcept>
#include "wf/bench.hpp"
#include "wf/scene.hpp"

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args;
        std::string current_flag;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg[0] == '-') {
                current_flag = arg;
                args[arg] = "";
            } else {
                if (current_flag.empty()) {
                    std::cerr << "Error: unexpected argument: " << arg << "\n";
                    return 2;
                }
                args[current_flag] = arg;
                current_flag = "";
            }
        }

        if (!current_flag.empty()) {
            std::cerr << "Error: missing value for " << current_flag << "\n";
            return 2;
        }

        auto get_required = [&](const std::string& flag) -> std::string {
            auto it = args.find(flag);
            if (it == args.end()) {
                std::cerr << "Error: missing required flag " << flag << "\n";
                return "";
            }
            if (it->second.empty()) {
                std::cerr << "Error: missing value for " << flag << "\n";
                return "";
            }
            return it->second;
        };

        std::string out = get_required("--out");
        if (out.empty()) return 2;
        
        std::string scene_name = get_required("--scene");
        if (scene_name.empty()) return 2;

        int width;
        {
            auto it = args.find("--width");
            if (it == args.end()) {
                std::cerr << "Error: missing required flag --width\n";
                return 2;
            }
            try {
                width = std::stoi(it->second);
            } catch (...) {
                std::cerr << "Error: invalid value for --width\n";
                return 2;
            }
        }

        int height = width * 9 / 16;
        {
            auto it = args.find("--height");
            if (it != args.end()) {
                try {
                    height = std::stoi(it->second);
                } catch (...) {
                    std::cerr << "Error: invalid value for --height\n";
                    return 2;
                }
            }
        }

        int spp = 64;
        {
            auto it = args.find("--spp");
            if (it != args.end()) {
                try {
                    spp = std::stoi(it->second);
                    if (spp <= 0) {
                        std::cerr << "Error: --spp must be positive\n";
                        return 2;
                    }
                } catch (...) {
                    std::cerr << "Error: invalid value for --spp\n";
                    return 2;
                }
            }
        }

        int reps = 5;
        {
            auto it = args.find("--reps");
            if (it != args.end()) {
                try {
                    reps = std::stoi(it->second);
                    if (reps <= 0) {
                        std::cerr << "Error: --reps must be positive\n";
                        return 2;
                    }
                } catch (...) {
                    std::cerr << "Error: missing value for --reps\n";
                    return 2;
                }
            }
        }

        int warmup = 2;
        {
            auto it = args.find("--warmup");
            if (it != args.end()) {
                try {
                    warmup = std::stoi(it->second);
                    if (warmup < 0) {
                        std::cerr << "Error: --warmup must be non-negative\n";
                        return 2;
                    }
                } catch (...) {
                    std::cerr << "Error: invalid value for --warmup\n";
                    return 2;
                }
            }
        }

        int max_depth = 8;
        {
            auto it = args.find("--max-depth");
            if (it != args.end()) {
                try {
                    max_depth = std::stoi(it->second);
                } catch (...) {
                    std::cerr << "Error: invalid value for --max-depth\n";
                    return 2;
                }
            }
        }

        int cpu_spp = 0;
        {
            auto it = args.find("--cpu-spp");
            if (it != args.end()) {
                try {
                    cpu_spp = std::stoi(it->second);
                } catch (...) {
                    std::cerr << "Error: invalid value for --cpu-spp\n";
                    return 2;
                }
            }
        }

        uint32_t seed = 1234;
        {
            auto it = args.find("--seed");
            if (it != args.end()) {
                try {
                    seed = std::stoul(it->second);
                } catch (...) {
                    std::cerr << "Error: invalid value for --seed\n";
                    return 2;
                }
            }
        }

        wf::Scene scene = wf::make_scene(scene_name, width, height);

        wf::BenchConfig cfg;
        cfg.reps = reps;
        cfg.warmup = warmup;
        cfg.cpu_spp = cpu_spp;
        cfg.params.spp = spp;
        cfg.params.seed = seed;
        cfg.params.max_depth = max_depth;

        wf::BenchResult result = wf::run_benchmark(scene, cfg);
        std::string json = wf::bench_to_json(result);

        std::ofstream output(out);
        if (!output.is_open()) {
            std::cerr << "Error: failed to open output file\n";
            return 2;
        }
        output << json;
        output.close();

        // Exit 0 for every valid result: the verdict (supported or weakened) is recorded in the JSON.
        return 0;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: --scene " << e.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
