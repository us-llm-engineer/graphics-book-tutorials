#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "wf/image_io.hpp"

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: imgdiff <img1> <img2> [--tol <tolerance>]\n";
            return 2;
        }

        std::string img1_path = argv[1];
        std::string img2_path = argv[2];
        double tolerance = 1e-5;

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--tol" && i + 1 < argc) {
                tolerance = std::stod(argv[++i]);
            } else {
                std::cerr << "Unknown option: " << arg << "\n";
                return 2;
            }
        }

        int width1, height1, width2, height2;
        std::vector<float> img1 = wf::read_pfm(img1_path, &width1, &height1);
        std::vector<float> img2 = wf::read_pfm(img2_path, &width2, &height2);

        if (width1 != width2 || height1 != height2) {
            std::cerr << "Image dimensions mismatch\n";
            return 2;
        }

        if (img1.size() != img2.size()) {
            std::cerr << "Image size mismatch\n";
            return 2;
        }

        double max_abs_rel = 0.0;
        double sum_rel = 0.0;
        int count = 0;
        double sum_a = 0.0, sum_b = 0.0;

        for (size_t i = 0; i < img1.size(); ++i) {
            float a = img1[i];
            float b = img2[i];

            if (std::isnan(a) || std::isnan(b) || std::isinf(a) || std::isinf(b)) {
                if (a != b) {
                    std::cerr << "Non-finite pixel mismatch\n";
                    return 2;
                }
                continue;
            }

            sum_a += a;
            sum_b += b;

            double ref = std::max(std::abs((double)a), std::abs((double)b));
            if (ref > 1e-20) {
                double rel = std::abs((double)a - (double)b) / ref;
                max_abs_rel = std::max(max_abs_rel, rel);
                sum_rel += rel;
                count++;
            }
        }

        double mean_rel = 0.0;
        if (sum_a > 1e-12) {
            mean_rel = std::abs(sum_a - sum_b) / sum_a;
        }

        printf("max_abs_rel=%.17g\n", max_abs_rel);
        printf("mean_rel=%.17g\n", mean_rel);

        if (max_abs_rel <= tolerance) {
            return 0;
        } else {
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
