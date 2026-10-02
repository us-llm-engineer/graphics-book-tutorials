// Render results, tone mapping and image comparison helpers.
#include "wf/film.hpp"
#include <cmath>
#include <algorithm>

namespace wf {

std::vector<uint8_t> tonemap_to_srgb8(const std::vector<float>& rgb, float exposure) {
    std::vector<uint8_t> result(rgb.size());

    for (size_t i = 0; i < rgb.size(); ++i) {
        // Apply exposure
        float x = rgb[i] * exposure;

        // Reinhard tone mapping: x / (1 + x)
        float tone_mapped = x / (1.0f + x);

        // sRGB transfer function (companding)
        float srgb;
        if (tone_mapped <= 0.0031308f) {
            srgb = 12.92f * tone_mapped;
        } else {
            srgb = 1.055f * std::pow(tone_mapped, 1.0f / 2.4f) - 0.055f;
        }

        // Clamp to [0, 1] and convert to [0, 255]
        srgb = std::max(0.0f, std::min(1.0f, srgb));
        result[i] = (uint8_t)(srgb * 255.0f + 0.5f);
    }

    return result;
}

int count_nonfinite(const std::vector<float>& rgb) {
    int count = 0;
    // Count pixels (groups of 3 floats) that have any non-finite channel
    for (size_t i = 0; i + 2 < rgb.size(); i += 3) {
        if (!std::isfinite(rgb[i]) || !std::isfinite(rgb[i+1]) || !std::isfinite(rgb[i+2])) {
            ++count;
        }
    }
    return count;
}

ImageDiff compare_results(const RenderResult& a, const RenderResult& b) {
    if (a.width != b.width || a.height != b.height) {
        throw std::invalid_argument("compare_results: image dimensions mismatch");
    }

    ImageDiff diff;
    int n_pixels = a.width * a.height;

    if (n_pixels == 0) {
        return diff;
    }

    // Count non-finite channels
    for (size_t i = 0; i < a.rgb.size(); ++i) {
        if (!std::isfinite(a.rgb[i]) || !std::isfinite(b.rgb[i])) {
            ++diff.nonfinite;
        }
    }

    // Compute per-pixel differences
    diff.max_abs_rel = 0.0;
    double mean_a = 0.0, mean_b = 0.0;
    int pixels_within_3se = 0;

    for (int i = 0; i < n_pixels; ++i) {
        // Compute mean for luminance calculation
        float lum_a = luminance(Vec3{a.rgb[3 * i], a.rgb[3 * i + 1], a.rgb[3 * i + 2]});
        float lum_b = luminance(Vec3{b.rgb[3 * i], b.rgb[3 * i + 1], b.rgb[3 * i + 2]});

        mean_a += lum_a;
        mean_b += lum_b;

        // Per-channel relative difference
        for (int c = 0; c < 3; ++c) {
            float va = a.rgb[3 * i + c];
            float vb = b.rgb[3 * i + c];

            if (std::isfinite(va) && std::isfinite(vb)) {
                float max_v = std::max(std::abs(va), std::abs(vb));
                max_v = std::max(max_v, 1e-6f);
                float rel_diff = std::abs(va - vb) / max_v;
                diff.max_abs_rel = std::max(diff.max_abs_rel, (double)rel_diff);
            }
        }

        // Check if luminance difference is within 3 SE
        if (std::isfinite(lum_a) && std::isfinite(lum_b)) {
            float sea = std::sqrt(a.lum_var[i] / a.spp);
            float seb = std::sqrt(b.lum_var[i] / b.spp);
            float combined_se = 3.0f * std::sqrt(sea * sea + seb * seb);
            if (std::abs(lum_a - lum_b) <= combined_se) {
                ++pixels_within_3se;
            }
        }
    }

    mean_a /= n_pixels;
    mean_b /= n_pixels;

    if (mean_a > 1e-12) {
        diff.mean_rel = std::abs(mean_a - mean_b) / mean_a;
    } else {
        diff.mean_rel = std::abs(mean_a - mean_b);
    }

    diff.frac_within_3se = (double)pixels_within_3se / n_pixels;

    return diff;
}

}  // namespace wf
