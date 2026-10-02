// Render results, tone mapping and image comparison helpers (host only).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "wf/math.hpp"

namespace wf {

struct RenderResult {
    int width = 0, height = 0, spp = 0;
    std::vector<float> rgb;      // 3*w*h, mean radiance over spp samples
    std::vector<float> lum_var;  // w*h, unbiased per-pixel variance of per-sample luminance
};

// exposure scale, then Reinhard x/(1+x), then sRGB transfer; returns 8-bit RGB (3*w*h).
std::vector<uint8_t> tonemap_to_srgb8(const std::vector<float>& rgb, float exposure = 1.0f);

// Count of pixels with any non-finite channel.
int count_nonfinite(const std::vector<float>& rgb);

// Luminance everywhere in this header is wf::luminance (0.2126, 0.7152, 0.0722).
// mean_rel is asymmetric (divides by mean(a)); nonfinite counts non-finite CHANNELS of both images,
// whereas count_nonfinite counts PIXELS.
struct ImageDiff {
    double max_abs_rel = 0.0;       // max over pixels/channels of |a-b| / max(|a|,|b|,1e-6)
    double mean_rel = 0.0;          // |mean(a)-mean(b)| / max(mean(a), 1e-12)
    double frac_within_3se = 0.0;   // fraction of pixels with |lum(a)-lum(b)| <= 3*sqrt(sea^2+seb^2)
    int nonfinite = 0;              // non-finite channels across both images
};
// se = sqrt(lum_var / spp). Images must have equal dimensions, otherwise throws std::invalid_argument.
ImageDiff compare_results(const RenderResult& a, const RenderResult& b);

}  // namespace wf
