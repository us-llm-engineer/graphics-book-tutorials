// Tests for tonemap_to_srgb8, count_nonfinite, compare_results (include/wf/film.hpp).
// Oracles: the tone-map formula below, hand-computed ImageDiff values, the 3-standard-error rule from DESIGN.md s8.
// Luminance weights are not stated in film.hpp, so every comparison test uses gray pixels (r = g = b = v), for
// which any set of weights summing to 1 gives lum = v.
#include "test_framework.hpp"
#include "wf/film.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using wf::RenderResult;

namespace {

// Independent reference for the documented chain:
//   y = x * exposure;  r = y / (1 + y)  (Reinhard);  s = r <= 0.0031308 ? 12.92 r : 1.055 r^(1/2.4) - 0.055  (sRGB);
//   byte = round(255 * s).
double ref_byte(double x, double exposure = 1.0) {
    double y = x * exposure;
    double r = y / (1.0 + y);
    double s = r <= 0.0031308 ? 12.92 * r : 1.055 * std::pow(r, 1.0 / 2.4) - 0.055;
    return std::round(255.0 * s);
}

std::vector<float> gray_pixel(float v) { return {v, v, v}; }

RenderResult make_gray(int w, int h, int spp, const std::vector<float>& gray, float var) {
    RenderResult r;
    r.width = w; r.height = h; r.spp = spp;
    for (float g : gray) { r.rgb.push_back(g); r.rgb.push_back(g); r.rgb.push_back(g); }
    r.lum_var.assign(gray.size(), var);
    return r;
}

template <class F>
bool throws_invalid_argument(F f) {
    try { f(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

}  // namespace

// ================================================================ tonemap_to_srgb8

TEST(tonemap_black_is_zero_and_huge_values_clamp_to_255) {
    std::vector<uint8_t> black = wf::tonemap_to_srgb8({0.f, 0.f, 0.f});
    CHECK(black.size() == 3);
    CHECK(black.size() == 3 && black[0] == 0 && black[1] == 0 && black[2] == 0);
    std::vector<uint8_t> huge = wf::tonemap_to_srgb8({1e30f, 1e30f, 1e30f});
    CHECK(huge.size() == 3 && huge[0] == 255 && huge[1] == 255 && huge[2] == 255);
    std::vector<uint8_t> big = wf::tonemap_to_srgb8({1e6f, 1e6f, 1e6f});  // 1e6/(1+1e6) -> 255 after rounding
    CHECK(big.size() == 3 && big[0] == 255);
}

TEST(tonemap_output_has_three_bytes_per_pixel) {
    std::vector<float> img(7 * 3, 0.5f);  // 7 pixels
    CHECK(wf::tonemap_to_srgb8(img).size() == 21);
    CHECK(wf::tonemap_to_srgb8(img, 2.0f).size() == 21);
    CHECK(wf::tonemap_to_srgb8({}).empty());
}

TEST(tonemap_mid_values_match_independent_reinhard_plus_srgb_within_one_level) {
    // x = 1 -> r = 0.5 -> sRGB 0.7354 -> 187.5 (hand check); others from ref_byte.
    const float xs[] = {0.001f, 0.01f, 0.18f, 0.5f, 1.0f, 4.0f, 50.0f};
    for (float x : xs) {
        std::vector<uint8_t> o = wf::tonemap_to_srgb8({x, x, x});
        double want = ref_byte(x);
        if (o.size() != 3 || std::fabs(double(o[0]) - want) > 1.0) {
            std::printf("  FAIL x=%g: got %d, want %g +-1\n", x, o.empty() ? -1 : int(o[0]), want);
            ++wftest::failures();
        }
    }
    CHECK_NEAR(ref_byte(1.0), 188.0, 1.0);  // sanity of the oracle itself: 187.5 rounds to 188
}

TEST(tonemap_applies_the_curve_per_channel_in_rgb_order) {
    // Distinct channel values: a channel swap or a luminance-only mapping would change these bytes.
    std::vector<uint8_t> o = wf::tonemap_to_srgb8({0.0f, 1.0f, 4.0f});
    CHECK(o.size() == 3);
    if (o.size() == 3) {
        CHECK(o[0] == 0);
        CHECK_NEAR(o[1], ref_byte(1.0), 1.0);
        CHECK_NEAR(o[2], ref_byte(4.0), 1.0);
        CHECK(o[2] > o[1]);
    }
    // Pixels are independent and keep their order.
    std::vector<uint8_t> two = wf::tonemap_to_srgb8({4.0f, 4.0f, 4.0f, 0.0f, 0.0f, 0.0f});
    CHECK(two.size() == 6 && two[0] > 200 && two[3] == 0);
}

TEST(tonemap_is_monotone_non_decreasing_in_the_input) {
    std::vector<float> xs;
    for (int i = 0; i <= 400; ++i) xs.push_back(std::pow(10.0f, -4.0f + 8.0f * i / 400.0f));  // 1e-4 .. 1e4
    std::vector<float> img;
    for (float x : xs) { img.push_back(x); img.push_back(x); img.push_back(x); }
    std::vector<uint8_t> o = wf::tonemap_to_srgb8(img);
    CHECK(o.size() == img.size());
    int violations = 0;
    for (size_t i = 1; i < xs.size() && 3 * i < o.size(); ++i)
        if (o[3 * i] < o[3 * (i - 1)]) ++violations;
    CHECK(violations == 0);
    CHECK(o.size() == img.size() && o[0] <= 2);       // 1e-4 is nearly black
    CHECK(o.size() == img.size() && o[o.size() - 1] >= 254);  // 1e4 is nearly white
}

TEST(tonemap_exposure_two_brightens_and_equals_doubling_the_input) {
    std::vector<uint8_t> e1 = wf::tonemap_to_srgb8({0.1f, 0.1f, 0.1f}, 1.0f);
    std::vector<uint8_t> e2 = wf::tonemap_to_srgb8({0.1f, 0.1f, 0.1f}, 2.0f);
    std::vector<uint8_t> dbl = wf::tonemap_to_srgb8({0.2f, 0.2f, 0.2f}, 1.0f);  // metamorphic relation
    CHECK(e1.size() == 3 && e2.size() == 3 && dbl.size() == 3);
    if (e1.size() == 3 && e2.size() == 3 && dbl.size() == 3) {
        CHECK(e2[0] > e1[0]);
        CHECK_NEAR(e2[0], dbl[0], 1.0);
        CHECK_NEAR(e2[0], ref_byte(0.1, 2.0), 1.0);
        CHECK_NEAR(e1[0], ref_byte(0.1, 1.0), 1.0);
    }
}

TEST(tonemap_default_exposure_is_one_and_zero_exposure_is_black) {
    std::vector<float> img{0.3f, 0.6f, 1.2f};
    CHECK(wf::tonemap_to_srgb8(img) == wf::tonemap_to_srgb8(img, 1.0f));
    std::vector<uint8_t> z = wf::tonemap_to_srgb8({5.0f, 5.0f, 5.0f}, 0.0f);
    CHECK(z.size() == 3 && z[0] == 0 && z[1] == 0 && z[2] == 0);
    std::vector<uint8_t> half = wf::tonemap_to_srgb8({1.0f, 1.0f, 1.0f}, 0.5f);  // exposure < 1 darkens
    std::vector<uint8_t> one = wf::tonemap_to_srgb8({1.0f, 1.0f, 1.0f}, 1.0f);
    CHECK(half.size() == 3 && one.size() == 3 && half[0] < one[0]);
}

// ================================================================ count_nonfinite

TEST(count_nonfinite_counts_pixels_not_channels) {
    // One pixel with three NaN channels counts once.
    CHECK(wf::count_nonfinite({kNaN, kNaN, kNaN}) == 1);
    // One pixel with one bad channel counts once.
    CHECK(wf::count_nonfinite({1.f, kNaN, 2.f}) == 1);
    // Mixed: finite, 3xNaN, +inf in R, -inf in G, NaN in B, finite -> 4 bad pixels out of 6.
    std::vector<float> img{1.f, 2.f, 3.f,
                           kNaN, kNaN, kNaN,
                           kInf, 0.f, 0.f,
                           0.f, -kInf, 0.f,
                           0.f, 0.f, kNaN,
                           4.f, 5.f, 6.f};
    CHECK(wf::count_nonfinite(img) == 4);
}

TEST(count_nonfinite_is_zero_for_finite_and_empty_images_including_extremes) {
    CHECK(wf::count_nonfinite({}) == 0);
    CHECK(wf::count_nonfinite({0.f, -1.f, 3.4028235e38f, 1.4e-45f, -0.f, 7.f}) == 0);
}

// ================================================================ compare_results

TEST(compare_identical_images_has_zero_difference_and_full_agreement) {
    RenderResult a = make_gray(2, 2, 16, {0.5f, 1.0f, 2.0f, 4.0f}, 64.0f);
    RenderResult b = a;
    wf::ImageDiff d = wf::compare_results(a, b);
    CHECK_NEAR(d.max_abs_rel, 0.0, 0.0);
    CHECK_NEAR(d.mean_rel, 0.0, 0.0);
    CHECK_NEAR(d.frac_within_3se, 1.0, 0.0);
    CHECK(d.nonfinite == 0);
}

TEST(compare_max_abs_rel_is_largest_channel_difference_over_the_larger_magnitude) {
    // 1 pixel: a = (1,2,3), b = (1,2,4). Per channel: 0, 0, |3-4|/max(3,4) = 0.25.
    RenderResult a, b;
    a.width = b.width = 1; a.height = b.height = 1; a.spp = b.spp = 4;
    a.rgb = {1.f, 2.f, 3.f}; b.rgb = {1.f, 2.f, 4.f};
    a.lum_var = {1.f}; b.lum_var = {1.f};
    CHECK_NEAR(wf::compare_results(a, b).max_abs_rel, 0.25, 1e-9);
    CHECK_NEAR(wf::compare_results(b, a).max_abs_rel, 0.25, 1e-9);  // symmetric: denominator is max(|a|,|b|)
    // Zero versus 0.5: 0.5 / max(0, 0.5, 1e-6) = 1.
    a.rgb = {0.f, 0.f, 0.f}; b.rgb = {0.f, 0.5f, 0.f};
    CHECK_NEAR(wf::compare_results(a, b).max_abs_rel, 1.0, 1e-9);
    // Both zero: 0 / 1e-6 = 0, not NaN.
    b.rgb = {0.f, 0.f, 0.f};
    CHECK_NEAR(wf::compare_results(a, b).max_abs_rel, 0.0, 0.0);
}

TEST(compare_mean_rel_divides_by_the_mean_of_the_first_image) {
    // Constant gray images: mean(a) = 2, mean(b) = 1. |2-1| / mean(a) = 0.5; swapped arguments give 1/1 = 1.
    RenderResult a = make_gray(2, 1, 8, {2.f, 2.f}, 1.0f);
    RenderResult b = make_gray(2, 1, 8, {1.f, 1.f}, 1.0f);
    CHECK_NEAR(wf::compare_results(a, b).mean_rel, 0.5, 1e-9);
    CHECK_NEAR(wf::compare_results(b, a).mean_rel, 1.0, 1e-9);
}

TEST(compare_frac_within_3se_uses_the_combined_standard_error_of_both_images) {
    // spp = 16, lum_var = 64 in both images -> se = sqrt(64/16) = 2 each; combined se = sqrt(2^2+2^2) = 2.828;
    // threshold = 3 * 2.828 = 8.485 (a rule using 3*se = 6 or 3*(sea+seb) = 12 would classify differently).
    // Pixel 0: diff 0 (inside).  Pixel 1: diff 8.0 (inside).  Pixel 2: diff 9.0 (outside).  Pixel 3: diff 20 (outside).
    RenderResult a = make_gray(4, 1, 16, {10.f, 10.f, 10.f, 10.f}, 64.0f);
    RenderResult b = make_gray(4, 1, 16, {10.f, 18.f, 19.f, 30.f}, 64.0f);
    wf::ImageDiff d = wf::compare_results(a, b);
    CHECK_NEAR(d.frac_within_3se, 0.5, 1e-12);
    CHECK(d.nonfinite == 0);
}

TEST(compare_pixel_differing_by_ten_standard_errors_is_outside_and_the_rest_are_inside) {
    // se = 1 per image (var 16, spp 16), combined 1.414: a 10-se difference (10) exceeds 4.243; 3-se (3) does not.
    RenderResult a = make_gray(4, 1, 16, {5.f, 5.f, 5.f, 5.f}, 16.0f);
    RenderResult b = make_gray(4, 1, 16, {5.f, 8.f, 5.f, 15.f}, 16.0f);
    wf::ImageDiff d = wf::compare_results(a, b);
    CHECK_NEAR(d.frac_within_3se, 0.75, 1e-12);
    // exactly one pixel in a 1-pixel image outside -> fraction 0
    RenderResult a1 = make_gray(1, 1, 16, {5.f}, 16.0f);
    RenderResult b1 = make_gray(1, 1, 16, {15.f}, 16.0f);
    CHECK_NEAR(wf::compare_results(a1, b1).frac_within_3se, 0.0, 0.0);
}

TEST(compare_se_scales_with_variance_over_spp_not_the_other_way_round) {
    // var 64, spp 16 -> se 2 (threshold 8.485). If var and spp were swapped, se would be 0.5 (threshold 2.12) and a
    // difference of 8 would be outside. It must be inside.
    RenderResult a = make_gray(1, 1, 16, {10.f}, 64.0f);
    RenderResult b = make_gray(1, 1, 16, {18.f}, 64.0f);
    CHECK_NEAR(wf::compare_results(a, b).frac_within_3se, 1.0, 0.0);
    // Quadrupling spp halves se: spp 64, var 64 -> se 1, threshold 4.243; difference 8 is now outside.
    RenderResult a4 = make_gray(1, 1, 64, {10.f}, 64.0f);
    RenderResult b4 = make_gray(1, 1, 64, {18.f}, 64.0f);
    CHECK_NEAR(wf::compare_results(a4, b4).frac_within_3se, 0.0, 0.0);
}

TEST(compare_mismatched_dimensions_throw_invalid_argument) {
    RenderResult a = make_gray(3, 2, 4, {1, 1, 1, 1, 1, 1}, 1.0f);
    RenderResult wide = make_gray(4, 2, 4, {1, 1, 1, 1, 1, 1, 1, 1}, 1.0f);
    RenderResult tall = make_gray(3, 3, 4, {1, 1, 1, 1, 1, 1, 1, 1, 1}, 1.0f);
    RenderResult transposed = make_gray(2, 3, 4, {1, 1, 1, 1, 1, 1}, 1.0f);  // same pixel count, different shape
    CHECK(throws_invalid_argument([&] { (void)wf::compare_results(a, wide); }));
    CHECK(throws_invalid_argument([&] { (void)wf::compare_results(a, tall); }));
    CHECK(throws_invalid_argument([&] { (void)wf::compare_results(a, transposed); }));
    // Positive counterpart: identical dimensions are accepted.
    RenderResult same = make_gray(3, 2, 4, {1, 1, 1, 1, 1, 1}, 1.0f);
    CHECK(!throws_invalid_argument([&] { (void)wf::compare_results(a, same); }));
}

TEST(compare_counts_nonfinite_channels_from_either_image) {
    // header: "non-finite channels across both images".
    RenderResult a = make_gray(2, 1, 4, {1.f, 2.f}, 1.0f);
    RenderResult b = make_gray(2, 1, 4, {1.f, 2.f}, 1.0f);
    CHECK(wf::compare_results(a, b).nonfinite == 0);
    a.rgb[1] = kNaN;  // one bad channel in a
    CHECK(wf::compare_results(a, b).nonfinite == 1);
    CHECK(wf::compare_results(b, a).nonfinite == 1);  // either position
    b.rgb[3] = kNaN; b.rgb[4] = kInf; b.rgb[5] = -kInf;  // three bad channels in b's second pixel
    CHECK(wf::compare_results(a, b).nonfinite == 4);
}
