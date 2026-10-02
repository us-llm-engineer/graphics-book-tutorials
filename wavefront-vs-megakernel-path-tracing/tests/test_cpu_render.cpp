// CPU renderer tests: furnace oracle, determinism, thread independence, variance buffer.
// Needs src/scenes.cpp, bvh_build.cpp, cpu_renderer.cpp, image_io.cpp to link.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "wf/film.hpp"
#include "wf/path_core.hpp"
#include "wf/renderer.hpp"
#include "wf/scene.hpp"

#include "test_framework.hpp"

using namespace wf;

namespace {

const Scene& cached(const std::string& name, int w, int h) {
    static std::map<std::string, Scene> cache;
    std::string key = name + ":" + std::to_string(w) + "x" + std::to_string(h);
    auto it = cache.find(key);
    if (it == cache.end()) it = cache.emplace(key, make_scene(name, w, h)).first;
    return it->second;
}

RenderResult run(const Scene& sc, int spp, uint32_t seed, int depth, int threads, RenderStats* st = nullptr) {
    RenderParams p;
    p.spp = spp; p.seed = seed; p.max_depth = depth; p.threads = threads;
    RenderStats local;
    return render_cpu(sc, p, st ? st : &local);
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

double pixel_lum(const RenderResult& r, int i) {
    return luminance(Vec3{r.rgb[3 * i], r.rgb[3 * i + 1], r.rgb[3 * i + 2]});
}
double mean_lum(const RenderResult& r) {
    double s = 0;
    int n = r.width * r.height;
    for (int i = 0; i < n; ++i) s += pixel_lum(r, i);
    return s / n;
}
// Standard error of the image-mean luminance: pixels are independent, Var(mean pixel) = lum_var / spp,
// Var(image mean) = sum(lum_var / spp) / N^2.
double mean_se(const RenderResult& r) {
    double s = 0;
    int n = r.width * r.height;
    for (int i = 0; i < n; ++i) s += (double)r.lum_var[i] / r.spp;
    return std::sqrt(s) / n;
}

}  // namespace

TEST(furnace_image_mean_matches_le_over_one_minus_rho_within_3_standard_errors) {
    // Oracle (Pharr-Humphreys 16.2.2): closed Lambertian shell with emission Le and albedo rho has
    // radiance Le / (1 - rho) everywhere. Le and rho are read from the scene's emissive material.
    const Scene& sc = cached("furnace", 32, 18);
    Vec3 Le{0, 0, 0}, rho{0, 0, 0};
    int found = 0;
    for (const Sphere& s : sc.spheres) {
        const Material& m = sc.materials[s.material];
        if (m.emission.x > 0 || m.emission.y > 0 || m.emission.z > 0) { Le = m.emission; rho = m.albedo; ++found; }
    }
    CHECK(found >= 1);
    CHECK(sc.env.x == 0.0f && sc.env.y == 0.0f && sc.env.z == 0.0f);
    CHECK(rho.x < 1.0f && rho.y < 1.0f && rho.z < 1.0f);
    if (found < 1 || !(rho.x < 1.0f && rho.y < 1.0f && rho.z < 1.0f)) return;
    Vec3 expect_rgb{Le.x / (1.0f - rho.x), Le.y / (1.0f - rho.y), Le.z / (1.0f - rho.z)};
    double expect = luminance(expect_rgb);
    CHECK(expect > 0.0);

    RenderResult r = run(sc, 256, 1234, 64, 0);
    CHECK(r.rgb.size() == (size_t)(3 * 32 * 18));
    double mean = mean_lum(r);
    double se = mean_se(r);
    // Depth cap of 64 drops at most a rho_max^60 fraction; 1e-4 relative covers float accumulation.
    double rmax = std::max(rho.x, std::max(rho.y, rho.z));
    double tol = 3.0 * se + expect * std::pow((double)rmax, 60.0) + 1e-4 * expect;
    std::printf("  furnace: mean %.6f expected %.6f se %.6f z=%.2f\n", mean, expect, se, (mean - expect) / std::max(se, 1e-12));
    CHECK_NEAR(mean, expect, tol);

    // Per pixel the same oracle: >= 97% of pixels within 3 own standard errors (expected ~99.7%).
    int ok = 0, n = r.width * r.height;
    for (int i = 0; i < n; ++i) {
        double pse = std::sqrt((double)r.lum_var[i] / r.spp);
        if (std::fabs(pixel_lum(r, i) - expect) <= 3.0 * pse + 1e-4 * expect) ++ok;
    }
    std::printf("  furnace: %d/%d pixels within 3 se\n", ok, n);
    CHECK((double)ok / n >= 0.97);
}

TEST(furnace_has_no_black_pixels_because_the_shell_is_closed) {
    const Scene& sc = cached("furnace", 32, 18);
    RenderResult r = run(sc, 16, 7, 64, 0);
    int black = 0;
    for (int i = 0; i < r.width * r.height; ++i) black += !(pixel_lum(r, i) > 0.0);
    CHECK(black == 0);
}

TEST(render_is_deterministic_bitwise_across_two_runs) {
    const Scene& c = cached("cornell", 32, 18);
    RenderResult a = run(c, 8, 99, 8, 0), b = run(c, 8, 99, 8, 0);
    CHECK(bitwise_equal(a.rgb, b.rgb));
    CHECK(bitwise_equal(a.lum_var, b.lum_var));
    const Scene& f = cached("furnace", 32, 18);
    RenderResult fa = run(f, 16, 99, 16, 0), fb = run(f, 16, 99, 16, 0);
    CHECK(bitwise_equal(fa.rgb, fb.rgb));
}

TEST(render_result_is_independent_of_thread_count_bitwise) {
    // Each pixel sample is one path keyed only by (seed, pixel, sample); partitioning must not matter.
    const Scene& c = cached("cornell", 32, 18);
    RenderResult t1 = run(c, 8, 5, 8, 1), t3 = run(c, 8, 5, 8, 3), t8 = run(c, 8, 5, 8, 8);
    CHECK(bitwise_equal(t1.rgb, t8.rgb));
    CHECK(bitwise_equal(t1.rgb, t3.rgb));   // 3 does not divide 18 rows
    CHECK(bitwise_equal(t1.lum_var, t8.lum_var));
    CHECK(bitwise_equal(t1.lum_var, t3.lum_var));
}

TEST(render_with_more_threads_than_rows_and_odd_size_still_matches_single_thread) {
    const Scene& c = cached("cornell", 7, 5);
    RenderResult t1 = run(c, 4, 5, 8, 1), t8 = run(c, 4, 5, 8, 8);
    CHECK(t1.rgb.size() == (size_t)(3 * 7 * 5));
    CHECK(bitwise_equal(t1.rgb, t8.rgb));
    const Scene& one = cached("cornell", 1, 1);
    RenderResult p = run(one, 4, 5, 8, 8);
    CHECK(p.rgb.size() == 3u);
    CHECK(p.lum_var.size() == 1u);
}

TEST(changing_the_seed_changes_most_pixels) {
    const Scene& c = cached("cornell", 32, 18);
    RenderResult a = run(c, 8, 1, 8, 0), b = run(c, 8, 2, 8, 0);
    int diff = 0, n = a.width * a.height;
    for (int i = 0; i < n; ++i)
        diff += (a.rgb[3 * i] != b.rgb[3 * i] || a.rgb[3 * i + 1] != b.rgb[3 * i + 1] || a.rgb[3 * i + 2] != b.rgb[3 * i + 2]);
    // Pixels filled entirely by the light are identical for any seed; the rest must change.
    CHECK(diff > n / 2);
}

TEST(image_mean_is_statistically_the_same_for_two_seeds) {
    // Metamorphic: the estimator is unbiased for every seed, so two seeds agree within 4 combined SE.
    const Scene& c = cached("cornell", 32, 18);
    RenderResult a = run(c, 64, 11, 8, 0), b = run(c, 64, 12, 8, 0);
    double se = std::sqrt(mean_se(a) * mean_se(a) + mean_se(b) * mean_se(b));
    CHECK_NEAR(mean_lum(a), mean_lum(b), 4.0 * se + 1e-6);
    CHECK(mean_lum(a) != mean_lum(b));   // and they are different samples
}

TEST(cornell_64x36_has_no_nonfinite_pixels_negative_values_or_invalid_hits) {
    const Scene& c = cached("cornell", 64, 36);
    RenderStats st;
    RenderResult r = run(c, 16, 1234, 8, 0, &st);
    CHECK(r.width == 64 && r.height == 36 && r.spp == 16);
    CHECK(r.rgb.size() == (size_t)(3 * 64 * 36));
    CHECK(count_nonfinite(r.rgb) == 0);
    int neg = 0;
    for (float v : r.rgb) neg += (v < 0.0f);   // radiance is non-negative
    CHECK(neg == 0);
    CHECK(st.invalid_hits == 0);
    CHECK(st.render_ms > 0.0);
    double total = 0;
    for (int i = 0; i < 64 * 36; ++i) total += pixel_lum(r, i);
    CHECK(total > 0.0);   // the scene is lit
}

TEST(variance_buffer_has_one_nonnegative_finite_entry_per_pixel) {
    const Scene& c = cached("cornell", 64, 36);
    RenderResult r = run(c, 16, 1234, 8, 0);
    CHECK(r.lum_var.size() == (size_t)(64 * 36));
    int bad = 0, positive = 0;
    for (float v : r.lum_var) {
        bad += !(std::isfinite(v) && v >= 0.0f);
        positive += v > 0.0f;
    }
    CHECK(bad == 0);
    CHECK(positive > 64 * 36 / 4);   // a noisy diffuse scene must show variance in many pixels
}

TEST(single_sample_per_pixel_has_zero_variance_and_equals_trace_path) {
    // spp = 1: unbiased variance of one sample is defined as 0. The mean must equal the shared
    // estimator's path for sample 0 (px, py, sample, seed, max_depth).
    const Scene& c = cached("cornell", 16, 9);
    SceneView v = c.view();
    RenderResult r = run(c, 1, 321, 8, 0);
    int bad_var = 0, bad_val = 0;
    for (int py = 0; py < 9; ++py)
        for (int px = 0; px < 16; ++px) {
            int i = py * 16 + px;
            if (r.lum_var[i] != 0.0f) ++bad_var;
            Vec3 L = trace_path(v, px, py, 0, 321, 8);
            const float want[3] = {L.x, L.y, L.z};
            for (int k = 0; k < 3; ++k)
                if (std::fabs(r.rgb[3 * i + k] - want[k]) > 1e-5f * std::fabs(want[k]) + 1e-7f) ++bad_val;
        }
    CHECK(bad_var == 0);
    CHECK(bad_val == 0);
}

TEST(two_samples_give_the_hand_computed_mean_and_unbiased_variance) {
    // Oracle: for samples a, b the unbiased variance is (a-b)^2 / 2 and the mean (a+b)/2, with
    // a, b the luminances / radiances from trace_path for sample indices 0 and 1.
    const Scene& c = cached("cornell", 16, 9);
    SceneView v = c.view();
    RenderResult r = run(c, 2, 77, 8, 0);
    int bad_mean = 0, bad_var = 0;
    for (int py = 0; py < 9; ++py)
        for (int px = 0; px < 16; ++px) {
            int i = py * 16 + px;
            Vec3 a = trace_path(v, px, py, 0, 77, 8), b = trace_path(v, px, py, 1, 77, 8);
            Vec3 m = (a + b) * 0.5f;
            const float want[3] = {m.x, m.y, m.z};
            for (int k = 0; k < 3; ++k)
                if (std::fabs(r.rgb[3 * i + k] - want[k]) > 1e-5f * std::fabs(want[k]) + 1e-7f) ++bad_mean;
            double la = luminance(a), lb = luminance(b);
            double var = (la - lb) * (la - lb) / 2.0;
            if (std::fabs((double)r.lum_var[i] - var) > 1e-4 * var + 1e-5 * (la * la + lb * lb) + 1e-9) ++bad_var;
        }
    CHECK(bad_mean == 0);
    CHECK(bad_var == 0);
}

TEST(deeper_max_depth_never_darkens_any_pixel_and_adds_indirect_light) {
    // Metamorphic: with identical counter-based random numbers a depth-d path is a prefix of the
    // depth-(d+k) path and every extra contribution is >= 0, so each channel is monotone in max_depth.
    const Scene& c = cached("cornell", 32, 18);
    RenderResult d0 = run(c, 8, 3, 0, 0), d1 = run(c, 8, 3, 1, 0), d4 = run(c, 8, 3, 4, 0);
    int bad = 0;
    for (size_t i = 0; i < d0.rgb.size(); ++i) {
        if (d1.rgb[i] < d0.rgb[i] - 1e-6f * std::fabs(d0.rgb[i])) ++bad;
        if (d4.rgb[i] < d1.rgb[i] - 1e-6f * std::fabs(d1.rgb[i])) ++bad;
    }
    CHECK(bad == 0);
    CHECK(mean_lum(d4) > mean_lum(d1));
    CHECK(mean_lum(d1) > mean_lum(d0));
}

TEST(max_depth_zero_renders_only_directly_visible_emission) {
    // With depth 0 each path stops at its first vertex: a pixel is either an emitter (value > 0) or
    // black (env is 0 in cornell). So the number of distinct non-zero pixels is small versus full light.
    const Scene& c = cached("cornell", 32, 18);
    RenderResult d0 = run(c, 4, 3, 0, 0);
    int lit = 0, n = d0.width * d0.height;
    for (int i = 0; i < n; ++i) lit += pixel_lum(d0, i) > 0.0;
    CHECK(lit < n / 4);
    RenderResult full = run(c, 4, 3, 8, 0);
    int lit_full = 0;
    for (int i = 0; i < n; ++i) lit_full += pixel_lum(full, i) > 0.0;
    CHECK(lit_full > n * 3 / 4);
}
