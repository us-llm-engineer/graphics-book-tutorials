// GPU megakernel render correctness and validity.
// Only GpuMode::Megakernel is constructed here; the wavefront arm is covered in round 2.
// Oracles: DESIGN s8 pre-registered criteria vs the CPU reference, the analytic furnace value
// Le/(1-rho), bitwise repeatability (no float atomics, DESIGN s2), progressive == one-shot
// metamorphic relation, and statistical equality of means across seeds.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "wf/film.hpp"
#include "wf/renderer.hpp"
#include "wf/scene.hpp"

using namespace wf;

namespace {

RenderParams make_params(int spp, uint32_t seed = 7, int depth = 8, bool count = false) {
    RenderParams p;
    p.spp = spp;
    p.seed = seed;
    p.max_depth = depth;
    p.count_rays = count;
    p.threads = 0;
    return p;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

size_t count_bit_diff(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) n += std::memcmp(&a[i], &b[i], 4) != 0;
    return n;
}

bool same_image(const RenderResult& a, const RenderResult& b) {
    return a.width == b.width && a.height == b.height && a.spp == b.spp &&
           bitwise_equal(a.rgb, b.rgb) && bitwise_equal(a.lum_var, b.lum_var);
}

// Independent reimplementation of the image-mean luminance and its standard error
// (pixels independent: se_mean = sqrt(sum var_i / spp) / n).
void mean_and_se(const RenderResult& r, double* mean, double* se) {
    const int n = r.width * r.height;
    double m = 0, v = 0;
    for (int i = 0; i < n; ++i) {
        m += 0.2126 * r.rgb[3 * i] + 0.7152 * r.rgb[3 * i + 1] + 0.0722 * r.rgb[3 * i + 2];
        v += r.lum_var[i];
    }
    *mean = m / n;
    *se = std::sqrt(v / r.spp) / n;
}

double rgb_sum(const RenderResult& r) {
    double s = 0;
    for (float v : r.rgb) s += v;
    return s;
}

void require_valid_image(const RenderResult& r, int w, int h, int spp) {
    REQUIRE(r.width == w);
    REQUIRE(r.height == h);
    REQUIRE(r.spp == spp);
    REQUIRE(r.rgb.size() == (size_t)3 * w * h);
    REQUIRE(r.lum_var.size() == (size_t)w * h);
    CHECK(count_nonfinite(r.rgb) == 0);
    bool var_ok = true;
    for (float v : r.lum_var) var_ok = var_ok && std::isfinite(v) && v >= 0.0f;
    CHECK(var_ok);
}

void compare_with_cpu(const char* scene_name) {
    Scene s = make_scene(scene_name, 64, 36);
    RenderStats sc, sg;
    RenderResult cpu = render_cpu(s, make_params(64), &sc);
    GpuRenderer g(s, GpuMode::Megakernel);
    RenderResult gpu = g.render(make_params(64), &sg);
    REQUIRE(rgb_sum(cpu) > 0.0);   // guard: both being black must not pass
    REQUIRE(rgb_sum(gpu) > 0.0);
    ImageDiff d = compare_results(cpu, gpu);
    INFO(scene_name << ": frac_within_3se=" << d.frac_within_3se << " mean_rel=" << d.mean_rel
                    << " max_abs_rel=" << d.max_abs_rel << " nonfinite=" << d.nonfinite);
    CHECK(d.frac_within_3se >= 0.995);   // DESIGN s8
    CHECK(d.mean_rel <= 0.005);          // DESIGN s8
    CHECK(d.nonfinite == 0);
}

}  // namespace

TEST_CASE("megakernel agrees with the CPU reference on cornell 64x36 spp 64", "[gpu_megakernel]") {
    // Given the same scene, seed and spp, when both arms render, then the pre-registered CPU-vs-GPU
    // criteria hold (>=99.5% pixels within 3 combined SE, mean_rel <= 0.5%, no non-finite).
    compare_with_cpu("cornell");
}

TEST_CASE("megakernel agrees with the CPU reference on divergence 64x36 spp 64", "[gpu_megakernel]") {
    // Divergence interleaves Lambertian, metal, dielectric and emissive materials, so a defect in
    // one material branch of the GPU path loop shows here and not on cornell.
    compare_with_cpu("divergence");
}

TEST_CASE("megakernel furnace mean matches Le/(1-rho) within 4.5 SE", "[gpu_megakernel]") {
    // Oracle: closed Lambertian emissive sphere of emission Le, albedo rho renders Le/(1-rho)
    // (Pharr-Humphreys Ch16.2.2). max_depth 200 makes truncation bias rho^200 negligible.
    Scene s = make_scene("furnace", 64, 36);
    float le = 0, rho = 0;
    bool found = false;
    for (const Material& m : s.materials)
        if (m.type == kLambert && m.emission.x > 0) { le = m.emission.x; rho = m.albedo.x; found = true; break; }
    REQUIRE(found);
    REQUIRE(rho > 0.0f);
    REQUIRE(rho < 1.0f);
    const double expect = (double)le / (1.0 - (double)rho);
    GpuRenderer g(s, GpuMode::Megakernel);
    RenderStats st;
    RenderResult r = g.render(make_params(64, 7, 200), &st);
    double mean, se;
    mean_and_se(r, &mean, &se);
    INFO("mean=" << mean << " expect=" << expect << " se=" << se);
    CHECK(std::fabs(mean - expect) <= 4.5 * se + 1e-6);
    // Gross-bias guard so a huge se cannot hide a missing rho or wrong 1/q (exact value computable).
    CHECK(std::fabs(mean - expect) <= 0.02 * expect);
    CHECK(st.invalid_hits == 0);
}

TEST_CASE("two renders on one GpuRenderer are bitwise identical", "[gpu_megakernel]") {
    // Given no float atomics, when the same params render twice on one renderer (render() must
    // reset accumulators), then rgb and lum_var are bitwise equal.
    Scene s = make_scene("cornell", 64, 36);
    GpuRenderer g(s, GpuMode::Megakernel);
    RenderStats st;
    RenderResult a = g.render(make_params(4, 7), &st);
    RenderResult b = g.render(make_params(4, 7), &st);
    REQUIRE(rgb_sum(a) > 0.0);
    INFO("differing rgb floats: " << count_bit_diff(a.rgb, b.rgb) << " lum_var: " << count_bit_diff(a.lum_var, b.lum_var));
    CHECK(bitwise_equal(a.rgb, b.rgb));
    CHECK(bitwise_equal(a.lum_var, b.lum_var));
    // A fresh renderer must also reproduce it (no state leaking via a reused object).
    GpuRenderer g2(s, GpuMode::Megakernel);
    RenderResult c = g2.render(make_params(4, 7), &st);
    CHECK(same_image(a, c));
}

TEST_CASE("a different seed changes the image but not its expectation", "[gpu_megakernel]") {
    // Given seeds 7 and 8, then most pixels differ bitwise (independent noise), while the image
    // means agree within 4.5 combined SE (same estimator). Returning to seed 7 restores image 1.
    Scene s = make_scene("cornell", 64, 36);
    GpuRenderer g(s, GpuMode::Megakernel);
    RenderStats st;
    RenderResult a = g.render(make_params(8, 7), &st);
    RenderResult b = g.render(make_params(8, 8), &st);
    CHECK(count_bit_diff(a.rgb, b.rgb) > a.rgb.size() / 2);   // independent MC noise perturbs nearly every channel
    double ma, sa, mb, sb;
    mean_and_se(a, &ma, &sa);
    mean_and_se(b, &mb, &sb);
    CHECK(std::fabs(ma - mb) <= 4.5 * std::sqrt(sa * sa + sb * sb));
    RenderResult c = g.render(make_params(8, 7), &st);
    CHECK(same_image(a, c));
}

TEST_CASE("progressive passes equal a one-shot render", "[gpu_megakernel]") {
    // Given reset, run_passes(0,2), run_passes(2,2), then download(4) is bitwise render(spp 4)
    // (one path per pixel per pass, fixed sample order); download_mean equals download().rgb.
    Scene s = make_scene("divergence", 64, 36);
    GpuRenderer g(s, GpuMode::Megakernel);
    RenderStats st;
    RenderParams p = make_params(4);
    RenderResult one = g.render(p, &st);
    REQUIRE(rgb_sum(one) > 0.0);
    g.reset();
    g.run_passes(0, 2, p, &st);
    RenderResult half = g.download(2);
    g.run_passes(2, 2, p, &st);
    RenderResult prog = g.download(4);
    CHECK(prog.spp == 4);
    CHECK(bitwise_equal(one.rgb, prog.rgb));
    CHECK(bitwise_equal(one.lum_var, prog.lum_var));
    // The partial after samples [0,2) equals a one-shot spp 2 render.
    RenderResult two = g.render(make_params(2), &st);
    CHECK(half.spp == 2);
    CHECK(bitwise_equal(two.rgb, half.rgb));
    // download_mean matches download().rgb.
    g.reset();
    g.run_passes(0, 4, p, &st);
    std::vector<float> mean;
    g.download_mean(4, &mean);
    CHECK(bitwise_equal(mean, one.rgb));
    // reset really discards: after reset, 1 pass equals spp-1 render, not 1 + earlier samples.
    RenderResult ref1 = g.render(make_params(1), &st);
    g.run_passes(0, 3, make_params(1), &st);
    g.reset();
    g.run_passes(0, 1, make_params(1), &st);
    CHECK(bitwise_equal(ref1.rgb, g.download(1).rgb));
}

TEST_CASE("edge image sizes 1x1, 33x17 and 1x720 render valid finite images", "[gpu_megakernel]") {
    // Non-multiple-of-32 and degenerate sizes: launch-grid bounds bugs show as crashes, NaNs,
    // negative variance, or untouched (zero) pixels. The box need not fill a wide frame (outer
    // columns may see empty space), so lit pixels are taken from the CPU reference, not assumed.
    struct Dim { int w, h; };
    for (Dim d : {Dim{1, 1}, Dim{33, 17}, Dim{1, 720}}) {
        INFO("size " << d.w << "x" << d.h);
        Scene s = make_scene("cornell", d.w, d.h);
        GpuRenderer g(s, GpuMode::Megakernel);
        RenderStats st;
        RenderResult r = g.render(make_params(4), &st);
        require_valid_image(r, d.w, d.h, 4);
        CHECK(rgb_sum(r) > 0.0);
        CHECK(st.invalid_hits == 0);
        CHECK(st.queue_overflows == 0);
        // Metamorphic: every pixel the CPU reference lit must also be lit on the GPU. A launch-grid
        // off-by-one leaves pixels untouched (zero); it cannot hide at the border, wherever the box is.
        RenderStats sc;
        RenderResult cpu = render_cpu(s, make_params(4), &sc);
        REQUIRE(rgb_sum(cpu) > 0.0);
        const int n = d.w * d.h;
        int cpu_lit = 0, gpu_zero_where_cpu_lit = 0;
        for (int i = 0; i < n; ++i) {
            const double cv = cpu.rgb[3 * i] + cpu.rgb[3 * i + 1] + cpu.rgb[3 * i + 2];
            const double gv = r.rgb[3 * i] + r.rgb[3 * i + 1] + r.rgb[3 * i + 2];
            if (cv > 0.0) { ++cpu_lit; gpu_zero_where_cpu_lit += !(gv > 0.0); }
        }
        INFO("cpu-lit pixels " << cpu_lit << ", of which zero on GPU " << gpu_zero_where_cpu_lit);
        CHECK(gpu_zero_where_cpu_lit == 0);
    }
}

TEST_CASE("stats are valid and count_rays only adds counters", "[gpu_megakernel]") {
    // divergence 1280x720 spp 1: no invalid hits, no overflows, positive render time, VRAM below
    // the 3.5e9 budget (DESIGN s1). count_rays off => ray counters zero; on => every pixel casts
    // at least its camera ray; image unchanged (instrumentation is observation only).
    {
        Scene s = make_scene("divergence", 1280, 720);
        GpuRenderer g(s, GpuMode::Megakernel);
        RenderStats st;
        RenderResult r = g.render(make_params(1), &st);
        require_valid_image(r, 1280, 720, 1);
        CHECK(st.invalid_hits == 0);
        CHECK(st.queue_overflows == 0);
        CHECK(st.render_ms > 0.0);
        CHECK(st.peak_vram_bytes > 0u);
        CHECK(st.peak_vram_bytes < 3500000000ull);
    }
    Scene s = make_scene("cornell", 64, 36);
    const uint64_t npix = 64 * 36;
    GpuRenderer g(s, GpuMode::Megakernel);
    RenderStats off, on;
    RenderResult a = g.render(make_params(2, 7, 8, false), &off);
    RenderResult b = g.render(make_params(2, 7, 8, true), &on);
    CHECK(off.extension_rays == 0u);
    CHECK(off.shadow_rays == 0u);
    CHECK(on.extension_rays >= npix * 2);   // >= one camera ray per pixel per pass
    CHECK(on.shadow_rays > 0u);             // cornell has Lambertian walls under an area light
    CHECK(same_image(a, b));
}
