// wavefront arm correctness and validity.
// Oracles: the megakernel arm (same per-path math, no float atomics => bitwise identity, DESIGN s2/s8),
// the analytic furnace value Le/(1-rho), the CPU reference under the DESIGN s8 criteria,
// metamorphic relations (progressive == one-shot, instrumentation is observation only) and
// structural queue bounds that follow from "each path sits in at most one queue" (DESIGN s6).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
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

double rgb_sum(const RenderResult& r) {
    double s = 0;
    for (float v : r.rgb) s += v;
    return s;
}

// Independent reimplementation of the image-mean luminance and its standard error.
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

RenderResult render_with(const Scene& s, GpuMode mode, const RenderParams& p, RenderStats* st) {
    GpuRenderer g(s, mode);
    return g.render(p, st);
}

}  // namespace

TEST_CASE("wavefront images are bitwise identical to the megakernel on cornell, divergence and furnace", "[wavefront]") {
    // Given the same scene, seed 7, max_depth 8, when both arms render at spp 4 and spp 8, then rgb and
    // lum_var are bitwise equal (DESIGN s8: identical expected). Bitwise equality also pins
    // the float addition order into L (shadow contribution before the next emission), the material
    // routing and the active-list compaction, any of which changes rounding or the value.
    for (const char* name : {"cornell", "divergence", "furnace"}) {
        Scene s = make_scene(name, 64, 36);
        GpuRenderer mk(s, GpuMode::Megakernel);
        GpuRenderer wf(s, GpuMode::Wavefront);
        for (int spp : {4, 8}) {
            RenderStats a, b;
            RenderResult m = mk.render(make_params(spp), &a);
            RenderResult w = wf.render(make_params(spp), &b);
            INFO(name << " spp " << spp << " differing rgb floats " << count_bit_diff(m.rgb, w.rgb)
                      << " lum_var " << count_bit_diff(m.lum_var, w.lum_var));
            REQUIRE(rgb_sum(m) > 0.0);   // guard: two black images must not pass
            CHECK(bitwise_equal(m.rgb, w.rgb));
            CHECK(bitwise_equal(m.lum_var, w.lum_var));
            CHECK(w.spp == spp);
        }
    }
}

TEST_CASE("wavefront furnace mean matches Le/(1-rho) within 4.5 SE", "[wavefront]") {
    // Oracle: closed Lambertian emissive sphere renders Le/(1-rho) (Pharr-Humphreys Ch16.2.2);
    // max_depth 200 makes the truncation bias rho^200 negligible.
    Scene s = make_scene("furnace", 64, 36);
    float le = 0, rho = 0;
    bool found = false;
    for (const Material& m : s.materials)
        if (m.type == kLambert && m.emission.x > 0) { le = m.emission.x; rho = m.albedo.x; found = true; break; }
    REQUIRE(found);
    REQUIRE(rho > 0.0f);
    REQUIRE(rho < 1.0f);
    const double expect = (double)le / (1.0 - (double)rho);
    RenderStats st;
    RenderResult r = render_with(s, GpuMode::Wavefront, make_params(64, 7, 200), &st);
    double mean, se;
    mean_and_se(r, &mean, &se);
    INFO("mean=" << mean << " expect=" << expect << " se=" << se);
    CHECK(std::fabs(mean - expect) <= 4.5 * se + 1e-6);
    // Gross-bias guard: a huge se must not hide a dropped rho or a wrong 1/q (exact value computable).
    CHECK(std::fabs(mean - expect) <= 0.02 * expect);
    CHECK(st.invalid_hits == 0);
    CHECK(st.queue_overflows == 0);
}

TEST_CASE("wavefront agrees with the CPU reference on divergence 64x36 spp 64", "[wavefront]") {
    // Independent oracle (CPU arm, scalar trace_path) under the DESIGN s8 pre-registered criteria.
    // Divergence interleaves all four material types, so a wrong queue routing shows here.
    Scene s = make_scene("divergence", 64, 36);
    RenderStats sc, sg;
    RenderResult cpu = render_cpu(s, make_params(64), &sc);
    RenderResult gpu = render_with(s, GpuMode::Wavefront, make_params(64), &sg);
    REQUIRE(rgb_sum(cpu) > 0.0);
    REQUIRE(rgb_sum(gpu) > 0.0);
    ImageDiff d = compare_results(cpu, gpu);
    INFO("frac_within_3se=" << d.frac_within_3se << " mean_rel=" << d.mean_rel
                            << " max_abs_rel=" << d.max_abs_rel << " nonfinite=" << d.nonfinite);
    CHECK(d.frac_within_3se >= 0.995);   // DESIGN s8
    CHECK(d.mean_rel <= 0.005);          // DESIGN s8
    CHECK(d.nonfinite == 0);
}

TEST_CASE("wavefront renders are bitwise repeatable and a different seed changes the image", "[wavefront]") {
    // Given no float atomics and fixed compaction order, two renders on one GpuRenderer (and one on a
    // fresh renderer) are bitwise equal. A different seed perturbs nearly every channel while
    // the image means agree within 4.5 combined SE (same estimator); returning to seed 7 restores image 1.
    Scene s = make_scene("cornell", 64, 36);
    GpuRenderer g(s, GpuMode::Wavefront);
    RenderStats st;
    RenderResult a = g.render(make_params(8, 7), &st);
    RenderResult b = g.render(make_params(8, 7), &st);
    REQUIRE(rgb_sum(a) > 0.0);
    INFO("differing rgb floats: " << count_bit_diff(a.rgb, b.rgb));
    CHECK(same_image(a, b));
    GpuRenderer g2(s, GpuMode::Wavefront);
    CHECK(same_image(a, g2.render(make_params(8, 7), &st)));

    RenderResult c = g.render(make_params(8, 8), &st);
    CHECK(count_bit_diff(a.rgb, c.rgb) > a.rgb.size() / 2);   // independent MC noise
    double ma, sa, mc, sc;
    mean_and_se(a, &ma, &sa);
    mean_and_se(c, &mc, &sc);
    CHECK(std::fabs(ma - mc) <= 4.5 * std::sqrt(sa * sa + sc * sc));
    CHECK(same_image(a, g.render(make_params(8, 7), &st)));
}

TEST_CASE("wavefront progressive passes equal a one-shot render", "[wavefront]") {
    // Given reset(); run_passes(0,2); run_passes(2,2); download(4), then the result is bitwise
    // render(spp 4) (one path per pixel per pass, fixed order); reset() really discards the film.
    Scene s = make_scene("divergence", 64, 36);
    GpuRenderer g(s, GpuMode::Wavefront);
    RenderStats st;
    RenderParams p = make_params(4);
    RenderResult one = g.render(p, &st);
    REQUIRE(rgb_sum(one) > 0.0);
    g.reset();
    RenderStats s1, s2;
    g.run_passes(0, 2, p, &s1);
    g.run_passes(2, 2, p, &s2);
    RenderResult prog = g.download(4);
    CHECK(prog.spp == 4);
    CHECK(bitwise_equal(one.rgb, prog.rgb));
    CHECK(bitwise_equal(one.lum_var, prog.lum_var));
    // reset discards: after reset one pass equals a spp-1 render, not 1 + earlier samples.
    RenderResult ref1 = g.render(make_params(1), &st);
    g.run_passes(0, 3, make_params(1), &st);
    g.reset();
    g.run_passes(0, 1, make_params(1), &st);
    CHECK(bitwise_equal(ref1.rgb, g.download(1).rgb));
}

TEST_CASE("wavefront stats are valid and queue high-water marks obey the one-queue-per-path bound", "[wavefront]") {
    // Given one pass at 64x36 divergence, every path is active at bounce 0, so high_water[4] (active)
    // equals W*H exactly. Each path sits in at most one queue (DESIGN s6) so every queue high-water
    // <= W*H. With max_depth 1 only bounce 0 enqueues (depth == max_depth terminates before shading),
    // so the three material high-waters are simultaneous and their sum <= active; the shadow queue is fed
    // only by Lambertian shading, so shadow <= lambert. (For max_depth > 1 the per-queue maxima may occur
    // at different bounces, so the sum bound is only valid at max_depth 1.)
    const long long n = 64 * 36;
    Scene s = make_scene("divergence", 64, 36);
    GpuRenderer g(s, GpuMode::Wavefront);
    {
        RenderStats st;
        RenderResult r = g.render(make_params(1), &st);
        REQUIRE(rgb_sum(r) > 0.0);
        CHECK(st.queue_overflows == 0);
        CHECK(st.invalid_hits == 0);
        CHECK(st.queue_high_water[4] == n);
        for (int q = 0; q < 4; ++q) {
            INFO("queue " << q << " high-water " << st.queue_high_water[q]);
            CHECK(st.queue_high_water[q] >= 0);
            CHECK(st.queue_high_water[q] <= n);
        }
        // divergence has Lambertian, metal and dielectric surfaces in view: each queue is used.
        CHECK(st.queue_high_water[0] > 0);
        CHECK(st.queue_high_water[1] > 0);
        CHECK(st.queue_high_water[2] > 0);
        CHECK(st.queue_high_water[3] > 0);
    }
    {
        RenderStats st;
        g.render(make_params(1, 7, 1), &st);
        INFO("lambert " << st.queue_high_water[0] << " metal " << st.queue_high_water[1] << " dielectric "
                        << st.queue_high_water[2] << " shadow " << st.queue_high_water[3]
                        << " active " << st.queue_high_water[4]);
        CHECK(st.queue_high_water[4] == n);
        CHECK(st.queue_high_water[0] + st.queue_high_water[1] + st.queue_high_water[2] <= st.queue_high_water[4]);
        CHECK(st.queue_high_water[3] <= st.queue_high_water[0]);
        CHECK(st.queue_overflows == 0);
    }
}

TEST_CASE("stage_timing fills StageTimes and does not change the image", "[wavefront]") {
    // Given cornell (all-Lambertian walls), when stage_timing is on, extend, classify, compact and
    // shade_lambert are positive (kernels ran, events measured), no stage is negative, and the image is
    // bitwise the stage_timing=false image (instrumentation is observation only). With stage_timing off
    // StageTimes stays zero (the flag, not the render, produces it).
    Scene s = make_scene("cornell", 256, 144);
    GpuRenderer g(s, GpuMode::Wavefront);
    RenderParams off = make_params(2);
    RenderParams on = off;
    on.stage_timing = true;
    RenderStats so, sn;
    RenderResult a = g.render(off, &so);
    RenderResult b = g.render(on, &sn);
    const StageTimes& t = sn.stages;
    INFO("extend " << t.extend << " classify " << t.classify << " compact " << t.compact << " lambert " << t.shade_lambert);
    CHECK(t.extend > 0.0);
    CHECK(t.classify > 0.0);
    CHECK(t.compact > 0.0);
    CHECK(t.shade_lambert > 0.0);
    CHECK(t.shade_metal >= 0.0);
    CHECK(t.shade_dielectric >= 0.0);
    CHECK(t.shadow >= 0.0);
    CHECK(t.other >= 0.0);
    CHECK(so.stages.extend == 0.0);
    CHECK(so.stages.classify == 0.0);
    CHECK(so.stages.compact == 0.0);
    REQUIRE(rgb_sum(a) > 0.0);
    CHECK(same_image(a, b));
}

TEST_CASE("wavefront count_rays equals the megakernel counts and does not change the image", "[wavefront]") {
    // Both arms cast one extension ray per path per bounce and one shadow ray per Lambertian vertex, so
    // the counts for identical params are equal (same RNG draws, same paths). count_rays off => zero.
    for (const char* name : {"cornell", "divergence"}) {
        Scene s = make_scene(name, 64, 36);
        GpuRenderer mk(s, GpuMode::Megakernel);
        GpuRenderer wf(s, GpuMode::Wavefront);
        RenderStats mon, won, woff;
        RenderResult m = mk.render(make_params(2, 7, 8, true), &mon);
        RenderResult w = wf.render(make_params(2, 7, 8, true), &won);
        RenderResult w0 = wf.render(make_params(2, 7, 8, false), &woff);
        INFO(name << ": extension mk " << mon.extension_rays << " wf " << won.extension_rays
                  << " shadow mk " << mon.shadow_rays << " wf " << won.shadow_rays);
        CHECK(won.extension_rays > 0u);
        CHECK(won.shadow_rays > 0u);
        CHECK(won.extension_rays == mon.extension_rays);
        CHECK(won.shadow_rays == mon.shadow_rays);
        CHECK(won.extension_rays >= (uint64_t)64 * 36 * 2);   // at least one camera ray per pixel per pass
        CHECK(woff.extension_rays == 0u);
        CHECK(woff.shadow_rays == 0u);
        CHECK(same_image(w, w0));
        CHECK(same_image(m, w));
    }
}
