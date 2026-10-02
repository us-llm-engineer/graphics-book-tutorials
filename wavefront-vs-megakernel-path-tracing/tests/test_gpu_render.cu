// GPU renderer arms: megakernel vs wavefront vs CPU reference, furnace oracle, stats, progressive API.
// Oracles: DESIGN section 2 (identical pixels between GPU arms), section 8 (pre-registered
// statistical criteria CPU vs GPU, furnace Le/(1-rho)), section 6 (queue capacity == path count),
// plus metamorphic relations (progressive == one-shot, same seed == same image).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "wf/film.hpp"
#include "wf/renderer.hpp"
#include "wf/scene.hpp"
#include "test_framework.hpp"

using namespace wf;

namespace {

constexpr uint32_t kSeed = 7;

RenderParams params(int spp, uint32_t seed = kSeed, int depth = 8, bool count = false) {
    RenderParams p;
    p.spp = spp; p.seed = seed; p.max_depth = depth; p.count_rays = count; p.threads = 0;
    return p;
}

RenderResult gpu_render(const Scene& s, GpuMode m, const RenderParams& p, RenderStats* st) {
    GpuRenderer r(s, m);
    return r.render(p, st);
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

size_t count_diff(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) n += std::memcmp(&a[i], &b[i], 4) != 0;
    return n;
}

void expect_identical(const RenderResult& a, const RenderResult& b, const std::string& label) {
    bool shape = a.width == b.width && a.height == b.height && a.spp == b.spp;
    CHECK(shape);
    bool rgb_ok = bitwise_equal(a.rgb, b.rgb);
    bool var_ok = bitwise_equal(a.lum_var, b.lum_var);
    if (!rgb_ok || !var_ok)
        std::printf("  [%s] rgb differing floats: %zu of %zu, lum_var differing: %zu of %zu\n", label.c_str(),
                    count_diff(a.rgb, b.rgb), a.rgb.size(), count_diff(a.lum_var, b.lum_var), a.lum_var.size());
    CHECK(rgb_ok);
    CHECK(var_ok);
}

void check_clean_stats(const RenderStats& st, const std::string& label) {
    if (st.queue_overflows != 0 || st.invalid_hits != 0)
        std::printf("  [%s] overflows=%lld invalid_hits=%lld\n", label.c_str(), st.queue_overflows, st.invalid_hits);
    CHECK(st.queue_overflows == 0);   // capacity == path count, so overflow is a bug
    CHECK(st.invalid_hits == 0);      // every hit has finite t and in-range prim
    CHECK(st.render_ms > 0.0);
}

void check_shape(const RenderResult& r, int w, int h, int spp) {
    CHECK(r.width == w);
    CHECK(r.height == h);
    CHECK(r.spp == spp);
    CHECK(r.rgb.size() == (size_t)3 * w * h);
    CHECK(r.lum_var.size() == (size_t)w * h);
    CHECK(count_nonfinite(r.rgb) == 0);
}

}  // namespace

TEST(megakernel_and_wavefront_produce_bitwise_identical_images) {
    // DESIGN s2: same float ops per path (fmad off), counter-based RNG -> identical pixels and variance.
    for (const char* name : {"cornell", "divergence", "furnace"}) {
        Scene s = make_scene(name, 64, 36);
        for (int spp : {4, 8}) {
            RenderStats sm, sw;
            auto m = gpu_render(s, GpuMode::Megakernel, params(spp), &sm);
            auto w = gpu_render(s, GpuMode::Wavefront, params(spp), &sw);
            check_shape(m, 64, 36, spp);
            check_shape(w, 64, 36, spp);
            expect_identical(m, w, std::string(name) + " spp" + std::to_string(spp));
            check_clean_stats(sm, std::string("mega ") + name);
            check_clean_stats(sw, std::string("wave ") + name);
        }
    }
}

TEST(images_are_not_trivially_black_and_emitter_scenes_have_energy) {
    // Guard against both arms agreeing on an all-zero image (identical-for-the-wrong-reason).
    for (const char* name : {"cornell", "divergence", "furnace"}) {
        Scene s = make_scene(name, 64, 36);
        for (GpuMode mode : {GpuMode::Megakernel, GpuMode::Wavefront}) {
            RenderStats st;
            auto r = gpu_render(s, mode, params(4), &st);
            double sum = 0;
            for (float v : r.rgb) sum += v;
            CHECK(sum > 0.0);
            bool nonneg = true;
            for (float v : r.rgb) nonneg = nonneg && v >= 0.0f;
            CHECK(nonneg);   // radiance of non-negative emitters and albedos is non-negative
        }
    }
}

TEST(gpu_arms_agree_with_cpu_reference_statistically) {
    // DESIGN s8: >= 99.5% of pixels within 3 combined SE, image-mean relative difference <= 0.5%.
    for (const char* name : {"cornell", "divergence"}) {
        Scene s = make_scene(name, 64, 36);
        RenderStats sc;
        RenderResult cpu = render_cpu(s, params(64), &sc);
        for (GpuMode mode : {GpuMode::Megakernel, GpuMode::Wavefront}) {
            RenderStats sg;
            auto g = gpu_render(s, mode, params(64), &sg);
            ImageDiff d = compare_results(cpu, g);
            if (d.frac_within_3se < 0.995 || d.mean_rel > 0.005)
                std::printf("  [%s %s] frac3se=%.4f mean_rel=%.5f max_abs_rel=%.3g\n", name,
                            mode == GpuMode::Wavefront ? "wave" : "mega", d.frac_within_3se, d.mean_rel, d.max_abs_rel);
            CHECK(d.frac_within_3se >= 0.995);
            CHECK(d.mean_rel <= 0.005);
            CHECK(d.nonfinite == 0);
            check_clean_stats(sg, std::string("cpu-compare ") + name);
        }
    }
}

TEST(furnace_image_mean_matches_le_over_one_minus_rho_within_3_se) {
    // Oracle: closed Lambertian sphere, emission Le, albedo rho -> radiance Le/(1-rho) (DESIGN s8).
    // max_depth 200 makes the truncation bias rho^200 negligible (rho < 1 required, checked).
    Scene s = make_scene("furnace", 64, 36);
    float le = 0, rho = 0;
    bool found = false;
    for (const Material& m : s.materials)
        if (m.type == kLambert && (m.emission.x > 0 || m.emission.y > 0 || m.emission.z > 0)) {
            le = m.emission.x; rho = m.albedo.x; found = true; break;
        }
    CHECK(found);
    if (!found) return;
    CHECK(rho > 0.0f && rho < 1.0f);
    const double expect = (double)le / (1.0 - (double)rho);
    for (GpuMode mode : {GpuMode::Megakernel, GpuMode::Wavefront}) {
        RenderStats st;
        const int spp = 64;
        auto r = gpu_render(s, mode, params(spp, kSeed, 200), &st);
        const int n = r.width * r.height;
        double mean = 0, var_sum = 0;
        for (int i = 0; i < n; ++i) {
            mean += 0.2126 * r.rgb[3 * i] + 0.7152 * r.rgb[3 * i + 1] + 0.0722 * r.rgb[3 * i + 2];
            var_sum += r.lum_var[i];
        }
        mean /= n;
        // SE of the image mean over independent pixels and samples: sqrt(sum var_i / spp) / n.
        double se = std::sqrt(var_sum / spp) / n;
        // Furnace is grey, so luminance == channel value (weights sum to 1).
        if (std::fabs(mean - expect) > 3 * se)
            std::printf("  furnace mean=%.6f expect=%.6f se=%.3g\n", mean, expect, se);
        CHECK(std::fabs(mean - expect) <= 3 * se + 1e-6);
        // Also a coarse sanity bound so a huge se cannot hide a gross bias (wrong 1/q, missing rho).
        CHECK(std::fabs(mean - expect) <= 0.02 * expect);
        check_clean_stats(st, "furnace");
    }
}

TEST(wavefront_first_pass_queue_high_water_marks) {
    // Spp 1: every pixel path is active at depth 0 -> active-queue high water == w*h; no material
    // queue can hold more paths than exist; megakernel has no queues.
    for (const char* name : {"cornell", "divergence"}) {
        Scene s = make_scene(name, 64, 36);
        const long long npix = 64 * 36;
        RenderStats sw, sm;
        gpu_render(s, GpuMode::Wavefront, params(1), &sw);
        gpu_render(s, GpuMode::Megakernel, params(1), &sm);
        CHECK(sw.queue_high_water[4] == npix);
        for (int q = 0; q < 5; ++q) { CHECK(sw.queue_high_water[q] >= 0); CHECK(sw.queue_high_water[q] <= npix); }
        long long mats = sw.queue_high_water[0] + sw.queue_high_water[1] + sw.queue_high_water[2];
        CHECK(mats > 0);   // at least one path reaches a scattering material
        for (int q = 0; q < 5; ++q) CHECK(sm.queue_high_water[q] == 0);
        check_clean_stats(sw, std::string("hw wave ") + name);
        check_clean_stats(sm, std::string("hw mega ") + name);
    }
}

TEST(divergence_scene_exercises_more_than_one_material_queue) {
    // The scene has Lambertian, metal and dielectric materials interleaved in view (DESIGN s3):
    // at depth 0 at least two of the three material queues must receive paths.
    Scene s = make_scene("divergence", 64, 36);
    RenderStats st;
    gpu_render(s, GpuMode::Wavefront, params(1), &st);
    int nonempty = 0;
    for (int q = 0; q < 3; ++q) nonempty += st.queue_high_water[q] > 0;
    CHECK(nonempty >= 2);
}

TEST(wavefront_peak_vram_below_budget_at_full_resolution) {
    // DESIGN s1: <= 3.5 GB at 1280x720. Path state is ~140 MB, so also require it to be sane (> 1 MB).
    Scene s = make_scene("divergence", 1280, 720);
    GpuRenderer r(s, GpuMode::Wavefront);
    RenderStats st;
    auto res = r.render(params(1), &st);
    check_shape(res, 1280, 720, 1);
    CHECK(st.peak_vram_bytes > 0);
    CHECK(st.peak_vram_bytes < 3500000000ull);
    CHECK(st.peak_vram_bytes > (1u << 20));
    CHECK(r.device_bytes() > 0);
    CHECK(r.device_bytes() < 3500000000ull);
    CHECK(st.queue_high_water[4] == 1280LL * 720);
    check_clean_stats(st, "1280x720");
}

TEST(ray_counters_are_zero_unless_requested) {
    Scene s = make_scene("cornell", 64, 36);
    const uint64_t npix = 64 * 36;
    for (GpuMode mode : {GpuMode::Megakernel, GpuMode::Wavefront}) {
        RenderStats off, on;
        auto a = gpu_render(s, mode, params(2, kSeed, 8, false), &off);
        auto b = gpu_render(s, mode, params(2, kSeed, 8, true), &on);
        CHECK(off.extension_rays == 0u);
        CHECK(off.shadow_rays == 0u);
        // Every pixel casts at least its camera ray in each of 2 passes; cornell has Lambertian walls
        // lit by an area light, so shadow rays exist.
        CHECK(on.extension_rays >= npix * 2);
        CHECK(on.shadow_rays > 0u);
        // Counting must not change the image (instrumentation is observation only).
        expect_identical(a, b, "count_rays on/off");
    }
}

TEST(ray_counts_agree_between_arms) {
    // Same paths -> same number of extension and shadow rays (metamorphic across arms).
    Scene s = make_scene("cornell", 64, 36);
    RenderStats sm, sw;
    gpu_render(s, GpuMode::Megakernel, params(2, kSeed, 8, true), &sm);
    gpu_render(s, GpuMode::Wavefront, params(2, kSeed, 8, true), &sw);
    CHECK(sm.extension_rays == sw.extension_rays);
    CHECK(sm.shadow_rays == sw.shadow_rays);
}

TEST(same_renderer_same_params_renders_identically_and_seed_matters) {
    Scene s = make_scene("cornell", 64, 36);
    for (GpuMode mode : {GpuMode::Megakernel, GpuMode::Wavefront}) {
        GpuRenderer r(s, mode);
        RenderStats st;
        auto a = r.render(params(4, 7), &st);
        auto b = r.render(params(4, 7), &st);    // render() must reset accumulators itself
        expect_identical(a, b, "repeat");
        auto c = r.render(params(4, 8), &st);
        size_t diff = count_diff(a.rgb, c.rgb);
        CHECK(diff > a.rgb.size() / 10);          // Monte Carlo noise differs for nearly every pixel
        auto d = r.render(params(4, 7), &st);     // returning to seed 7 restores the first image
        expect_identical(a, d, "after other seed");
    }
}

TEST(progressive_api_equals_one_shot_render) {
    // Metamorphic: reset, passes [0,2) then [2,4), download(4) == render(spp 4); no atomics so exact.
    Scene s = make_scene("divergence", 64, 36);
    for (GpuMode mode : {GpuMode::Megakernel, GpuMode::Wavefront}) {
        GpuRenderer r(s, mode);
        RenderStats st;
        RenderParams p = params(4);
        auto one = r.render(p, &st);
        r.reset();
        r.run_passes(0, 2, p, &st);
        auto half = r.download(2);
        r.run_passes(2, 2, p, &st);
        auto prog = r.download(4);
        expect_identical(one, prog, "progressive 0+2+2");
        CHECK(half.spp == 2);
        // The 2-sample partial equals a one-shot spp 2 render (same first two samples).
        auto two = r.render(params(2), &st);
        expect_identical(two, half, "partial vs spp2");
        // download_mean returns the same mean radiance as download.
        r.reset();
        r.run_passes(0, 4, p, &st);
        std::vector<float> mean;
        r.download_mean(4, &mean);
        CHECK(bitwise_equal(mean, one.rgb));
    }
}

TEST(reset_discards_previous_accumulation) {
    // After reset, a 1-sample pass must give the 1-spp image, not 1 + earlier samples.
    Scene s = make_scene("cornell", 64, 36);
    GpuRenderer r(s, GpuMode::Wavefront);
    RenderStats st;
    RenderParams p = params(1);
    auto ref = r.render(p, &st);
    r.run_passes(0, 3, p, &st);    // dirty the accumulators
    r.reset();
    r.run_passes(0, 1, p, &st);
    expect_identical(ref, r.download(1), "reset");
}

TEST(spp_one_and_odd_resolutions_render_validly_and_arms_agree) {
    struct Dim { int w, h; };
    for (Dim d : {Dim{1, 1}, Dim{33, 17}, Dim{31, 1}, Dim{1, 33}}) {
        Scene s = make_scene("cornell", d.w, d.h);
        for (int spp : {1, 3}) {
            RenderStats sm, sw;
            auto m = gpu_render(s, GpuMode::Megakernel, params(spp), &sm);
            auto w = gpu_render(s, GpuMode::Wavefront, params(spp), &sw);
            check_shape(m, d.w, d.h, spp);
            check_shape(w, d.w, d.h, spp);
            expect_identical(m, w, std::to_string(d.w) + "x" + std::to_string(d.h));
            bool var_ok = true;
            for (float v : w.lum_var) var_ok = var_ok && std::isfinite(v) && v >= 0.0f;
            CHECK(var_ok);   // variance is a finite non-negative number even at spp 1
            CHECK(sw.queue_high_water[4] == (long long)d.w * d.h || spp > 1);
            check_clean_stats(sm, "tiny mega");
            check_clean_stats(sw, "tiny wave");
        }
    }
}

TEST(multi_sample_render_reports_positive_variance) {
    // A noisy diffuse scene with spp 16 must show positive variance in some pixel; a constant-zero
    // lum_var would pass shape checks yet make the 3 SE criterion vacuous.
    Scene s = make_scene("cornell", 64, 36);
    RenderStats st;
    auto r = gpu_render(s, GpuMode::Wavefront, params(16), &st);
    double sum = 0;
    for (float v : r.lum_var) sum += v;
    CHECK(sum > 0.0);
}

TEST(gpu_renderer_on_unfinalised_scene_throws_runtime_error) {
    // A scene with primitives but no BVH (finalize skipped) must be rejected at construction.
    Scene bad = make_scene("cornell", 16, 16);
    bad.nodes.clear();
    for (GpuMode mode : {GpuMode::Megakernel, GpuMode::Wavefront}) {
        bool thrown = false, right_type = false;
        try { GpuRenderer r(bad, mode); }
        catch (const std::runtime_error&) { thrown = right_type = true; }
        catch (...) { thrown = true; }
        CHECK(thrown);
        CHECK(right_type);
    }
    Scene raw;   // never finalised: nodes empty, nothing derived
    raw.width = 8; raw.height = 8;
    Material m{};
    m.type = kLambert; m.albedo = Vec3{0.5f, 0.5f, 0.5f};
    raw.materials.push_back(m);
    Sphere sp{};
    sp.center = Vec3{0, 0, -3}; sp.radius = 1.0f; sp.material = 0;
    raw.spheres.push_back(sp);
    bool thrown = false;
    try { GpuRenderer r(raw, GpuMode::Wavefront); } catch (const std::runtime_error&) { thrown = true; } catch (...) {}
    CHECK(thrown);
}

TEST(query_cuda_device_reports_gtx_1650_ti_facts) {
    CudaDeviceFacts f = query_cuda_device();
    CHECK(f.compute_capability == "7.5");
    CHECK(!f.name.empty());
    CHECK(f.total_memory_bytes > 3000000000ull);
    CHECK(f.free_memory_bytes > 0u);
    CHECK(f.free_memory_bytes <= f.total_memory_bytes);
}
