// wavefront invariants, failure mapping and memory.
// Oracles: the megakernel arm (bitwise identity, DESIGN s2/s8), cudaMemGetInfo deltas, the design
// memory figure (about 140 MB path pool at 1280x720) and the 3.5e9 budget (DESIGN s1), the CPU
// reference under the DESIGN s8 criteria, and the metamorphic relation "N passes of one sample
// averaged == one N-sample render".
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "wf/film.hpp"
#include "wf/renderer.hpp"
#include "wf/scene.hpp"

using namespace wf;

namespace {

RenderParams make_params(int spp, uint32_t seed = 7, int depth = 8, bool count = false, int cap = 0) {
    RenderParams p;
    p.spp = spp;
    p.seed = seed;
    p.max_depth = depth;
    p.count_rays = count;
    p.threads = 0;
    p.queue_capacity = cap;
    return p;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
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

bool all_finite_valid(const RenderResult& r) {
    if (count_nonfinite(r.rgb) != 0) return false;
    for (float v : r.lum_var)
        if (!std::isfinite(v) || v < 0.0f) return false;
    return true;
}

// Hand-built scene: non-emissive Lambertian spheres only, constant environment 0.5, no lights.
Scene make_unlit_scene(int w, int h, bool do_finalize) {
    Scene s;
    s.name = "unlit";
    s.width = w;
    s.height = h;
    s.env = Vec3{0.5f, 0.5f, 0.5f};
    Material m{};
    m.type = kLambert;
    m.ior = 1.0f;
    m.albedo = Vec3{0.5f, 0.5f, 0.5f};
    m.emission = Vec3{0.0f, 0.0f, 0.0f};
    s.materials.push_back(m);
    Sphere a{};
    a.center = Vec3{0.0f, 0.0f, -3.0f};
    a.radius = 1.0f;
    a.material = 0;
    s.spheres.push_back(a);
    Sphere ground{};
    ground.center = Vec3{0.0f, -101.0f, -3.0f};
    ground.radius = 100.0f;
    ground.material = 0;
    s.spheres.push_back(ground);
    s.camera.position = Vec3{0.0f, 0.0f, 0.0f};
    s.camera.look_at = Vec3{0.0f, 0.0f, -3.0f};
    s.camera.up = Vec3{0.0f, 1.0f, 0.0f};
    s.camera.vfov_deg = 50.0f;
    if (do_finalize) finalize_scene(s);
    return s;
}

}  // namespace

TEST_CASE("a queue capacity far below the path count counts overflows and the renderer survives", "[wavefront_failure]") {
    // Given 2304 paths and queue_capacity 100, when the divergence scene renders, then overflows are
    // COUNTED (>0), the call returns normally, the image is finite, and the same renderer then renders
    // with capacity 0 bitwise equal to the megakernel (no corrupted memory or poisoned context).
    Scene s = make_scene("divergence", 64, 36);
    GpuRenderer wf(s, GpuMode::Wavefront);
    GpuRenderer mk(s, GpuMode::Megakernel);
    RenderStats small;
    RenderResult r;
    REQUIRE_NOTHROW(r = wf.render(make_params(2, 7, 8, false, 100), &small));
    INFO("overflows " << small.queue_overflows);
    CHECK(small.queue_overflows > 0);
    CHECK(r.width == 64);
    CHECK(r.height == 36);
    CHECK(all_finite_valid(r));
    CHECK(cudaGetLastError() == cudaSuccess);   // no sticky device error left behind
    RenderStats after, ref;
    RenderResult second = wf.render(make_params(2, 7, 8, false, 0), &after);
    RenderResult expect = mk.render(make_params(2, 7, 8, false, 0), &ref);
    REQUIRE(rgb_sum(expect) > 0.0);
    CHECK(after.queue_overflows == 0);
    CHECK(same_image(second, expect));
}

TEST_CASE("capacity 0 (auto) and exactly width*height never overflow and match the megakernel", "[wavefront_failure]") {
    // The design guarantee (Laine Sec4.4, DESIGN s6): capacity == path count cannot overflow.
    // A boundary of exactly W*H must be sufficient (off-by-one in the capacity check would drop the
    // last push), so the image must also be bitwise the megakernel image.
    Scene s = make_scene("divergence", 64, 36);
    GpuRenderer mk(s, GpuMode::Megakernel);
    RenderStats ref;
    RenderResult expect = mk.render(make_params(4, 7, 8), &ref);
    REQUIRE(rgb_sum(expect) > 0.0);
    GpuRenderer wf(s, GpuMode::Wavefront);
    for (int cap : {0, 64 * 36}) {
        INFO("queue_capacity " << cap);
        RenderStats st;
        RenderResult r = wf.render(make_params(4, 7, 8, false, cap), &st);
        CHECK(st.queue_overflows == 0);
        CHECK(st.invalid_hits == 0);
        CHECK(same_image(r, expect));
    }
}

TEST_CASE("an unfinalized scene is rejected with runtime_error and 20 renderers leak under 1 MB", "[wavefront_failure]") {
    // Part A: Scene with no BVH (finalize_scene never called). The constructor must throw
    // std::runtime_error (CUDA/upload error mapping), and the process must still build a good renderer.
    {
        Scene bad = make_unlit_scene(16, 9, false);
        REQUIRE(bad.nodes.empty());   // guard: the scene really is unfinalized
        auto construct = [&] { GpuRenderer r(bad, GpuMode::Wavefront); };
        CHECK_THROWS_AS(construct(), std::runtime_error);
        Scene ok = make_unlit_scene(16, 9, true);
        GpuRenderer r(ok, GpuMode::Wavefront);   // process unharmed
        RenderStats st;
        CHECK(rgb_sum(r.render(make_params(1), &st)) > 0.0);
    }
    // Part B: construct and destroy 20 wavefront renderers; cudaMemGetInfo free memory may not
    // shrink by 1 MB or more (a leaked path pool at 64x36 is >100 KB each, so 20 leaks exceed 1 MB;
    // the megakernel suite uses the same tolerance). One warm-up renderer first absorbs lazy
    // context/allocator growth.
    Scene s = make_scene("divergence", 64, 36);
    { GpuRenderer warm(s, GpuMode::Wavefront); RenderStats st; warm.render(make_params(1), &st); }
    size_t free_before = 0, free_after = 0, total = 0;
    REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);
    for (int i = 0; i < 20; ++i) {
        GpuRenderer r(s, GpuMode::Wavefront);
        RenderStats st;
        r.render(make_params(1), &st);
    }
    REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);
    const long long leaked = (long long)free_before - (long long)free_after;
    INFO("leaked bytes " << leaked);
    CHECK(leaked < 1000000);
}

TEST_CASE("wavefront memory at 1280x720 is positive, near the design pool and inside the budget", "[wavefront_failure]") {
    // Design: about 140 MB path pool at 1280x720 (DESIGN s6); budget 3.5e9 (DESIGN s1); the claim
    // of this test is "below 1.0e9". Lower bound 44 MB = 921600 paths * 48 bytes (origin, direction,
    // throughput, L as floats), the least a structure-of-arrays pool can occupy: guards a pool that
    // was never allocated at the real size.
    Scene s = make_scene("divergence", 1280, 720);
    GpuRenderer g(s, GpuMode::Wavefront);
    RenderStats st;
    RenderResult r = g.render(make_params(1), &st);
    REQUIRE(r.width == 1280);
    CHECK(all_finite_valid(r));
    INFO("peak_vram " << st.peak_vram_bytes << " device_bytes " << g.device_bytes());
    CHECK(st.peak_vram_bytes > 0u);
    CHECK(g.device_bytes() > 0u);
    CHECK(st.peak_vram_bytes < 1000000000ull);
    CHECK(g.device_bytes() < 1000000000ull);
    CHECK(st.peak_vram_bytes < 3500000000ull);
    CHECK(g.device_bytes() >= 44000000ull);
    CHECK(st.queue_overflows == 0);
    CHECK(st.invalid_hits == 0);
}

TEST_CASE("max_depth 1 matches the CPU reference and max_depth 64 finishes finite", "[wavefront_failure]") {
    // max_depth 1: only the camera vertex is shaded (emission + NEE at depth 0, then terminate).
    // Oracle: render_cpu with the same max_depth under DESIGN s8. A wavefront that runs a second
    // bounce when it should not adds energy beyond 0.5% and fails mean_rel. max_depth 64 exercises
    // the loop limit of the active list with long paths: finite, valid, bitwise the megakernel.
    Scene s = make_scene("divergence", 64, 36);
    RenderStats sc, sg;
    RenderResult cpu = render_cpu(s, make_params(64, 7, 1), &sc);
    GpuRenderer g(s, GpuMode::Wavefront);
    RenderResult gpu = g.render(make_params(64, 7, 1), &sg);
    REQUIRE(rgb_sum(cpu) > 0.0);
    REQUIRE(rgb_sum(gpu) > 0.0);
    ImageDiff d = compare_results(cpu, gpu);
    INFO("frac_within_3se=" << d.frac_within_3se << " mean_rel=" << d.mean_rel << " nonfinite=" << d.nonfinite);
    CHECK(d.frac_within_3se >= 0.995);
    CHECK(d.mean_rel <= 0.005);
    CHECK(d.nonfinite == 0);
    CHECK(sg.queue_overflows == 0);

    RenderStats deep, ref;
    RenderResult r64 = g.render(make_params(2, 7, 64), &deep);
    GpuRenderer mk(s, GpuMode::Megakernel);
    RenderResult m64 = mk.render(make_params(2, 7, 64), &ref);
    CHECK(all_finite_valid(r64));
    CHECK(deep.invalid_hits == 0);
    CHECK(deep.queue_overflows == 0);
    CHECK(same_image(r64, m64));
}

TEST_CASE("a scene with zero lights renders finite and bitwise equal to the megakernel", "[wavefront_failure]") {
    // Given non-emissive spheres under env 0.5 and an empty light list (hand-built), no NEE is
    // possible: a wavefront that indexes lights[0] or divides by n_lights==0 yields NaN or a crash.
    // Oracle: megakernel image; positive energy comes from escaped rays seeing env 0.5.
    Scene s = make_unlit_scene(32, 18, true);
    REQUIRE(s.lights.empty());   // guard: really zero lights
    GpuRenderer mk(s, GpuMode::Megakernel);
    GpuRenderer wf(s, GpuMode::Wavefront);
    RenderStats a, b;
    RenderResult m = mk.render(make_params(4, 7, 8, true), &a);
    RenderResult w = wf.render(make_params(4, 7, 8, true), &b);
    REQUIRE(rgb_sum(m) > 0.0);
    CHECK(all_finite_valid(w));
    CHECK(same_image(m, w));
    CHECK(a.shadow_rays == 0u);
    CHECK(b.shadow_rays == 0u);
    CHECK(b.queue_overflows == 0);
    CHECK(b.invalid_hits == 0);
}

TEST_CASE("sizes 1x1, 33x17 and 1x720 are bitwise equal to the megakernel", "[wavefront_failure]") {
    // Non-multiples of the warp/block size and degenerate widths: launch-grid and scan-block boundary
    // bugs (last partial block dropped, compaction losing the final path) change pixels vs the megakernel.
    struct Dim { int w, h; };
    for (Dim d : {Dim{1, 1}, Dim{33, 17}, Dim{1, 720}}) {
        INFO("size " << d.w << "x" << d.h);
        Scene s = make_scene("cornell", d.w, d.h);
        GpuRenderer mk(s, GpuMode::Megakernel);
        GpuRenderer wf(s, GpuMode::Wavefront);
        RenderStats a, b;
        RenderResult m = mk.render(make_params(4), &a);
        RenderResult w = wf.render(make_params(4), &b);
        REQUIRE(rgb_sum(m) > 0.0);
        CHECK(all_finite_valid(w));
        CHECK(same_image(m, w));
        CHECK(b.queue_overflows == 0);
        CHECK(b.queue_high_water[4] == (long long)d.w * d.h);
    }
}

TEST_CASE("an spp 8 render equals the average of eight single-pass progressive downloads", "[wavefront_failure]") {
    // Metamorphic relation: one render of 8 samples == mean of the 8 per-sample images (reset;
    // run_passes(i,1); download(1)). Tolerance 1e-6 relative (float addition order differs only by
    // rounding of 8 positive terms, <= 4e-7). A render that reuses state between passes, or a pass
    // index not threaded into the RNG, breaks it far beyond that.
    Scene s = make_scene("divergence", 64, 36);
    GpuRenderer g(s, GpuMode::Wavefront);
    RenderStats st;
    RenderResult one = g.render(make_params(8), &st);
    REQUIRE(rgb_sum(one) > 0.0);
    std::vector<double> acc(one.rgb.size(), 0.0);
    for (int i = 0; i < 8; ++i) {
        g.reset();
        g.run_passes(i, 1, make_params(1), &st);
        RenderResult pass = g.download(1);
        REQUIRE(pass.rgb.size() == acc.size());
        for (size_t k = 0; k < acc.size(); ++k) acc[k] += pass.rgb[k];
    }
    size_t bad = 0;
    double worst = 0;
    for (size_t k = 0; k < acc.size(); ++k) {
        const double avg = acc[k] / 8.0;
        const double ref = one.rgb[k];
        const double err = std::fabs(avg - ref) / std::max(std::fabs(ref), 1e-6);
        worst = std::max(worst, err);
        bad += err > 1e-6;
    }
    INFO("channels beyond 1e-6: " << bad << " worst relative error " << worst);
    CHECK(bad == 0u);
}
