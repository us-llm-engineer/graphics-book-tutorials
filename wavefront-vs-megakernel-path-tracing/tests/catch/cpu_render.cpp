// estimator correctness of render_cpu at image level.
// Oracles: closed-form furnace radiance Le/(1-rho); an independent naive BSDF-sampling-only path
// tracer written in this file (own RNG, own sphere intersection, own cosine sampling, no NEE, no
// MIS, no Russian roulette); metamorphic relations (seed, thread count, max_depth); the single
// path estimator wf::trace_path for the variance buffer. Statistical bound: 4.5 standard errors.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "wf/film.hpp"
#include "wf/path_core.hpp"
#include "wf/renderer.hpp"
#include "wf/scene.hpp"

using namespace wf;

namespace {

RenderResult run(const Scene& sc, int spp, uint32_t seed, int depth, int threads,
                 RenderStats* st = nullptr) {
    RenderParams p;
    p.spp = spp;
    p.seed = seed;
    p.max_depth = depth;
    p.threads = threads;
    RenderStats local;
    return render_cpu(sc, p, st ? st : &local);
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
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
// Pixels are independent: Var(image mean) = sum_p (lum_var_p / spp) / N^2.
double mean_se(const RenderResult& r) {
    double s = 0;
    int n = r.width * r.height;
    for (int i = 0; i < n; ++i) s += (double)r.lum_var[i] / r.spp;
    return std::sqrt(s) / n;
}

// ---------------------------------------------------------------------------------------------
// Tiny closed scenes of spheres, described once and turned into both a wf::Scene and the naive
// reference's own representation, so the two cannot disagree about the geometry.
// ---------------------------------------------------------------------------------------------
enum RefKind { kRefLambert, kRefMetal, kRefEmitter };
struct RefSphere {
    double c[3];
    double r;
    RefKind kind;
    double albedo[3];
    double Le[3];
};

struct Room {
    std::vector<RefSphere> spheres;
    double cam_pos[3], cam_look[3], cam_up[3];
    double vfov_deg;
};

Scene to_scene(const Room& room, int w, int h) {
    Scene s;
    s.name = "room";
    s.width = w;
    s.height = h;
    for (size_t i = 0; i < room.spheres.size(); ++i) {
        const RefSphere& rs = room.spheres[i];
        Material m{};
        m.type = rs.kind == kRefLambert ? (int)kLambert
                 : rs.kind == kRefMetal ? (int)kMetal
                                        : (int)kEmissive;
        m.ior = 1.5f;
        m.albedo = Vec3{(float)rs.albedo[0], (float)rs.albedo[1], (float)rs.albedo[2]};
        m.emission = Vec3{(float)rs.Le[0], (float)rs.Le[1], (float)rs.Le[2]};
        s.materials.push_back(m);
        Sphere sp{};
        sp.center = Vec3{(float)rs.c[0], (float)rs.c[1], (float)rs.c[2]};
        sp.radius = (float)rs.r;
        sp.material = (int)i;
        s.spheres.push_back(sp);
    }
    s.camera = Camera{Vec3{(float)room.cam_pos[0], (float)room.cam_pos[1], (float)room.cam_pos[2]},
                      Vec3{(float)room.cam_look[0], (float)room.cam_look[1], (float)room.cam_look[2]},
                      Vec3{(float)room.cam_up[0], (float)room.cam_up[1], (float)room.cam_up[2]},
                      (float)room.vfov_deg};
    finalize_scene(s);
    return s;
}

// ---- naive reference: doubles, own RNG, BSDF sampling only -----------------------------------
struct D3 {
    double x, y, z;
};
D3 operator+(D3 a, D3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
D3 operator-(D3 a, D3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
D3 operator*(D3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
double dotd(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
D3 crossd(D3 a, D3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
D3 unit(D3 a) {
    double l = std::sqrt(dotd(a, a));
    return a * (1.0 / l);
}
D3 v3(const double* a) { return {a[0], a[1], a[2]}; }

struct SplitMix {  // own RNG (splitmix64), unrelated to wf::rng
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double u() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
};

// Nearest intersection with t > 0 (sphere roots by the plain quadratic in double).
bool ref_hit(const Room& room, D3 o, D3 d, int* idx, double* t_out) {
    double best = 1e300;
    int bi = -1;
    for (size_t i = 0; i < room.spheres.size(); ++i) {
        const RefSphere& s = room.spheres[i];
        D3 oc = o - v3(s.c);
        double b = dotd(oc, d);
        double c = dotd(oc, oc) - s.r * s.r;
        double disc = b * b - c;
        if (disc < 0) continue;
        double sq = std::sqrt(disc);
        double t = -b - sq;
        if (t <= 0) t = -b + sq;
        if (t > 0 && t < best) {
            best = t;
            bi = (int)i;
        }
    }
    *idx = bi;
    *t_out = best;
    return bi >= 0;
}

D3 ref_trace(const Room& room, D3 o, D3 d, int max_depth, SplitMix& rng) {
    D3 L{0, 0, 0}, T{1, 1, 1};
    for (int depth = 0;; ++depth) {
        int i;
        double t;
        if (!ref_hit(room, o, d, &i, &t)) break;  // env radiance is zero in every room
        const RefSphere& s = room.spheres[i];
        L = L + D3{T.x * s.Le[0], T.y * s.Le[1], T.z * s.Le[2]};
        if (s.kind == kRefEmitter || depth >= max_depth) break;
        D3 p = o + d * t;
        D3 n = (p - v3(s.c)) * (1.0 / s.r);
        if (dotd(n, d) > 0) n = n * -1.0;  // facing the ray
        T = D3{T.x * s.albedo[0], T.y * s.albedo[1], T.z * s.albedo[2]};
        D3 nd;
        if (s.kind == kRefMetal) {
            nd = unit(d - n * (2.0 * dotd(d, n)));
        } else {
            // Malley's method: uniform disk, lift to the cosine-weighted hemisphere about n.
            double u1 = rng.u(), u2 = rng.u();
            double r = std::sqrt(u1), phi = 2.0 * 3.14159265358979323846 * u2;
            D3 a = std::fabs(n.x) > 0.9 ? D3{0, 1, 0} : D3{1, 0, 0};
            D3 tt = unit(crossd(a, n));
            D3 bb = crossd(n, tt);
            nd = unit(tt * (r * std::cos(phi)) + bb * (r * std::sin(phi)) +
                      n * std::sqrt(std::max(0.0, 1.0 - u1)));
        }
        o = p + n * 1e-7;
        d = nd;
    }
    return L;
}

struct RefImage {
    double mean_lum;
    double se;
};

RefImage ref_render(const Room& room, int w, int h, int spp, int max_depth, uint64_t seed) {
    D3 pos = v3(room.cam_pos), look = v3(room.cam_look), up = v3(room.cam_up);
    D3 wv = unit(pos - look);
    D3 uv = unit(crossd(up, wv));
    D3 vv = crossd(wv, uv);
    double hh = std::tan(room.vfov_deg * 0.5 * 3.14159265358979323846 / 180.0);
    D3 horiz = uv * (2.0 * hh * (double)w / (double)h);
    D3 vert = vv * (2.0 * hh);
    D3 ll = pos - horiz * 0.5 - vert * 0.5 - wv;
    double sum_means = 0, sum_var_over_spp = 0;
    for (int py = 0; py < h; ++py) {
        for (int px = 0; px < w; ++px) {
            SplitMix rng{seed * 0x100000001b3ULL + (uint64_t)(py * w + px) * 0x9e3779b97f4a7c15ULL + 12345};
            double s1 = 0, s2 = 0;
            for (int k = 0; k < spp; ++k) {
                double u = ((double)px + rng.u()) / w;
                double v = ((double)(h - 1 - py) + rng.u()) / h;
                D3 dir = unit(ll + horiz * u + vert * v - pos);
                D3 L = ref_trace(room, pos, dir, max_depth, rng);
                double lum = 0.2126 * L.x + 0.7152 * L.y + 0.0722 * L.z;
                s1 += lum;
                s2 += lum * lum;
            }
            double m = s1 / spp;
            double var = std::max(0.0, (s2 - spp * m * m) / (spp - 1));
            sum_means += m;
            sum_var_over_spp += var / spp;
        }
    }
    int n = w * h;
    return {sum_means / n, std::sqrt(sum_var_over_spp) / n};
}

// Camera at (0,0,4.2) inside a closed enclosure (radius 5) holding a light, a diffuse ball and a
// tinted mirror ball.
Room room_general() {
    Room r;
    r.spheres = {
        {{0, 0, 0}, 5.0, kRefLambert, {0.6, 0.5, 0.4}, {0, 0, 0}},
        {{-1.5, 1.5, 0}, 1.0, kRefEmitter, {0, 0, 0}, {4.0, 4.0, 3.5}},
        {{1.2, -1.5, 0}, 1.2, kRefLambert, {0.7, 0.7, 0.7}, {0, 0, 0}},
        {{0.8, 1.8, -1.0}, 0.8, kRefMetal, {0.9, 0.8, 0.6}, {0, 0, 0}},
    };
    r.cam_pos[0] = 0; r.cam_pos[1] = 0; r.cam_pos[2] = 4.2;
    r.cam_look[0] = r.cam_look[1] = r.cam_look[2] = 0;
    r.cam_up[0] = 0; r.cam_up[1] = 1; r.cam_up[2] = 0;
    r.vfov_deg = 70.0;
    return r;
}

// One small, very bright light (radius 0.3, Le 60) plus an occluding diffuse ball: here the
// light is found mostly by next-event estimation, so the MIS weights decide the answer.
Room room_small_light() {
    Room r;
    r.spheres = {
        {{0, 0, 0}, 5.0, kRefLambert, {0.5, 0.5, 0.5}, {0, 0, 0}},
        {{0, 2.0, -1.0}, 0.3, kRefEmitter, {0, 0, 0}, {60.0, 60.0, 60.0}},
        {{0, -1.0, -1.0}, 1.5, kRefLambert, {0.7, 0.7, 0.7}, {0, 0, 0}},
    };
    r.cam_pos[0] = 0; r.cam_pos[1] = 0; r.cam_pos[2] = 4.2;
    r.cam_look[0] = r.cam_look[1] = r.cam_look[2] = 0;
    r.cam_up[0] = 0; r.cam_up[1] = 1; r.cam_up[2] = 0;
    r.vfov_deg = 70.0;
    return r;
}

struct Furnace {
    Scene scene;
    double expect;  // luminance of Le/(1-rho) per channel
    double rho_max;
    bool ok;
};

Furnace make_furnace(int w, int h) {
    Furnace f;
    f.scene = make_scene("furnace", w, h);
    Vec3 Le{0, 0, 0}, rho{0, 0, 0};
    int found = 0;
    for (const Sphere& s : f.scene.spheres) {
        const Material& m = f.scene.materials[s.material];
        if (m.emission.x > 0 || m.emission.y > 0 || m.emission.z > 0) {
            Le = m.emission;
            rho = m.albedo;
            ++found;
        }
    }
    f.ok = found >= 1 && rho.x < 1 && rho.y < 1 && rho.z < 1;
    f.rho_max = std::max(rho.x, std::max(rho.y, rho.z));
    f.expect = f.ok ? luminance(Vec3{Le.x / (1 - rho.x), Le.y / (1 - rho.y), Le.z / (1 - rho.z)}) : 0.0;
    return f;
}

}  // namespace

TEST_CASE("furnace sphere image mean equals Le/(1-rho) within 4.5 SE", "[cpu_render]") {
    // Given a closed Lambertian emissive shell (Le, rho read from the scene), camera inside, no env
    // When rendered at max_depth 200  Then the image mean luminance is lum(Le/(1-rho)) (geometric
    // series, Pharr-Humphreys 16.2.2) within 4.5 standard errors, and the SE itself is small
    // (so a huge lum_var cannot make the test pass vacuously).
    Furnace f = make_furnace(32, 18);
    REQUIRE(f.ok);
    REQUIRE(f.expect > 0.0);
    REQUIRE(f.scene.env.x == 0.0f);
    RenderResult r = run(f.scene, 64, 1234, 200, 0);
    REQUIRE(r.rgb.size() == (size_t)(3 * 32 * 18));
    double mean = mean_lum(r), se = mean_se(r);
    double tol = 4.5 * se + f.expect * std::pow(f.rho_max, 190.0) + 1e-4 * f.expect;
    INFO("mean " << mean << " expected " << f.expect << " se " << se);
    CHECK(std::fabs(mean - f.expect) <= tol);
    CHECK(se > 0.0);
    CHECK(se < 0.05 * f.expect);
}

TEST_CASE("render_cpu matches an independent naive BSDF-only path tracer on a closed room",
          "[cpu_render]") {
    // Given a closed room with a large light, a diffuse ball and a tinted mirror ball
    // When rendered by render_cpu (NEE + MIS + roulette) and by the naive reference in this file
    // (BSDF sampling only, no roulette, same max_depth 24, own RNG/intersection/cosine sampling)
    // Then image mean luminances agree within 4.5 combined standard errors.
    const int w = 48, h = 27, depth = 24;
    Room room = room_general();
    Scene sc = to_scene(room, w, h);
    RenderResult r = run(sc, 384, 7, depth, 0);
    RefImage ref = ref_render(room, w, h, 384, depth, 99);
    double se = std::sqrt(mean_se(r) * mean_se(r) + ref.se * ref.se);
    double mean = mean_lum(r);
    INFO("render_cpu " << mean << " reference " << ref.mean_lum << " combined se " << se);
    REQUIRE(ref.mean_lum > 0.0);
    CHECK(se < 0.05 * ref.mean_lum);
    CHECK(std::fabs(mean - ref.mean_lum) <= 4.5 * se + 2e-3 * ref.mean_lum);
}

TEST_CASE("small bright light: MIS render equals the naive reference mean", "[cpu_render]") {
    // Given a closed room whose only light is small and bright (radius 0.3, Le 60) behind which an
    // occluder sits, When rendered with NEE+MIS (render_cpu) and by the naive BSDF-only reference
    // (rarely hits the light but is unbiased by construction), Then the means agree within 4.5
    // combined SE. A wrong MIS weight (e.g. weights not summing to 1) or a missing 1/pdf biases it.
    const int w = 48, h = 27, depth = 24;
    Room room = room_small_light();
    Scene sc = to_scene(room, w, h);
    RenderResult r = run(sc, 512, 21, depth, 0);
    RefImage ref = ref_render(room, w, h, 512, depth, 5);
    double se = std::sqrt(mean_se(r) * mean_se(r) + ref.se * ref.se);
    double mean = mean_lum(r);
    INFO("render_cpu " << mean << " reference " << ref.mean_lum << " combined se " << se);
    REQUIRE(ref.mean_lum > 0.0);
    CHECK(se < 0.10 * ref.mean_lum);
    CHECK(std::fabs(mean - ref.mean_lum) <= 4.5 * se + 2e-3 * ref.mean_lum);
}

TEST_CASE("russian roulette leaves the furnace mean unchanged between max_depth 64 and 200",
          "[cpu_render]") {
    // Given the furnace, When rendered with independent seeds at max_depth 64 and 200 (roulette is
    // active from depth 3 in both; truncation beyond depth 64 is rho^64, negligible)
    // Then both means equal each other within 4.5 combined SE AND each equals the analytic value
    // within 4.5 own SE (a roulette weight not 1/q would move both off the analytic value).
    Furnace f = make_furnace(24, 14);
    REQUIRE(f.ok);
    RenderResult a = run(f.scene, 96, 11, 64, 0);
    RenderResult b = run(f.scene, 96, 12, 200, 0);
    double ma = mean_lum(a), mb = mean_lum(b), sa = mean_se(a), sb = mean_se(b);
    INFO("depth64 " << ma << " depth200 " << mb << " analytic " << f.expect);
    CHECK(std::fabs(ma - mb) <= 4.5 * std::sqrt(sa * sa + sb * sb));
    double trunc = f.expect * std::pow(f.rho_max, 60.0) + 1e-4 * f.expect;
    CHECK(std::fabs(ma - f.expect) <= 4.5 * sa + trunc);
    CHECK(std::fabs(mb - f.expect) <= 4.5 * sb + trunc);
}

TEST_CASE("two renders with identical parameters are bitwise identical", "[cpu_render]") {
    // Given the counter-based RNG, When the same scene/params are rendered twice (default thread
    // count), Then rgb and lum_var are bit-for-bit equal; the image is non-trivial (not vacuous).
    Scene sc = make_scene("cornell", 32, 18);
    RenderResult a = run(sc, 8, 99, 8, 0);
    RenderResult b = run(sc, 8, 99, 8, 0);
    REQUIRE(a.rgb.size() == (size_t)(3 * 32 * 18));
    CHECK(bitwise_equal(a.rgb, b.rgb));
    CHECK(bitwise_equal(a.lum_var, b.lum_var));
    CHECK(mean_lum(a) > 0.0);
}

TEST_CASE("thread count does not change the image: 1 vs 8 (and 3) threads bitwise equal",
          "[cpu_render]") {
    // Given a 31x17 image (rows not divisible by 8 or 3, so chunking edges are exercised)
    // When rendered with threads 1, 8 and 3  Then rgb and lum_var are bitwise equal.
    Scene sc = make_scene("cornell", 31, 17);
    RenderResult t1 = run(sc, 8, 5, 8, 1);
    RenderResult t8 = run(sc, 8, 5, 8, 8);
    RenderResult t3 = run(sc, 8, 5, 8, 3);
    REQUIRE(t1.rgb.size() == (size_t)(3 * 31 * 17));
    CHECK(bitwise_equal(t1.rgb, t8.rgb));
    CHECK(bitwise_equal(t1.lum_var, t8.lum_var));
    CHECK(bitwise_equal(t1.rgb, t3.rgb));
    CHECK(bitwise_equal(t1.lum_var, t3.lum_var));
    CHECK(mean_lum(t1) > 0.0);
}

TEST_CASE("changing the seed changes the image but not its expectation", "[cpu_render]") {
    // Given cornell 32x18 spp 32, When rendered with seeds 1 and 2
    // Then most pixels differ (independent noise) and the image means agree within 4.5 combined SE.
    Scene sc = make_scene("cornell", 32, 18);
    RenderResult a = run(sc, 32, 1, 8, 0);
    RenderResult b = run(sc, 32, 2, 8, 0);
    int n = 32 * 18, differing = 0;
    for (int i = 0; i < n; ++i)
        if (a.rgb[3 * i] != b.rgb[3 * i] || a.rgb[3 * i + 1] != b.rgb[3 * i + 1] ||
            a.rgb[3 * i + 2] != b.rgb[3 * i + 2])
            ++differing;
    double sa = mean_se(a), sb = mean_se(b);
    INFO("differing pixels " << differing << "/" << n << " means " << mean_lum(a) << " " << mean_lum(b));
    CHECK(differing >= n / 2);
    REQUIRE(sa > 0.0);
    CHECK(std::fabs(mean_lum(a) - mean_lum(b)) <= 4.5 * std::sqrt(sa * sa + sb * sb));
}

TEST_CASE("variance buffer is sized, finite, non-negative and matches trace_path samples; no non-finite pixels",
          "[cpu_render]") {
    // (a) cornell 64x36 spp 16: rgb 3*w*h, lum_var w*h, all finite, lum_var >= 0, zero non-finite
    //     pixels, zero invalid hits / overflows in stats (claim: non-finite pixel falsifier).
    // (b) spp 1 is accepted: finite, lum_var >= 0 (undefined sample variance must not become NaN).
    // (c) independent oracle: with spp 4, rgb = mean and lum_var = unbiased variance of the
    //     per-sample luminance of wf::trace_path(sample 0..3) recomputed in double here.
    Scene big = make_scene("cornell", 64, 36);
    RenderStats st;
    RenderResult r = run(big, 16, 1234, 8, 0, &st);
    REQUIRE(r.width == 64);
    REQUIRE(r.height == 36);
    REQUIRE(r.spp == 16);
    REQUIRE(r.rgb.size() == (size_t)(3 * 64 * 36));
    REQUIRE(r.lum_var.size() == (size_t)(64 * 36));
    CHECK(count_nonfinite(r.rgb) == 0);
    bool var_ok = true;
    for (float v : r.lum_var) var_ok = var_ok && std::isfinite(v) && v >= 0.0f;
    CHECK(var_ok);
    CHECK(st.invalid_hits == 0);
    CHECK(st.queue_overflows == 0);

    Scene small = make_scene("cornell", 8, 6);
    RenderResult one = run(small, 1, 3, 8, 0);
    REQUIRE(one.spp == 1);
    REQUIRE(one.rgb.size() == (size_t)(3 * 8 * 6));
    REQUIRE(one.lum_var.size() == (size_t)(8 * 6));
    CHECK(count_nonfinite(one.rgb) == 0);
    bool one_ok = true;
    for (float v : one.lum_var) one_ok = one_ok && std::isfinite(v) && v >= 0.0f;
    CHECK(one_ok);

    RenderResult four = run(small, 4, 3, 8, 1);
    SceneView view = small.view();
    int bad = 0;
    for (int py = 0; py < 6; ++py)
        for (int px = 0; px < 8; ++px) {
            double mean[3] = {0, 0, 0}, lum[4];
            for (int s = 0; s < 4; ++s) {
                Vec3 L = trace_path(view, px, py, (uint32_t)s, 3, 8);
                mean[0] += L.x / 4; mean[1] += L.y / 4; mean[2] += L.z / 4;
                lum[s] = luminance(L);
            }
            double lm = (lum[0] + lum[1] + lum[2] + lum[3]) / 4, var = 0;
            for (int s = 0; s < 4; ++s) var += (lum[s] - lm) * (lum[s] - lm) / 3.0;
            int i = py * 8 + px;
            for (int c = 0; c < 3; ++c)
                if (std::fabs(four.rgb[3 * i + c] - mean[c]) > 1e-4 * std::fabs(mean[c]) + 1e-6) ++bad;
            if (std::fabs(four.lum_var[i] - var) > 1e-3 * var + 1e-5) ++bad;
        }
    INFO("mismatching channels/variances versus trace_path: " << bad);
    CHECK(bad == 0);
}
