// Tests for include/wf/math.hpp and include/wf/rng.hpp (header-only, solitary width).
// Every assertion states its oracle: hand-computed value, algebraic identity, or a statistical bound
// (3 standard errors) derived from the analytic moments of U[0,1).
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <vector>
#include <random>
#include <set>
#include <utility>

#include "test_framework.hpp"
#include "wf/math.hpp"
#include "wf/rng.hpp"

using namespace wf;

namespace {

// Independent test-local generator (not wf::rng) for random inputs.
struct Rnd {
    std::mt19937_64 g;
    explicit Rnd(uint64_t s) : g(s) {}
    double u() { return (double)(g() >> 11) * (1.0 / 9007199254740992.0); }
    double range(double a, double b) { return a + (b - a) * u(); }
    Vec3 unit() {  // uniform on the sphere
        double z = range(-1, 1), phi = range(0, 2 * 3.14159265358979323846);
        double r = std::sqrt(1 - z * z);
        return Vec3{(float)(r * std::cos(phi)), (float)(r * std::sin(phi)), (float)z};
    }
    Vec3 any(double lim) { return Vec3{(float)range(-lim, lim), (float)range(-lim, lim), (float)range(-lim, lim)}; }
};

bool near3(Vec3 a, Vec3 b, double tol) {
    return std::fabs(a.x - b.x) <= tol && std::fabs(a.y - b.y) <= tol && std::fabs(a.z - b.z) <= tol;
}

struct Pearson {
    double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    void add(double x, double y) { n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; }
    double r() const {
        double cov = sxy / n - (sx / n) * (sy / n);
        double vx = sxx / n - (sx / n) * (sx / n), vy = syy / n - (sy / n) * (sy / n);
        return cov / std::sqrt(vx * vy);
    }
};

// Mean/variance of a stream must sit within kSigma SE of U[0,1): mean 1/2, var 1/12.
// SE(mean) = sqrt(var/N); SE(sample var) = sqrt((E[(x-mu)^4]-var^2)/N) = sqrt((1/80-1/144)/N) = sqrt(1/(180N)).
// The suite runs ~8 moment checks (4 streams x mean/var) with fixed seeds; at 3 SE the family-wise false-fail
// chance is ~2% (and one was observed at 3.2 SE on a verified-unbiased generator). Bonferroni: 4.5 SE gives a
// per-check two-sided p of 7e-6, family-wise < 1e-4. A real bias of 0.003 in the mean is 10 SE at N=1e6, so
// the power is kept (see the negative control test).
constexpr double kSigma = 4.5;

template <class F>
bool moments_ok(const char* what, F draw, int N) {
    double s = 0, ss = 0;
    for (int i = 0; i < N; ++i) { double v = draw(i); s += v; ss += v * v; }
    double mean = s / N, var = ss / N - mean * mean;
    double se_mean = std::sqrt((1.0 / 12.0) / N), se_var = std::sqrt(1.0 / (180.0 * N));
    bool ok = std::fabs(mean - 0.5) <= kSigma * se_mean && std::fabs(var - 1.0 / 12.0) <= kSigma * se_var;
    if (!ok) std::printf("  moment fail in %s (mean=%.6f z=%.2f, var=%.6f z=%.2f)\n", what, mean, (mean - 0.5) / se_mean, var, (var - 1.0 / 12.0) / se_var);
    return ok;
}

}  // namespace

// ------------------------------------------------------------------ math

TEST(vec_arithmetic_matches_hand_values) {
    // Given a=(1,2,3), b=(4,5,6) the componentwise results are computed by hand.
    Vec3 a{1, 2, 3}, b{4, 5, 6};
    CHECK(near3(a + b, Vec3{5, 7, 9}, 0));
    CHECK(near3(b - a, Vec3{3, 3, 3}, 0));  // operand order matters: b-a, not a-b
    CHECK(near3(-a, Vec3{-1, -2, -3}, 0));
    CHECK(near3(a * b, Vec3{4, 10, 18}, 0));  // Hadamard product, not dot
    CHECK(near3(a * 2.0f, Vec3{2, 4, 6}, 0));
    CHECK(near3(2.0f * a, Vec3{2, 4, 6}, 0));
    CHECK(near3(a / 2.0f, Vec3{0.5f, 1.0f, 1.5f}, 1e-7));
    Vec3 c = a; c += b; CHECK(near3(c, Vec3{5, 7, 9}, 0));
    c = a; c *= b; CHECK(near3(c, Vec3{4, 10, 18}, 0));
    c = a; c *= 3.0f; CHECK(near3(c, Vec3{3, 6, 9}, 0));
}

TEST(dot_and_cross_match_hand_values) {
    Vec3 a{1, 2, 3}, b{4, 5, 6};
    CHECK_NEAR(dot(a, b), 32.0, 0);                       // 4+10+18
    CHECK(near3(cross(a, b), Vec3{-3, 6, -3}, 0));        // (2*6-3*5, 3*4-1*6, 1*5-2*4)
    CHECK(near3(cross(Vec3{1, 0, 0}, Vec3{0, 1, 0}), Vec3{0, 0, 1}, 0));   // right-handed
    CHECK(near3(cross(Vec3{0, 1, 0}, Vec3{1, 0, 0}), Vec3{0, 0, -1}, 0));  // swapped args flip sign
    CHECK_NEAR(length(Vec3{2, 3, 6}), 7.0, 1e-6);
    CHECK_NEAR(length2(Vec3{2, 3, 6}), 49.0, 0);
}

TEST(cross_and_dot_identities_hold_for_random_vectors) {
    Rnd rnd(1);
    for (int i = 0; i < 20000; ++i) {
        Vec3 a = rnd.any(10), b = rnd.any(10);
        Vec3 c = cross(a, b);
        double scale = (length(a) * length(b)) + 1e-3;
        // c is orthogonal to both factors (identity a.(a x b) = 0)
        CHECK_NEAR(dot(c, a), 0, 2e-5 * scale * length(a) + 1e-4);
        CHECK_NEAR(dot(c, b), 0, 2e-5 * scale * length(b) + 1e-4);
        // anticommutativity
        CHECK(near3(cross(b, a), -c, 1e-4));
        // Lagrange: |a x b|^2 = |a|^2|b|^2 - (a.b)^2
        // reference in double precision from the raw components (float cancellation is the only error source)
        double ax = a.x, ay = a.y, az = a.z, bx = b.x, by = b.y, bz = b.z;
        double aa = ax * ax + ay * ay + az * az, bb = bx * bx + by * by + bz * bz, ab = ax * bx + ay * by + az * bz;
        double lhs = length2(c), rhs = aa * bb - ab * ab;
        CHECK_NEAR(lhs, rhs, 1e-5 * (aa * bb) + 1e-4);
        // dot symmetric and bilinear in scale
        CHECK_NEAR(dot(a, b), dot(b, a), 1e-5 * scale);
        CHECK_NEAR(dot(a * 2.0f, b), 2.0 * dot(a, b), 1e-4 * scale);
    }
}

TEST(normalize_gives_unit_vectors_and_keeps_direction) {
    CHECK(near3(normalize(Vec3{3, 4, 0}), Vec3{0.6f, 0.8f, 0}, 1e-7));
    CHECK(near3(normalize(Vec3{0, 0, -5}), Vec3{0, 0, -1}, 1e-7));
    Rnd rnd(2);
    for (int i = 0; i < 20000; ++i) {
        Vec3 a = rnd.any(100);
        Vec3 n = normalize(a);
        CHECK_NEAR(length(n), 1.0, 2e-6);
        // parallel to the input and same orientation: a = |a| n
        CHECK(near3(n * length(a), a, 1e-4));
    }
}

TEST(normalize_of_zero_vector_stays_finite) {
    // Edge: zero length must not produce NaN/Inf (division by zero guard).
    Vec3 z = normalize(Vec3{0, 0, 0});
    CHECK(is_finite3(z));
    CHECK(near3(z, Vec3{0, 0, 0}, 0));
}

TEST(reflect_obeys_mirror_law) {
    // Hand-computed: incoming (1,-1,0) on the +y plane leaves as (1,1,0).
    CHECK(near3(reflect(Vec3{1, -1, 0}, Vec3{0, 1, 0}), Vec3{1, 1, 0}, 0));
    CHECK(near3(reflect(Vec3{0, -1, 0}, Vec3{0, 1, 0}), Vec3{0, 1, 0}, 0));
    // d=(0,0,-1), n=(0,0.6,0.8): d.n=-0.8, r = d + 1.6 n = (0, 0.96, 0.28)
    CHECK(near3(reflect(Vec3{0, 0, -1}, Vec3{0, 0.6f, 0.8f}), Vec3{0, 0.96f, 0.28f}, 1e-6));
    Rnd rnd(3);
    for (int i = 0; i < 20000; ++i) {
        Vec3 d = rnd.unit(), n = rnd.unit();
        Vec3 r = reflect(d, n);
        CHECK_NEAR(length(r), 1.0, 3e-6);                       // length preserved
        CHECK_NEAR(dot(r, n), -dot(d, n), 3e-6);                // normal component flips
        CHECK(near3(r - n * dot(r, n), d - n * dot(d, n), 3e-6));  // tangential part preserved
        CHECK(near3(reflect(r, n), d, 5e-6));                   // involution
    }
}

TEST(make_basis_is_orthonormal_for_special_and_random_normals) {
    std::vector<Vec3> normals = {
        {0, 0, 1}, {0, 0, -1}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0},
        // near the poles and near the z=0 branch switch
        {1e-3f, 0, std::sqrt(1 - 1e-6f)}, {1e-3f, 0, -std::sqrt(1 - 1e-6f)},
        {std::sqrt(1 - 1e-12f), 0, 1e-6f}, {std::sqrt(1 - 1e-12f), 0, -1e-6f},
        normalize(Vec3{1, 1, 1}), normalize(Vec3{-1, 2, -3}),
    };
    Rnd rnd(4);
    for (int i = 0; i < 20000; ++i) normals.push_back(rnd.unit());
    int bad = 0;
    for (Vec3 n : normals) {
        Vec3 t, b;
        make_basis(n, &t, &b);
        bool ok = std::fabs(length(t) - 1) < 1e-5 && std::fabs(length(b) - 1) < 1e-5 &&
                  std::fabs(dot(t, b)) < 1e-5 && std::fabs(dot(t, n)) < 1e-5 && std::fabs(dot(b, n)) < 1e-5;
        // completeness: the frame reconstructs an arbitrary vector
        Vec3 v{0.3f, -0.7f, 0.5f};
        Vec3 rec = t * dot(v, t) + b * dot(v, b) + n * dot(v, n);
        ok = ok && near3(rec, v, 1e-5) && is_finite3(t) && is_finite3(b);
        if (!ok) {
            if (bad < 3) std::printf("  basis fail n=(%g,%g,%g) t=(%g,%g,%g) b=(%g,%g,%g)\n", n.x, n.y, n.z, t.x, t.y, t.z, b.x, b.y, b.z);
            ++bad;
        }
    }
    CHECK(bad == 0);
}

TEST(luminance_uses_rec709_weights) {
    CHECK_NEAR(luminance(Vec3{1, 1, 1}), 1.0, 1e-6);      // weights sum to 1
    CHECK_NEAR(luminance(Vec3{0, 0, 0}), 0.0, 0);
    CHECK_NEAR(luminance(Vec3{1, 0, 0}), 0.2126, 1e-7);   // channel order: R, G, B
    CHECK_NEAR(luminance(Vec3{0, 1, 0}), 0.7152, 1e-7);
    CHECK_NEAR(luminance(Vec3{0, 0, 1}), 0.0722, 1e-7);
    Rnd rnd(5);
    for (int i = 0; i < 1000; ++i) {  // linear and homogeneous
        Vec3 a = rnd.any(3), b = rnd.any(3);
        CHECK_NEAR(luminance(a + b), luminance(a) + luminance(b), 1e-5);
        CHECK_NEAR(luminance(a * 2.5f), 2.5 * luminance(a), 1e-5);
    }
}

TEST(scalar_helpers_handle_boundaries) {
    CHECK_NEAR(clampf(-1, 0, 1), 0, 0);
    CHECK_NEAR(clampf(2, 0, 1), 1, 0);
    CHECK_NEAR(clampf(0.25f, 0, 1), 0.25, 0);
    CHECK_NEAR(clampf(0, 0, 1), 0, 0);   // boundary values pass through
    CHECK_NEAR(clampf(1, 0, 1), 1, 0);
    CHECK_NEAR(max_component(Vec3{1, 5, 3}), 5, 0);
    CHECK_NEAR(max_component(Vec3{7, 5, 3}), 7, 0);
    CHECK_NEAR(max_component(Vec3{1, 5, 9}), 9, 0);
    CHECK_NEAR(max_component(Vec3{-4, -5, -3}), -3, 0);  // negative inputs: not clamped at 0
    CHECK(near3(vmin(Vec3{1, 5, -3}, Vec3{2, 4, -6}), Vec3{1, 4, -6}, 0));
    CHECK(near3(vmax(Vec3{1, 5, -3}, Vec3{2, 4, -6}), Vec3{2, 5, -3}, 0));
    CHECK_NEAR(fminf3(1, 2), 1, 0);
    CHECK_NEAR(fmaxf3(1, 2), 2, 0);
}

TEST(is_finite3_rejects_nan_and_inf_in_any_component) {
    float nan = std::nanf(""), inf = kInf * 2.0f;
    CHECK(is_finite3(Vec3{1, 2, 3}));
    CHECK(!is_finite3(Vec3{nan, 0, 0}));
    CHECK(!is_finite3(Vec3{0, nan, 0}));
    CHECK(!is_finite3(Vec3{0, 0, nan}));
    CHECK(!is_finite3(Vec3{inf, 0, 0}));
    CHECK(!is_finite3(Vec3{0, 0, -inf}));
    CHECK(is_finite3(Vec3{kInf, 0, 0}));  // kInf is the largest finite float, not infinity
}

TEST(constants_are_consistent) {
    CHECK_NEAR(kPi, 3.14159265358979323846, 1e-7);
    CHECK_NEAR(kInvPi * kPi, 1.0, 2e-7);
    CHECK_NEAR(kInf, 3.4028234663852886e38, 1e32);
}

// ------------------------------------------------------------------ rng

TEST(rng_is_a_pure_function_of_its_arguments) {
    PathKey k = make_path_key(1234, 77, 5);
    float first = rng(k, 3, 2);
    for (int i = 0; i < 100; ++i) CHECK(rng(k, 3, 2) == first);
    PathKey k2 = make_path_key(1234, 77, 5);
    CHECK(k2.k0 == k.k0 && k2.k1 == k.k1);
    CHECK(rng(k2, 3, 2) == first);
}

TEST(rng_draw_order_does_not_change_values) {
    // Given a table drawn depth-major ascending, when drawn dim-major descending with unrelated
    // keys interleaved, then every cell is bit-identical (this is what lets wavefront == megakernel).
    PathKey k = make_path_key(99, 1234, 17), other = make_path_key(100, 1, 1);
    float fwd[8][kDimCount], rev[8][kDimCount];
    for (int d = 0; d < 8; ++d)
        for (int m = 0; m < (int)kDimCount; ++m) fwd[d][m] = rng(k, d, m);
    volatile float sink = 0;
    for (int m = (int)kDimCount - 1; m >= 0; --m)
        for (int d = 7; d >= 0; --d) {
            sink = sink + rng(other, d, m);  // interleaved foreign draw must not disturb state
            rev[d][m] = rng(k, d, m);
        }
    CHECK(std::memcmp(fwd, rev, sizeof(fwd)) == 0);
}

TEST(rng_values_lie_in_half_open_unit_interval_with_24_bit_grid) {
    float lo = 1, hi = 0;
    int off_grid = 0;
    for (int i = 0; i < 1000000; ++i) {
        float v = rng(make_path_key(7, (uint32_t)i, 0), 0, 0);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        float scaled = v * 16777216.0f;
        if (scaled != std::floor(scaled)) ++off_grid;
    }
    CHECK(lo >= 0.0f);
    CHECK(hi < 1.0f);                         // 1.0 must never be returned
    CHECK(hi <= 1.0f - 1.0f / 16777216.0f);   // largest representable value of the 24-bit grid
    CHECK(off_grid == 0);                     // spec: 24 bits of resolution
    CHECK(lo < 1e-4f && hi > 1 - 1e-4f);      // the whole interval is actually used
}

TEST(rng_is_uniform_in_mean_and_variance_across_every_key_axis) {
    const int N = 1000000;
    CHECK(moments_ok("varying pixel", [](int i) { return rng(make_path_key(1234, (uint32_t)i, 0), 0, 0); }, N));
    CHECK(moments_ok("varying sample", [](int i) { return rng(make_path_key(1234, 5, (uint32_t)i), 1, 3); }, N));
    CHECK(moments_ok("varying seed", [](int i) { return rng(make_path_key((uint32_t)i, 5, 9), 2, 7); }, N));
    PathKey fixed = make_path_key(42, 42, 42);
    CHECK(moments_ok("varying depth and dim", [&](int i) { return rng(fixed, (uint32_t)(i / 16), (uint32_t)(i % 16)); }, N));
}

TEST(moment_check_has_power_negative_control_rejects_biased_streams) {
    // The same checker must reject a stream whose mean is shifted by 0.003 (10 SE) or whose variance is
    // scaled by 3% (about 8 SE at N=1e6), otherwise the widened bound would have hidden real bias.
    const int N = 1000000;
    auto base = [](int i) { return (double)rng(make_path_key(1234, (uint32_t)i, 0), 0, 0); };
    CHECK(!moments_ok("control: mean+0.003", [&](int i) { return base(i) + 0.003; }, N));
    CHECK(!moments_ok("control: var*1.03", [&](int i) { return 0.5 + (base(i) - 0.5) * std::sqrt(1.03); }, N));
}

TEST(rng_histogram_is_flat) {
    // 16 equal bins, N draws: chi^2 with 15 dof; 99.9% quantile is 37.7.
    const int N = 1000000, B = 16;
    long cnt[B] = {0};
    for (int i = 0; i < N; ++i) cnt[(int)(rng(make_path_key(3, (uint32_t)i, 1), 0, 4) * B)]++;
    double chi2 = 0, e = (double)N / B;
    for (int b = 0; b < B; ++b) chi2 += (cnt[b] - e) * (cnt[b] - e) / e;
    CHECK(chi2 < 37.7);
}

TEST(rng_neighbouring_dimensions_are_uncorrelated) {
    // Pearson r of independent uniforms has SE ~ 1/sqrt(N) = 1e-3; bound at 3 SE.
    const int N = 1000000;
    for (int d = 0; d + 1 < (int)kDimCount; ++d) {
        Pearson p;
        for (int i = 0; i < N / 4; ++i) {
            PathKey k = make_path_key(11, (uint32_t)i, 2);
            p.add(rng(k, 0, d), rng(k, 0, d + 1));
        }
        CHECK_NEAR(p.r(), 0.0, 3.0 / std::sqrt((double)(N / 4)));
    }
    Pearson depth;  // neighbouring depths of the same dim
    for (int i = 0; i < N; ++i) {
        PathKey k = make_path_key(11, (uint32_t)i, 2);
        depth.add(rng(k, 0, 5), rng(k, 1, 5));
    }
    CHECK_NEAR(depth.r(), 0.0, 3.0 / std::sqrt((double)N));
}

TEST(rng_neighbouring_pixels_samples_and_seeds_are_uncorrelated) {
    const int N = 1000000;
    Pearson px, row, smp, seed;
    for (int i = 0; i < N; ++i) {
        float a = rng(make_path_key(1234, (uint32_t)i, 0), 0, 0);
        px.add(a, rng(make_path_key(1234, (uint32_t)i + 1, 0), 0, 0));              // pixel to its right
        row.add(a, rng(make_path_key(1234, (uint32_t)i + 1280, 0), 0, 0));          // pixel below (width 1280)
        smp.add(a, rng(make_path_key(1234, (uint32_t)i, 1), 0, 0));                 // next sample, same pixel
        seed.add(a, rng(make_path_key(1235, (uint32_t)i, 0), 0, 0));                // next seed
    }
    double tol = 3.0 / std::sqrt((double)N);
    CHECK_NEAR(px.r(), 0.0, tol);
    CHECK_NEAR(row.r(), 0.0, tol);
    CHECK_NEAR(smp.r(), 0.0, tol);
    CHECK_NEAR(seed.r(), 0.0, tol);
}

TEST(rng_neighbouring_streams_have_no_joint_structure) {
    // 2-D chi^2 on a 16x16 grid (255 dof, 99.9% quantile ~ 330). Catches non-linear dependence
    // (e.g. y = 1-x or y = x^2), which Pearson r misses.
    const int N = 1000000, G = 16;
    auto chi2_of = [&](auto pair_fn) {
        std::vector<long> c(G * G, 0);
        for (int i = 0; i < N; ++i) {
            auto p = pair_fn(i);
            c[(int)(p.first * G) * G + (int)(p.second * G)]++;
        }
        double e = (double)N / (G * G), chi = 0;
        for (long v : c) chi += (v - e) * (v - e) / e;
        return chi;
    };
    CHECK(chi2_of([](int i) { PathKey k = make_path_key(5, (uint32_t)i, 0); return std::make_pair(rng(k, 0, 0), rng(k, 0, 1)); }) < 330.0);
    CHECK(chi2_of([](int i) { return std::make_pair(rng(make_path_key(5, (uint32_t)i, 0), 0, 0), rng(make_path_key(5, (uint32_t)i + 1, 0), 0, 0)); }) < 330.0);
}

TEST(rng_streams_of_different_seeds_are_uncorrelated_and_different) {
    const int N = 1000000;
    Pearson p;
    int equal = 0;
    for (int i = 0; i < N; ++i) {
        float a = rng(make_path_key(1, (uint32_t)i, 0), 0, 0), b = rng(make_path_key(2, (uint32_t)i, 0), 0, 0);
        p.add(a, b);
        if (a == b) ++equal;
    }
    CHECK_NEAR(p.r(), 0.0, 3.0 / std::sqrt((double)N));
    // Two independent 24-bit streams agree in ~N/2^24 = 0.06 positions; allow a handful.
    CHECK(equal <= 5);
}

TEST(rng_depth_and_dim_slots_do_not_alias) {
    // 8 depths x 16 dims = 128 draws from one key; each must differ (collision prob ~ 5e-4 for a
    // random 24-bit source, so any repeat indicates aliasing such as ctr = depth + dim).
    PathKey k = make_path_key(1234, 0, 0);
    std::set<float> seen;
    for (uint32_t d = 0; d < 8; ++d)
        for (uint32_t m = 0; m < kDimCount; ++m) seen.insert(rng(k, d, m));
    CHECK(seen.size() == 128u);
}

TEST(path_key_changes_when_any_single_coordinate_changes) {
    // For fixed other coordinates the key map is a composition of bijections, so 1e5 values must be
    // collision-free in k0 exactly (not just probabilistically).
    const int N = 100000;
    std::set<uint32_t> a, b, c;
    for (int i = 0; i < N; ++i) {
        a.insert(make_path_key((uint32_t)i, 10, 20).k0);
        b.insert(make_path_key(10, (uint32_t)i, 20).k0);
        c.insert(make_path_key(10, 20, (uint32_t)i).k0);
    }
    CHECK(a.size() == (size_t)N);
    CHECK(b.size() == (size_t)N);
    CHECK(c.size() == (size_t)N);
}

TEST(path_key_grid_over_seed_pixel_sample_has_no_collisions) {
    // 16*32*32 = 16384 triples; k0 and k1 are independent hash chains (effective key ~64 bits), so the
    // birthday expectation of a full-key collision is 16384^2 / 2 / 2^64 ~ 7e-12.
    std::set<std::pair<uint32_t, uint32_t>> keys;
    for (uint32_t s = 0; s < 16; ++s)
        for (uint32_t p = 0; p < 32; ++p)
            for (uint32_t m = 0; m < 32; ++m) {
                PathKey k = make_path_key(s, p, m);
                keys.insert({k.k0, k.k1});
            }
    CHECK(keys.size() == 16384u);
}

TEST(path_key_full_keys_are_unique_for_one_million_pixel_sample_pairs) {
    // 1000 pixels x 1000 samples: no two full (k0,k1) keys may be equal (expected collisions ~ 3e-8 at 64 bits;
    // a 32-bit effective key would collide ~116 times).
    const uint32_t P = 1000, S = 1000;
    std::vector<uint64_t> keys;
    keys.reserve((size_t)P * S);
    for (uint32_t p = 0; p < P; ++p)
        for (uint32_t m = 0; m < S; ++m) {
            PathKey k = make_path_key(1234, p, m);
            keys.push_back(((uint64_t)k.k0 << 32) | k.k1);
        }
    std::sort(keys.begin(), keys.end());
    CHECK(std::adjacent_find(keys.begin(), keys.end()) == keys.end());
}

TEST(path_key_k1_is_not_a_function_of_k0) {
    // For n=1e6 random 32-bit values among M=2^32 slots the expected number of distinct values is
    // M(1-exp(-n/M)) ~ n - 116.4, with collision count ~ Poisson(116.4), sd 10.8. If k1 were independent of
    // k0 its distinct count follows this law; k0 must be within 3 SD of it and k1 no more than 3 SD below it. (A k1 that is
    // a bijection of k0 is indistinguishable here, which is why the previous test checks full-key
    // uniqueness AND the k0-collision count below.)
    const uint32_t P = 1000, S = 1000;
    std::vector<uint32_t> k0s, k1s;
    for (uint32_t p = 0; p < P; ++p)
        for (uint32_t m = 0; m < S; ++m) {
            PathKey k = make_path_key(77, p, m);
            k0s.push_back(k.k0); k1s.push_back(k.k1);
        }
    auto distinct = [](std::vector<uint32_t> v) {
        std::sort(v.begin(), v.end());
        return (double)(std::unique(v.begin(), v.end()) - v.begin());
    };
    double n = (double)P * S, M = 4294967296.0;
    double expect = M * (1 - std::exp(-n / M)), sd = std::sqrt(n * n / 2 / M);
    // k1 is additive in (pixel, sample) before its last hash, so on this grid it is measurably *more* uniform
    // than random (0 collisions observed on 4 seeds); that is harmless, so only the lower side is bounded.
    CHECK(distinct(k1s) >= expect - 3 * sd);
    CHECK_NEAR(distinct(k0s), expect, 3 * sd);
    // Independence evidence: positions where k0 collides must not systematically be positions where k1 collides.
    // Count pairs equal in k0 (expected ~116) and, among them, pairs also equal in k1 (expected ~0).
    std::vector<std::pair<uint32_t, uint32_t>> pr;
    for (size_t i = 0; i < k0s.size(); ++i) pr.push_back({k0s[i], k1s[i]});
    std::sort(pr.begin(), pr.end());
    int k0_equal = 0, both_equal = 0;
    for (size_t i = 1; i < pr.size(); ++i)
        if (pr[i].first == pr[i - 1].first) { ++k0_equal; if (pr[i].second == pr[i - 1].second) ++both_equal; }
    CHECK(both_equal == 0);
    CHECK(k0_equal <= 116 + 40);   // 116 expected, ~3.7 SD upper allowance; also proves k0 itself is not degenerate
}

TEST(path_key_is_not_symmetric_in_its_arguments) {
    // Plausible bug: swapped arguments / commutative mixing makes (seed,pixel,sample) permutations collide.
    int same = 0;
    for (uint32_t i = 1; i < 2000; ++i) {
        PathKey a = make_path_key(i, i + 1, i + 2), b = make_path_key(i + 1, i, i + 2), c = make_path_key(i, i + 2, i + 1);
        if (a.k0 == b.k0) ++same;
        if (a.k0 == c.k0) ++same;
    }
    CHECK(same == 0);
}

TEST(path_key_avalanche_flipping_one_input_bit_flips_half_of_the_key_bits) {
    // Mean Hamming distance of k0 under a one-bit input flip is 16, sd per sample sqrt(8)=2.83;
    // with N samples SE = 2.83/sqrt(N). Bound: 3 SE (N=20000 -> 0.06).
    Rnd rnd(6);
    const int N = 20000;
    for (int axis = 0; axis < 3; ++axis) {
        double total = 0;
        for (int i = 0; i < N; ++i) {
            uint32_t v[3] = {(uint32_t)rnd.g(), (uint32_t)rnd.g(), (uint32_t)rnd.g()};
            uint32_t w[3] = {v[0], v[1], v[2]};
            w[axis] ^= 1u << (i % 32);
            uint32_t x = make_path_key(v[0], v[1], v[2]).k0 ^ make_path_key(w[0], w[1], w[2]).k0;
            total += __builtin_popcount(x);
        }
        CHECK_NEAR(total / N, 16.0, 3 * std::sqrt(8.0 / N));
    }
}

TEST(hash_u32_is_a_bijection_on_a_sample_and_mixes_zero) {
    // lowbias32 is a permutation of 2^32; any repeat among 1e6 consecutive inputs would be a defect.
    std::set<uint32_t> out;
    for (uint32_t i = 0; i < 1000000; ++i) out.insert(hash_u32(i));
    CHECK(out.size() == 1000000u);
    CHECK(hash_u32(0) == 0u);  // known fixed point of xorshift-multiply hashes; documents why the key adds constants
    CHECK(make_path_key(0, 0, 0).k0 != 0u);  // ...and that a (0,0,0) path does not inherit it
}
