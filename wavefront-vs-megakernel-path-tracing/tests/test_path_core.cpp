// Tests for include/wf/path_core.hpp (the single estimator), header-only, at solitary / sociable width.
// Oracles: closed-form physics (Fresnel, Snell, solid-angle pdf, Le/(1-rho) furnace), independent
// double-precision reimplementations written below, and statistical bounds of 3 standard errors.
#include <cstdint>
#include <vector>

#include "test_framework.hpp"
#include "wf/math.hpp"
#include "wf/path_core.hpp"
#include "wf/rng.hpp"
#include "wf/scene.hpp"

using namespace wf;

namespace {

constexpr double kPiD = 3.14159265358979323846;

Vec3 V(float x, float y, float z) { return Vec3{x, y, z}; }

struct Acc {  // Welford running mean / variance
    double n = 0, mean = 0, m2 = 0;
    void add(double x) { n += 1; double d = x - mean; mean += d / n; m2 += d * (x - mean); }
    double var() const { return n > 1 ? m2 / (n - 1) : 0; }
    double se() const { return std::sqrt(var() / n); }
};

Material mat(int type, Vec3 albedo, Vec3 emission = V(0, 0, 0), float ior = 1.5f) {
    return Material{type, ior, 0, 0, albedo, 0, emission, 0};
}
Sphere sph(Vec3 c, float r, int m) { return Sphere{c, r, m, {0, 0, 0}}; }

// Owns the arrays behind a SceneView. Single-leaf BVH over all primitives is generated on demand.
struct Mini {
    std::vector<Sphere> sp;
    std::vector<Vec3> vtx;
    std::vector<Triangle> tri;
    std::vector<Material> mats;
    std::vector<Light> lights;
    std::vector<BvhNode> nodes;
    std::vector<int> order;
    Vec3 env = V(0, 0, 0);
    Camera cam{V(0, 0, 0), V(0, 0, -1), V(0, 1, 0), 60.0f};
    int width = 8, height = 8;

    void single_leaf(float half) {
        int n = (int)(sp.size() + tri.size());
        order.clear();
        for (int i = 0; i < n; ++i) order.push_back(i);
        nodes = {BvhNode{V(-half, -half, -half), -1, V(half, half, half), 0, n, {0, 0, 0}}};
    }
    SceneView view() const {
        SceneView v{};
        v.spheres = sp.data(); v.vertices = vtx.data(); v.triangles = tri.data(); v.materials = mats.data();
        v.nodes = nodes.data(); v.prim_order = order.data(); v.lights = lights.data();
        v.n_spheres = (int)sp.size(); v.n_triangles = (int)tri.size(); v.n_materials = (int)mats.size();
        v.n_nodes = (int)nodes.size(); v.n_lights = (int)lights.size();
        v.env = env; v.camera = make_camera_rig(cam, width, height); v.width = width; v.height = height;
        return v;
    }
};

// One sphere light of the given radius/centre; its prim id is its index.
void add_sphere_light(Mini& m, Vec3 c, float r, Vec3 Le) {
    m.mats.push_back(mat(kEmissive, V(0, 0, 0), Le));
    m.sp.push_back(sph(c, r, (int)m.mats.size() - 1));
    m.lights.push_back(Light{(int)m.sp.size() - 1, 4.0f * kPi * r * r, {0, 0}});
}

PathState fresh_path(PathKey key) {
    PathState ps{};
    ps.throughput = V(1, 1, 1);
    ps.L = V(0, 0, 0);
    ps.depth = 0; ps.specular = 1; ps.prev_pdf = 0; ps.key = key;
    ps.ray = Ray{V(0, 0, 1), kRayEps, V(0, 0, -1), kInf};
    return ps;
}

Surface flat_surface(int material, bool inside = false) {
    Surface f{};
    f.p = V(0, 0, 0);
    f.n = V(0, 0, 1);               // faces against the ray (rays come from +z going down)
    f.n_out = inside ? V(0, 0, -1) : V(0, 0, 1);
    f.inside = inside ? 1 : 0;
    f.material = material;
    return f;
}

double len(double x, double y, double z) { return std::sqrt(x * x + y * y + z * z); }

// Independent Fresnel via the trigonometric (Fresnel sine / tangent law) form, in double precision.
double fresnel_ref(double cos_i, double eta_i, double eta_t) {
    double sin_i = std::sqrt(std::max(0.0, 1 - cos_i * cos_i));
    double sin_t = eta_i / eta_t * sin_i;
    if (sin_t >= 1) return 1;
    double ti = std::acos(cos_i), tt = std::asin(sin_t);
    double rs = std::sin(ti - tt) / std::sin(ti + tt);
    double rp = std::tan(ti - tt) / std::tan(ti + tt);
    return 0.5 * (rs * rs + rp * rp);
}

double chi2_16(const std::vector<double>& vals) {  // vals in [0,1)
    long c[16] = {0};
    for (double v : vals) c[std::min(15, (int)(v * 16))]++;
    double e = (double)vals.size() / 16, chi = 0;
    for (long k : c) chi += (k - e) * (k - e) / e;
    return chi;
}

}  // namespace

// ------------------------------------------------------------------ power heuristic

TEST(power_heuristic_pairs_sum_to_one_and_match_hand_values) {
    CHECK_NEAR(power_heuristic(2, 1), 0.8, 1e-7);   // 4/(4+1)
    CHECK_NEAR(power_heuristic(1, 3), 0.1, 1e-7);   // 1/(1+9)
    CHECK_NEAR(power_heuristic(1, 2), 0.2, 1e-7);   // not the balance heuristic 1/3
    CHECK_NEAR(power_heuristic(5, 5), 0.5, 1e-7);
    CHECK_NEAR(power_heuristic(1, 0), 1.0, 0);      // only strategy a can generate the sample
    CHECK_NEAR(power_heuristic(0, 1), 0.0, 0);
    uint32_t s = 1;
    for (int i = 0; i < 5000; ++i) {
        s = hash_u32(s);
        float a = 1e-3f + (s >> 8) * (100.0f / 16777216.0f);
        s = hash_u32(s);
        float b = 1e-3f + (s >> 8) * (100.0f / 16777216.0f);
        // weights of the two strategies for the same sample must sum to one (MIS partition of unity)
        CHECK_NEAR(power_heuristic(a, b) + power_heuristic(b, a), 1.0, 2e-6);
        double e = (double)a * a / ((double)a * a + (double)b * b);
        CHECK_NEAR(power_heuristic(a, b), e, 1e-6);
        // monotone in own pdf
        CHECK(power_heuristic(a * 2, b) >= power_heuristic(a, b));
    }
}

TEST(power_heuristic_is_finite_for_degenerate_and_large_inputs) {
    float w = power_heuristic(0, 0);
    CHECK(std::isfinite(w) && w >= 0 && w <= 1);   // 0/0 must not produce NaN
    CHECK_NEAR(power_heuristic(1e15f, 1e15f), 0.5, 1e-6);
    CHECK_NEAR(power_heuristic(1e15f, 1.0f), 1.0, 1e-6);
}

// ------------------------------------------------------------------ fresnel

TEST(fresnel_at_normal_incidence_equals_squared_index_contrast) {
    double ns[] = {1.1, 1.33, 1.5, 2.0, 2.4};
    for (double n : ns) {
        double e = (n - 1) / (n + 1); e *= e;
        CHECK_NEAR(fresnel_dielectric(1.0f, 1.0f, (float)n), e, 2e-6);   // air -> glass
        CHECK_NEAR(fresnel_dielectric(1.0f, (float)n, 1.0f), e, 2e-6);   // glass -> air, same value
    }
    CHECK_NEAR(fresnel_dielectric(1.0f, 1.0f, 1.5f), 0.04, 1e-6);       // hand value 0.5^2/2.5^2
    CHECK_NEAR(fresnel_dielectric(0.7f, 1.0f, 1.0f), 0.0, 1e-6);        // index-matched: no reflection
}

TEST(fresnel_is_total_beyond_the_critical_angle_and_below_one_before_it) {
    // glass(1.5) -> air: critical sin = 2/3, cos_c = sqrt(5)/3 = 0.745356.
    CHECK(fresnel_dielectric(0.70f, 1.5f, 1.0f) == 1.0f);   // sin_i = 0.714 > 2/3: TIR
    CHECK(fresnel_dielectric(0.30f, 1.5f, 1.0f) == 1.0f);
    CHECK(fresnel_dielectric(0.0f, 1.5f, 1.0f) == 1.0f);
    CHECK(fresnel_dielectric(0.76f, 1.5f, 1.0f) < 1.0f);    // sin_i = 0.650 < 2/3: partial reflection
    CHECK(fresnel_dielectric(0.80f, 1.5f, 1.0f) < 0.5f);
    // water -> air at cos=0.5 (60 deg): sin_t = 1.33*0.866 > 1 -> TIR
    CHECK(fresnel_dielectric(0.5f, 1.33f, 1.0f) == 1.0f);
    // air -> glass never reflects totally except at exactly grazing
    CHECK(fresnel_dielectric(0.05f, 1.0f, 1.5f) < 1.0f);
}

TEST(fresnel_matches_independent_trigonometric_formula_over_random_angles) {
    uint32_t s = 7;
    int checked = 0;
    for (int i = 0; i < 20000; ++i) {
        s = hash_u32(s); double c = 0.01 + 0.97 * ((s >> 8) / 16777216.0);       // cos_i in (0.01, 0.98)
        s = hash_u32(s); double n = 1.05 + 1.6 * ((s >> 8) / 16777216.0);
        bool into = (s & 1) != 0;
        double ei = into ? 1.0 : n, et = into ? n : 1.0;
        // skip within 1e-3 of the critical angle where float sqrt amplifies rounding
        double sin_t = ei / et * std::sqrt(1 - c * c);
        if (std::fabs(sin_t - 1) < 1e-3) continue;
        // skip the Brewster neighbourhood, where tan(ti+tt) -> inf and the reference loses digits
        CHECK_NEAR(fresnel_dielectric((float)c, (float)ei, (float)et), fresnel_ref(c, ei, et), 3e-5);
        ++checked;
    }
    CHECK(checked > 15000);
}

TEST(fresnel_increases_monotonically_toward_one_at_grazing_incidence) {
    for (int dir = 0; dir < 2; ++dir) {
        float ei = dir ? 1.5f : 1.0f, et = dir ? 1.0f : 1.5f;
        float prev = fresnel_dielectric(1.0f, ei, et);
        int violations = 0;
        for (int k = 1; k <= 2000; ++k) {
            float c = 1.0f - k / 2000.0f;  // cos_i from 1 down to 0 (angle from 0 up to 90 deg)
            float f = fresnel_dielectric(c, ei, et);
            if (f < prev - 1e-6f) ++violations;
            if (!(f >= 0.0f && f <= 1.0f)) ++violations;
            prev = f;
        }
        CHECK(violations == 0);
        CHECK_NEAR(fresnel_dielectric(0.0f, ei, et), 1.0, 1e-6);   // grazing -> full reflection
    }
}

TEST(fresnel_is_reciprocal_under_reversal_of_the_path) {
    // F(theta_i, ni->nt) == F(theta_t, nt->ni) for the refracted angle theta_t (Stokes reciprocity).
    for (double c = 0.1; c < 0.99; c += 0.05) {
        double ei = 1.0, et = 1.52;
        double sin_t = ei / et * std::sqrt(1 - c * c), ct = std::sqrt(1 - sin_t * sin_t);
        CHECK_NEAR(fresnel_dielectric((float)c, (float)ei, (float)et), fresnel_dielectric((float)ct, (float)et, (float)ei), 2e-5);
    }
}

// ------------------------------------------------------------------ light sampling

TEST(sphere_light_pdf_equals_dist2_over_cos_area_for_every_sample) {
    Mini m;
    add_sphere_light(m, V(0, 0, 0), 2.0f, V(5, 5, 5));
    SceneView v = m.view();
    Vec3 from = V(0, 0, 10);
    int checked = 0;
    for (uint32_t i = 0; i < 5000; ++i) {
        LightSample ls = sample_light(v, from, make_path_key(1, i, 0), 0);
        // independent geometry: the sample must lie on the sphere with the radial normal
        CHECK_NEAR(len(ls.p.x, ls.p.y, ls.p.z), 2.0, 1e-4);
        CHECK(std::fabs(ls.n.x - ls.p.x / 2) < 1e-4 && std::fabs(ls.n.y - ls.p.y / 2) < 1e-4 && std::fabs(ls.n.z - ls.p.z / 2) < 1e-4);
        CHECK(ls.Le.x == 5.0f && ls.Le.y == 5.0f && ls.Le.z == 5.0f);
        double dx = ls.p.x - from.x, dy = ls.p.y - from.y, dz = ls.p.z - from.z;
        double d2 = dx * dx + dy * dy + dz * dz, d = std::sqrt(d2);
        double cosl = std::fabs((ls.p.x / 2) * dx + (ls.p.y / 2) * dy + (ls.p.z / 2) * dz) / d;
        double expect = d2 / (cosl * 4 * kPiD * 4.0);   // area = 4 pi r^2 with r=2, one light
        if (cosl > 1e-3) { CHECK_NEAR(ls.pdf_solid / expect, 1.0, 2e-3); ++checked; }
    }
    CHECK(checked > 4000);
}

TEST(sphere_light_pdf_halves_with_two_lights_and_selection_is_uniform) {
    Mini m;
    add_sphere_light(m, V(0, 0, 0), 2.0f, V(5, 5, 5));
    add_sphere_light(m, V(30, 0, 0), 2.0f, V(1, 2, 3));
    SceneView v = m.view();
    Vec3 from = V(0, 0, 10);
    int picked_second = 0; const int N = 20000;
    for (uint32_t i = 0; i < (uint32_t)N; ++i) {
        LightSample ls = sample_light(v, from, make_path_key(2, i, 0), 0);
        bool second = len(ls.p.x - 30, ls.p.y, ls.p.z) < 2.01;
        CHECK(second || len(ls.p.x, ls.p.y, ls.p.z) < 2.01);
        picked_second += second;
        Vec3 c = second ? V(30, 0, 0) : V(0, 0, 0);
        CHECK(ls.Le.x == (second ? 1.0f : 5.0f));   // emission follows the chosen light
        double dx = ls.p.x - from.x, dy = ls.p.y - from.y, dz = ls.p.z - from.z;
        double d2 = dx * dx + dy * dy + dz * dz, d = std::sqrt(d2);
        double cosl = std::fabs((ls.p.x - c.x) / 2 * dx + (ls.p.y - c.y) / 2 * dy + (ls.p.z - c.z) / 2 * dz) / d;
        if (cosl > 1e-3) CHECK_NEAR(ls.pdf_solid / (d2 / (cosl * 16 * kPiD) / 2.0), 1.0, 2e-3);  // pdf with n_lights=2
    }
    CHECK_NEAR((double)picked_second / N, 0.5, 3 * std::sqrt(0.25 / N));
}

TEST(sphere_light_inverse_pdf_integrates_to_twice_the_subtended_solid_angle) {
    // E[1/pdf_solid] over area-uniform samples = integral of cos/dist^2 dA over the whole surface
    // = near side + far side = 2 * Omega, with Omega = 2 pi (1 - sqrt(1 - (r/d)^2)) (cone half-angle asin(r/d)).
    Mini m;
    add_sphere_light(m, V(0, 0, 0), 2.0f, V(1, 1, 1));
    SceneView v = m.view();
    Acc a;
    for (uint32_t i = 0; i < 200000; ++i) {
        LightSample ls = sample_light(v, V(0, 0, 10), make_path_key(3, i, 0), 0);
        if (ls.pdf_solid > 0) a.add(1.0 / ls.pdf_solid);
    }
    double omega = 2 * kPiD * (1 - std::sqrt(1 - 0.04));
    CHECK_NEAR(a.mean, 2 * omega, 3 * a.se());
    CHECK(a.se() < 0.01 * 2 * omega);   // the test has enough power to see a 1% error
}

TEST(sphere_light_seen_from_inside_has_uniform_solid_angle_pdf) {
    Mini m;
    add_sphere_light(m, V(0, 0, 0), 5.0f, V(1, 1, 1));
    SceneView v = m.view();
    // From the centre every point is at distance r with cos=1: pdf = r^2/(4 pi r^2) = 1/(4 pi) exactly.
    for (uint32_t i = 0; i < 1000; ++i) {
        LightSample ls = sample_light(v, V(0, 0, 0), make_path_key(4, i, 0), 0);
        CHECK_NEAR(ls.pdf_solid, 1.0 / (4 * kPiD), 1e-5);
    }
    // From an off-centre interior point, E[1/pdf] = 4 pi (the whole sphere of directions).
    Acc a;
    for (uint32_t i = 0; i < 100000; ++i) {
        LightSample ls = sample_light(v, V(1, 2, -1), make_path_key(5, i, 0), 0);
        if (ls.pdf_solid > 0) a.add(1.0 / ls.pdf_solid);
    }
    CHECK_NEAR(a.mean, 4 * kPiD, 3 * a.se());
}

TEST(triangle_light_samples_are_area_uniform_with_correct_solid_angle_pdf) {
    // Light = triangle (0,0,0),(1,0,0),(0,1,0) as primitive id 1 (after one non-light sphere): area 0.5, normal +z.
    Mini m;
    m.mats.push_back(mat(kLambert, V(0.5f, 0.5f, 0.5f)));
    m.mats.push_back(mat(kEmissive, V(0, 0, 0), V(3, 3, 3)));
    m.sp.push_back(sph(V(10, 10, 10), 1, 0));
    m.vtx = {V(0, 0, 0), V(1, 0, 0), V(0, 1, 0)};
    m.tri.push_back(Triangle{0, 1, 2, 1});
    m.lights.push_back(Light{1, 0.5f, {0, 0}});
    SceneView v = m.view();
    Vec3 from = V(0.2f, 0.3f, 2.0f);
    const int N = 40000;
    Acc ax, ay, inv;
    for (uint32_t i = 0; i < (uint32_t)N; ++i) {
        LightSample ls = sample_light(v, from, make_path_key(6, i, 0), 0);
        CHECK(std::fabs(ls.p.z) < 1e-6 && ls.p.x >= -1e-6f && ls.p.y >= -1e-6f && ls.p.x + ls.p.y <= 1 + 1e-5f);
        CHECK(std::fabs(ls.n.z) > 0.99999f);
        CHECK(ls.Le.x == 3.0f);
        double dx = ls.p.x - from.x, dy = ls.p.y - from.y, dz = ls.p.z - from.z;
        double d2 = dx * dx + dy * dy + dz * dz, cosl = std::fabs(dz) / std::sqrt(d2);
        CHECK_NEAR(ls.pdf_solid / (d2 / (cosl * 0.5)), 1.0, 1e-4);
        ax.add(ls.p.x); ay.add(ls.p.y); inv.add(1.0 / ls.pdf_solid);
    }
    // Uniform over a right triangle with unit legs: centroid (1/3,1/3), Var = 1/18 per axis. A sampler without
    // the sqrt warp has mean 1/4 instead and fails here.
    double se = std::sqrt((1.0 / 18) / N);
    CHECK_NEAR(ax.mean, 1.0 / 3, 3 * se);
    CHECK_NEAR(ay.mean, 1.0 / 3, 3 * se);
    // E[1/pdf] = integral over the triangle of cos/dist^2 dA, evaluated by an independent midpoint quadrature.
    double omega = 0; const int G = 600;
    for (int i = 0; i < G; ++i) for (int j = 0; j < G - 1 - i; ++j) {   // full cells, midpoint rule
        double x = (i + 0.5) / G, y = (j + 0.5) / G;
        double dx = x - 0.2, dy = y - 0.3, d2 = dx * dx + dy * dy + 4.0;
        omega += (2.0 / std::sqrt(d2)) / d2 / ((double)G * G);
    }
    for (int i = 0; i < G; ++i) {   // the G half-cells on the hypotenuse, at their centroid
        double x = (i + 1.0 / 3) / G, y = (G - 1 - i + 1.0 / 3) / G;
        double dx = x - 0.2, dy = y - 0.3, d2 = dx * dx + dy * dy + 4.0;
        omega += 0.5 * (2.0 / std::sqrt(d2)) / d2 / ((double)G * G);
    }
    CHECK_NEAR(inv.mean, omega, 3 * inv.se());
}

TEST(light_pdf_at_hit_matches_hand_value_and_zero_at_grazing) {
    Mini m;
    add_sphere_light(m, V(0, 0, 0), 1.0f, V(1, 1, 1));
    SceneView v = m.view();
    // Hit at distance 4 head-on: cos=1, area 4 pi -> pdf = 16/(4 pi) = 1.2732395.
    CHECK_NEAR(light_pdf_at_hit(v, 0, 4.0f, V(0, 0, 1), V(0, 0, -1)), 16.0 / (4 * kPiD), 1e-5);
    // cos = 0.5 doubles it.
    CHECK_NEAR(light_pdf_at_hit(v, 0, 4.0f, V(0, 0, 1), V(0, 0.8660254f, -0.5f)), 32.0 / (4 * kPiD), 1e-5);
    // The sign of n_out must not matter (two-sided cosine).
    CHECK_NEAR(light_pdf_at_hit(v, 0, 4.0f, V(0, 0, 1), V(0, 0, 1)), 16.0 / (4 * kPiD), 1e-5);
    // Grazing: no pdf.
    CHECK(light_pdf_at_hit(v, 0, 4.0f, V(1, 0, 0), V(0, 0, 1)) == 0.0f);
    // t^2 scaling
    CHECK_NEAR(light_pdf_at_hit(v, 0, 8.0f, V(0, 0, 1), V(0, 0, -1)), 64.0 / (4 * kPiD), 1e-5);
    // Selection probability: two lights halve it.
    add_sphere_light(m, V(30, 0, 0), 1.0f, V(1, 1, 1));
    SceneView v2 = m.view();
    CHECK_NEAR(light_pdf_at_hit(v2, 0, 4.0f, V(0, 0, 1), V(0, 0, -1)), 8.0 / (4 * kPiD), 1e-5);
}

TEST(light_pdf_at_hit_agrees_with_sample_light_for_the_same_point) {
    // Scene: non-light sphere (prim 0), sphere light (prim 1), triangle light (prim 2) -> n_lights = 2.
    Mini m;
    m.mats.push_back(mat(kLambert, V(0.5f, 0.5f, 0.5f)));
    m.mats.push_back(mat(kEmissive, V(0, 0, 0), V(2, 2, 2)));
    m.sp.push_back(sph(V(0, -50, 0), 1, 0));
    m.sp.push_back(sph(V(0, 0, 0), 2, 1));
    m.vtx = {V(5, 0, 0), V(7, 0, 0), V(5, 3, 0)};
    m.tri.push_back(Triangle{0, 1, 2, 1});
    m.lights.push_back(Light{1, 4 * kPi * 4, {0, 0}});
    m.lights.push_back(Light{2, 3.0f, {0, 0}});
    SceneView v = m.view();
    Vec3 from = V(2, 1, 9);
    int sphere_seen = 0, tri_seen = 0;
    for (uint32_t i = 0; i < 4000; ++i) {
        LightSample ls = sample_light(v, from, make_path_key(8, i, 0), 0);
        if (ls.pdf_solid <= 0) continue;
        Vec3 d = ls.p - from;
        float t = length(d);
        bool on_tri = std::fabs(ls.p.z) < 1e-5f && ls.p.x > 4.9f;
        int prim = on_tri ? 2 : 1;
        (on_tri ? tri_seen : sphere_seen)++;
        float back = light_pdf_at_hit(v, prim, t, d / t, ls.n);
        CHECK_NEAR(back / ls.pdf_solid, 1.0, 2e-4);
    }
    CHECK(sphere_seen > 1000 && tri_seen > 1000);
}

TEST(sample_light_without_lights_returns_zero_pdf) {
    Mini m;
    m.mats.push_back(mat(kLambert, V(0.5f, 0.5f, 0.5f)));
    m.sp.push_back(sph(V(0, 0, 0), 1, 0));
    SceneView v = m.view();
    LightSample ls = sample_light(v, V(0, 0, 5), make_path_key(1, 1, 1), 0);
    CHECK(ls.pdf_solid == 0.0f);
}

// ------------------------------------------------------------------ classify / surface

TEST(classify_miss_adds_throughput_times_env_and_terminates) {
    Mini m;
    m.mats.push_back(mat(kLambert, V(0.5f, 0.5f, 0.5f)));
    m.sp.push_back(sph(V(0, 0, -50), 1, 0));
    m.env = V(1, 2, 3);
    SceneView v = m.view();
    PathState ps = fresh_path(make_path_key(1, 1, 1));
    ps.throughput = V(0.5f, 0.25f, 2.0f);
    ps.L = V(1, 1, 1);
    HitRecord miss{kInf, -1, 0, 0};
    Surface f{};
    CHECK(classify(v, ps, miss, 8, &f) == -1);
    // L += tp*env = (0.5, 0.5, 6)  -> (1.5, 1.5, 7)
    CHECK_NEAR(ps.L.x, 1.5, 1e-6); CHECK_NEAR(ps.L.y, 1.5, 1e-6); CHECK_NEAR(ps.L.z, 7.0, 1e-6);
    // zero env: nothing added
    m.env = V(0, 0, 0);
    SceneView v0 = m.view();
    PathState p2 = fresh_path(make_path_key(1, 1, 1));
    CHECK(classify(v0, p2, miss, 8, &f) == -1);
    CHECK(p2.L.x == 0.0f && p2.L.y == 0.0f && p2.L.z == 0.0f);
}

namespace {
// Sphere light radius 1 at the origin hit head-on from z=-5 (t=4); emission (2,4,6).
Mini emissive_target(int type = kEmissive, int extra_light = 0) {
    Mini m;
    m.mats.push_back(mat(type, V(0.5f, 0.5f, 0.5f), V(2, 4, 6)));
    m.sp.push_back(sph(V(0, 0, 0), 1, 0));
    m.lights.push_back(Light{0, 4 * kPi, {0, 0}});
    if (extra_light) m.lights.push_back(Light{0, 4 * kPi, {0, 0}});  // second entry halves selection probability
    m.single_leaf(2);
    return m;
}
PathState head_on(int depth, int specular, float prev_pdf) {
    PathState ps = fresh_path(make_path_key(1, 1, 1));
    ps.ray = Ray{V(0, 0, -5), kRayEps, V(0, 0, 1), kInf};
    ps.depth = depth; ps.specular = specular; ps.prev_pdf = prev_pdf;
    return ps;
}
}  // namespace

TEST(classify_emission_has_weight_one_at_depth_zero_and_after_specular_bounce) {
    Mini m = emissive_target(kLambert);   // a Lambertian that also emits (the furnace material)
    SceneView v = m.view();
    HitRecord h{4.0f, 0, 0, 0};
    Surface f;
    PathState a = head_on(0, 1, 0.0f);    // camera ray
    CHECK(classify(v, a, h, 8, &f) == kLambert);
    CHECK_NEAR(a.L.x, 2, 1e-6); CHECK_NEAR(a.L.y, 4, 1e-6); CHECK_NEAR(a.L.z, 6, 1e-6);   // weight exactly 1
    PathState b = head_on(2, 1, 0.0f);    // after a delta bounce, throughput 0.5
    b.throughput = V(0.5f, 0.5f, 0.5f);
    CHECK(classify(v, b, h, 8, &f) == kLambert);
    CHECK_NEAR(b.L.x, 1, 1e-6); CHECK_NEAR(b.L.y, 2, 1e-6); CHECK_NEAR(b.L.z, 3, 1e-6);
}

TEST(classify_emission_after_a_diffuse_bounce_is_mis_weighted_by_the_power_heuristic) {
    Mini m = emissive_target(kLambert);
    SceneView v = m.view();
    HitRecord h{4.0f, 0, 0, 0};
    Surface f;
    // Independent: BSDF pdf pb = 1/pi (cos=1); light pdf pl = t^2/(cos*A)/n_lights = 16/(4 pi).
    double pb = 1 / kPiD, pl = 16 / (4 * kPiD);
    double w = pb * pb / (pb * pb + pl * pl);
    PathState a = head_on(1, 0, (float)pb);
    CHECK(classify(v, a, h, 8, &f) == kLambert);
    CHECK_NEAR(a.L.x, 2 * w, 1e-6); CHECK_NEAR(a.L.z, 6 * w, 1e-6);
    CHECK(w > 0.01 && w < 0.99);  // not degenerate: the oracle can tell 1 from the heuristic
    // With two light entries the selection probability halves pl and raises the BSDF weight.
    Mini m2 = emissive_target(kLambert, 1);
    SceneView v2 = m2.view();
    double w2 = pb * pb / (pb * pb + (pl / 2) * (pl / 2));
    PathState b = head_on(1, 0, (float)pb);
    CHECK(classify(v2, b, h, 8, &f) == kLambert);
    CHECK_NEAR(b.L.x, 2 * w2, 1e-6);
    CHECK(w2 > w);
}

TEST(classify_return_codes_and_depth_limit) {
    Surface f;
    HitRecord h{4.0f, 0, 0, 0};
    // Scattering materials report their type; emission of a non-emitter adds nothing.
    for (int type : {kLambert, kMetal, kDielectric}) {
        Mini m;
        m.mats.push_back(mat(type, V(0.5f, 0.5f, 0.5f)));
        m.sp.push_back(sph(V(0, 0, 0), 1, 0));
        m.single_leaf(2);
        SceneView v = m.view();
        PathState ps = head_on(0, 1, 0);
        CHECK(classify(v, ps, h, 8, &f) == type);
        CHECK(ps.L.x == 0.0f && ps.L.y == 0.0f && ps.L.z == 0.0f);
    }
    // Pure emitter: emission added, path terminates.
    Mini e = emissive_target(kEmissive);
    SceneView ve = e.view();
    PathState pe = head_on(0, 1, 0);
    CHECK(classify(ve, pe, h, 8, &f) == -1);
    CHECK_NEAR(pe.L.x, 2, 1e-6);
    // Depth limit: depth == max_depth terminates (but still collects emission), depth == max_depth-1 continues.
    Mini l = emissive_target(kLambert);
    SceneView vl = l.view();
    PathState at = head_on(4, 1, 0), before = head_on(3, 1, 0);
    CHECK(classify(vl, before, h, 4, &f) == kLambert);
    CHECK(classify(vl, at, h, 4, &f) == -1);
    CHECK_NEAR(at.L.x, 2, 1e-6);
    PathState zero = head_on(0, 1, 0);
    CHECK(classify(vl, zero, h, 0, &f) == -1);   // max_depth 0: camera-visible emission only
    CHECK_NEAR(zero.L.x, 2, 1e-6);
}

TEST(make_surface_faces_against_the_ray_and_flags_inside_hits) {
    Mini m;
    m.mats.push_back(mat(kLambert, V(0.5f, 0.5f, 0.5f)));
    m.sp.push_back(sph(V(0, 0, 0), 5, 0));
    m.vtx = {V(10, 0, 0), V(11, 0, 0), V(10, 1, 0)};  // CCW seen from +z: winding normal +z
    m.tri.push_back(Triangle{0, 1, 2, 0});
    SceneView v = m.view();
    // Sphere, ray from outside: n = n_out radial.
    Ray out{V(0, 0, -8), kRayEps, V(0, 0, 1), kInf};
    Surface f = make_surface(v, out, HitRecord{3.0f, 0, 0, 0});
    CHECK_NEAR(f.p.z, -5, 1e-5); CHECK(f.inside == 0);
    CHECK_NEAR(f.n_out.z, -1, 1e-6); CHECK_NEAR(f.n.z, -1, 1e-6);
    // Sphere, ray from inside: n_out still radial outward (+z), n flips to face the ray.
    Ray in{V(0, 0, 0), kRayEps, V(0, 0, 1), kInf};
    f = make_surface(v, in, HitRecord{5.0f, 0, 0, 0});
    CHECK(f.inside == 1);
    CHECK_NEAR(f.n_out.z, 1, 1e-6); CHECK_NEAR(f.n.z, -1, 1e-6);
    // Triangle from the front and from behind.
    Ray above{V(10.2f, 0.2f, 3), kRayEps, V(0, 0, -1), kInf};
    f = make_surface(v, above, HitRecord{3.0f, 1, 0.2f, 0.2f});
    CHECK(f.inside == 0); CHECK_NEAR(f.n_out.z, 1, 1e-6); CHECK_NEAR(f.n.z, 1, 1e-6);
    CHECK_NEAR(f.p.z, 0, 1e-6); CHECK_NEAR(f.p.x, 10.2, 1e-5);
    Ray below{V(10.2f, 0.2f, -3), kRayEps, V(0, 0, 1), kInf};
    f = make_surface(v, below, HitRecord{3.0f, 1, 0.2f, 0.2f});
    CHECK(f.inside == 1); CHECK_NEAR(f.n_out.z, 1, 1e-6); CHECK_NEAR(f.n.z, -1, 1e-6);
}

// ------------------------------------------------------------------ shading

namespace {
Mini lambert_world(Vec3 albedo) {
    Mini m;
    m.mats.push_back(mat(kLambert, albedo));
    m.sp.push_back(sph(V(0, 0, -100), 1, 0));
    return m;
}
}  // namespace

TEST(lambert_scatter_is_a_unit_vector_in_the_upper_hemisphere_and_obeys_the_energy_bound) {
    Mini m = lambert_world(V(0.8f, 0.5f, 0.2f));
    SceneView v = m.view();
    Vec3 normals[] = {V(0, 0, 1), V(0, 0, -1), V(1, 0, 0), normalize(V(1, 2, 3)), normalize(V(-2, 0.1f, -1))};
    for (Vec3 n : normals) {
        Surface f = flat_surface(0);
        f.n = n; f.n_out = n; f.p = V(1, 2, 3);
        for (uint32_t i = 0; i < 20000; ++i) {
            PathState ps = fresh_path(make_path_key(9, i, 0));
            ps.ray.d = -n;
            Ray sh; Vec3 c; int has;
            bool alive = shade_lambert(v, ps, f, &sh, &c, &has);
            CHECK(alive);                                         // depth 0 -> 1: roulette never fires
            CHECK_NEAR(length(ps.ray.d), 1.0, 2e-6);
            CHECK(dot(ps.ray.d, n) > 0.0f);                       // strictly above the facing normal
            CHECK(max_component(ps.throughput) <= 0.8f + 1e-6f);  // <= max albedo: no energy created
            CHECK(ps.throughput.x == 0.8f && ps.throughput.y == 0.5f && ps.throughput.z == 0.2f);
            CHECK(ps.depth == 1); CHECK(ps.specular == 0);
            CHECK_NEAR(ps.prev_pdf, dot(ps.ray.d, n) / kPiD, 1e-6);   // cos/pi
            CHECK(has == 0);                                      // no lights in this scene
            // origin offset 1e-4 along the facing normal; tmin / tmax per spec
            CHECK_NEAR(ps.ray.o.x, f.p.x + n.x * 1e-4, 1e-6);
            CHECK_NEAR(ps.ray.o.y, f.p.y + n.y * 1e-4, 1e-6);
            CHECK_NEAR(ps.ray.o.z, f.p.z + n.z * 1e-4, 1e-6);
            CHECK(ps.ray.tmin == 1e-4f); CHECK(ps.ray.tmax >= 1e30f);
        }
    }
}

TEST(lambert_scatter_is_cosine_distributed) {
    // For cosine-weighted sampling: E[cos]=2/3, cos^2 ~ U[0,1), and the mean tangential component is 0.
    Mini m = lambert_world(V(1, 1, 1));
    SceneView v = m.view();
    Vec3 normals[] = {V(0, 0, 1), V(0, 0, -1), normalize(V(1, 2, 3))};
    for (Vec3 n : normals) {
        Surface f = flat_surface(0);
        f.n = n; f.n_out = n;
        Vec3 t = normalize(cross(n, std::fabs(n.x) < 0.9f ? V(1, 0, 0) : V(0, 1, 0)));  // independent tangent
        Acc cosm, tang;
        std::vector<double> c2;
        for (uint32_t i = 0; i < 160000; ++i) {
            PathState ps = fresh_path(make_path_key(10, i, 0));
            Ray sh; Vec3 c; int has;
            shade_lambert(v, ps, f, &sh, &c, &has);
            double cs = dot(ps.ray.d, n);
            cosm.add(cs); tang.add(dot(ps.ray.d, t));
            c2.push_back(cs * cs);
        }
        CHECK_NEAR(cosm.mean, 2.0 / 3, 3 * cosm.se());
        CHECK_NEAR(tang.mean, 0.0, 3 * tang.se());
        CHECK(chi2_16(c2) < 37.7);   // 15 dof, 99.9% quantile
    }
}

TEST(lambert_next_event_estimate_matches_independent_formula_and_skips_back_facing_lights) {
    // Light sphere radius 1, centre (0,0,5), Le=(10,5,2.5); receiver at the origin facing +z, albedo (0.8,0.5,0.2).
    Mini m;
    m.mats.push_back(mat(kLambert, V(0.8f, 0.5f, 0.2f)));
    m.sp.push_back(sph(V(0, 0, -100), 1, 0));
    add_sphere_light(m, V(0, 0, 5), 1.0f, V(10, 5, 2.5f));
    SceneView v = m.view();
    Surface f = flat_surface(0);
    for (uint32_t i = 0; i < 500; ++i) {
        PathState ps = fresh_path(make_path_key(11, i, 0));
        Ray sh; Vec3 c{}; int has = 0;
        shade_lambert(v, ps, f, &sh, &c, &has);
        CHECK(has == 1);   // the whole light sphere is above the receiver plane
        double dist = sh.tmax / (1 - 1e-3);                          // shadow ray stops 0.1% short of the light
        Vec3 q = sh.o + sh.d * (float)dist;                          // sampled light point
        CHECK_NEAR(len(sh.d.x, sh.d.y, sh.d.z), 1.0, 2e-6);
        CHECK(sh.o.z > 0 && std::fabs(sh.o.z - 1e-4) < 1e-6);
        CHECK(sh.tmin == 1e-4f);
        CHECK_NEAR(len(q.x, q.y, q.z - 5), 1.0, 2e-3);               // lies on the light
        double nlx = q.x, nly = q.y, nlz = q.z - 5;                  // radial normal (r=1)
        double cosl = std::fabs(nlx * sh.d.x + nly * sh.d.y + nlz * sh.d.z);
        double pdf = dist * dist / (cosl * 4 * kPiD);
        double cos_s = sh.d.z, pb = cos_s / kPiD;
        double w = pdf * pdf / (pdf * pdf + pb * pb);
        double k = cos_s / kPiD * w / pdf;
        if (cosl < 1e-2) continue;
        CHECK_NEAR(c.x / (0.8 * 10 * k), 1.0, 5e-3);
        CHECK_NEAR(c.y / (0.5 * 5 * k), 1.0, 5e-3);
        CHECK_NEAR(c.z / (0.2 * 2.5 * k), 1.0, 5e-3);
    }
    // Light entirely below the receiver plane: cosine at the receiver is <= 0, so no shadow ray, ever.
    Mini below;
    below.mats.push_back(mat(kLambert, V(0.8f, 0.5f, 0.2f)));
    below.sp.push_back(sph(V(0, 0, 100), 1, 0));
    add_sphere_light(below, V(0, 0, -5), 1.0f, V(10, 5, 2.5f));
    SceneView vb = below.view();
    int shadow_rays = 0;
    for (uint32_t i = 0; i < 500; ++i) {
        PathState ps = fresh_path(make_path_key(12, i, 0));
        Ray sh; Vec3 c{}; int has = 0;
        shade_lambert(vb, ps, f, &sh, &c, &has);
        shadow_rays += has;
        CHECK(ps.depth == 1);   // the BSDF scatter still happens
    }
    CHECK(shadow_rays == 0);
}

TEST(metal_reflects_about_the_facing_normal_and_tints_throughput) {
    Mini m;
    m.mats.push_back(mat(kMetal, V(0.9f, 0.8f, 0.7f)));
    m.sp.push_back(sph(V(0, -100, 0), 1, 0));
    SceneView v = m.view();
    // Hand case: d=(1,-1,0)/sqrt2 on the +y plane -> (1,1,0)/sqrt2.
    Surface f{}; f.p = V(1, 2, 3); f.n = V(0, 1, 0); f.n_out = V(0, 1, 0); f.material = 0;
    PathState ps = fresh_path(make_path_key(1, 1, 1));
    ps.ray.d = normalize(V(1, -1, 0));
    CHECK(shade_metal(v, ps, f));
    CHECK_NEAR(ps.ray.d.x, std::sqrt(0.5), 1e-6); CHECK_NEAR(ps.ray.d.y, std::sqrt(0.5), 1e-6); CHECK_NEAR(ps.ray.d.z, 0, 1e-7);
    CHECK_NEAR(ps.ray.o.y, 2.0001, 1e-6); CHECK_NEAR(ps.ray.o.x, 1, 1e-6); CHECK_NEAR(ps.ray.o.z, 3, 1e-6);
    CHECK_NEAR(ps.throughput.x, 0.9, 1e-7); CHECK_NEAR(ps.throughput.y, 0.8, 1e-7); CHECK_NEAR(ps.throughput.z, 0.7, 1e-7);
    CHECK(ps.specular == 1); CHECK(ps.prev_pdf == 0.0f); CHECK(ps.depth == 1);
    CHECK(ps.ray.tmin == 1e-4f);
    // Random geometry: out = d - 2 (d.n) n evaluated in double precision.
    for (uint32_t i = 0; i < 5000; ++i) {
        PathKey k = make_path_key(13, i, 0);
        Vec3 n = normalize(V(rng(k, 0, 0) - 0.5f, rng(k, 0, 1) - 0.5f, rng(k, 0, 2) - 0.5f));
        Vec3 d = normalize(V(rng(k, 1, 0) - 0.5f, rng(k, 1, 1) - 0.5f, rng(k, 1, 2) - 0.5f));
        if (dot(d, n) > 0) d = -d;
        if (dot(d, n) > -0.05f) continue;   // skip near-grazing
        Surface g{}; g.p = V(0, 0, 0); g.n = n; g.n_out = n; g.material = 0;
        PathState p = fresh_path(k); p.ray.d = d;
        CHECK(shade_metal(v, p, g));
        double dn = (double)d.x * n.x + (double)d.y * n.y + (double)d.z * n.z;
        CHECK_NEAR(p.ray.d.x, d.x - 2 * dn * n.x, 3e-6);
        CHECK_NEAR(p.ray.d.y, d.y - 2 * dn * n.y, 3e-6);
        CHECK_NEAR(p.ray.d.z, d.z - 2 * dn * n.z, 3e-6);
        CHECK(dot(p.ray.d, n) > 0);     // leaves on the facing side
    }
}

TEST(dielectric_refraction_obeys_snell_for_entering_and_exiting_rays) {
    Mini m;
    m.mats.push_back(mat(kDielectric, V(1, 1, 1), V(0, 0, 0), 1.5f));
    m.sp.push_back(sph(V(0, -100, 0), 1, 0));
    SceneView v = m.view();
    // Find keys forcing refraction by scanning for a large dielectric draw (u >= F is refracted).
    auto key_with = [&](double lo, double hi) {
        for (uint32_t i = 0; i < 1000000; ++i) {
            PathKey k = make_path_key(14, i, 0);
            double u = rng(k, 0, kDimDielectric);
            if (u >= lo && u < hi) return k;
        }
        return make_path_key(0, 0, 0);
    };
    // Entering: 45 deg in air, n=1.5. sin_t = sin_i / 1.5.
    {
        double si = std::sqrt(0.5), ci = std::sqrt(0.5);
        Surface f = flat_surface(0, false);
        PathState ps = fresh_path(key_with(0.5, 1.0));
        ps.ray.d = V((float)si, 0, (float)-ci);
        CHECK(shade_dielectric(v, ps, f));
        double st = std::hypot(ps.ray.d.x, ps.ray.d.y), ct = -ps.ray.d.z;
        CHECK_NEAR(st, si / 1.5, 1e-6);
        CHECK_NEAR(si / st, 1.5, 1e-4);                    // Snell: sin_i / sin_t = n_t / n_i
        CHECK_NEAR(ct, std::sqrt(1 - (si / 1.5) * (si / 1.5)), 1e-6);
        CHECK(ps.ray.d.x > 0 && std::fabs(ps.ray.d.y) < 1e-7);   // bends toward the normal, stays in the plane
        CHECK(ps.ray.o.z < 0);                             // origin pushed to the transmitted side
        CHECK_NEAR(ps.ray.o.z, -1e-4, 1e-7);
        CHECK(ps.specular == 1 && ps.prev_pdf == 0.0f && ps.depth == 1);
        CHECK_NEAR(length(ps.ray.d), 1.0, 2e-6);
    }
    // Exiting, below the critical angle: sin_i = 0.3 -> sin_t = 0.45.
    {
        double si = 0.3, ci = std::sqrt(1 - si * si);
        Surface f = flat_surface(0, true);
        PathState ps = fresh_path(key_with(0.5, 1.0));
        ps.ray.d = V((float)si, 0, (float)-ci);
        CHECK(shade_dielectric(v, ps, f));
        double st = std::hypot(ps.ray.d.x, ps.ray.d.y);
        CHECK_NEAR(st, 0.45, 1e-6);
        CHECK_NEAR(st / si, 1.5, 1e-4);                    // bends away from the normal
        CHECK(ps.ray.d.z < 0 && ps.ray.o.z < 0);
    }
    // Tint is applied to the throughput on both branches.
    Mini tinted;
    tinted.mats.push_back(mat(kDielectric, V(0.9f, 0.8f, 0.7f), V(0, 0, 0), 1.5f));
    tinted.sp.push_back(sph(V(0, -100, 0), 1, 0));
    SceneView vt = tinted.view();
    PathState pt = fresh_path(key_with(0.5, 1.0));
    pt.ray.d = V(0.3f, 0, -std::sqrt(0.91f));
    CHECK(shade_dielectric(vt, pt, flat_surface(0)));
    CHECK_NEAR(pt.throughput.x, 0.9, 1e-6); CHECK_NEAR(pt.throughput.z, 0.7, 1e-6);
}

TEST(dielectric_reflects_with_mirror_law_and_always_beyond_the_critical_angle) {
    Mini m;
    m.mats.push_back(mat(kDielectric, V(1, 1, 1), V(0, 0, 0), 1.5f));
    m.sp.push_back(sph(V(0, -100, 0), 1, 0));
    SceneView v = m.view();
    // Inside glass, sin_i = 0.8 > 2/3: Fresnel = 1, every key must reflect (u < 1 always).
    Surface f = flat_surface(0, true);
    for (uint32_t i = 0; i < 5000; ++i) {
        PathState ps = fresh_path(make_path_key(15, i, 0));
        ps.ray.d = V(0.8f, 0, -0.6f);
        CHECK(shade_dielectric(v, ps, f));
        CHECK_NEAR(ps.ray.d.x, 0.8, 1e-6); CHECK_NEAR(ps.ray.d.z, 0.6, 1e-6);   // mirror image about z=0
        CHECK(ps.ray.o.z > 0);                                                   // stays on the incident side
        CHECK(is_finite3(ps.ray.d));
    }
    // Reflection branch outside: choose a key with u < F/2 at 45 deg (F ~ 0.05).
    double F = fresnel_ref(std::sqrt(0.5), 1.0, 1.5);
    int found = 0;
    for (uint32_t i = 0; i < 200000 && found < 20; ++i) {
        PathKey k = make_path_key(16, i, 0);
        if (rng(k, 0, kDimDielectric) >= 0.5 * F) continue;
        ++found;
        PathState ps = fresh_path(k);
        ps.ray.d = V(std::sqrt(0.5f), 0, -std::sqrt(0.5f));
        CHECK(shade_dielectric(v, ps, flat_surface(0, false)));
        CHECK_NEAR(ps.ray.d.x, std::sqrt(0.5), 1e-6); CHECK_NEAR(ps.ray.d.z, std::sqrt(0.5), 1e-6);
        CHECK(ps.ray.o.z > 0);
    }
    CHECK(found == 20);
}

TEST(dielectric_reflection_probability_equals_fresnel_reflectance) {
    Mini m;
    m.mats.push_back(mat(kDielectric, V(1, 1, 1), V(0, 0, 0), 1.5f));
    m.sp.push_back(sph(V(0, -100, 0), 1, 0));
    SceneView v = m.view();
    // 60 deg incidence from air: independent F via the trigonometric formula (~0.0896).
    double F = fresnel_ref(0.5, 1.0, 1.5);
    const int N = 400000;
    int refl = 0;
    for (uint32_t i = 0; i < (uint32_t)N; ++i) {
        PathState ps = fresh_path(make_path_key(17, i, 0));
        ps.ray.d = V(std::sqrt(0.75f), 0, -0.5f);
        shade_dielectric(v, ps, flat_surface(0, false));
        refl += ps.ray.o.z > 0 ? 1 : 0;
    }
    CHECK_NEAR((double)refl / N, F, 3 * std::sqrt(F * (1 - F) / N));
    CHECK(F > 0.05 && F < 0.2);
}

// ------------------------------------------------------------------ russian roulette

TEST(russian_roulette_never_fires_before_depth_three) {
    for (int depth = 0; depth < 3; ++depth)
        for (uint32_t i = 0; i < 2000; ++i) {
            PathState ps = fresh_path(make_path_key(18, i, 0));
            ps.depth = depth;
            ps.throughput = V(0.001f, 0.001f, 0.001f);   // tiny weight would be killed immediately if roulette ran
            CHECK(russian_roulette(ps));
            CHECK(ps.throughput.x == 0.001f);            // and must not be rescaled either
        }
}

TEST(russian_roulette_survival_follows_the_clamped_max_component_and_rescales) {
    struct Case { Vec3 tp; double q; } cases[] = {
        {V(0.4f, 0.2f, 0.1f), 0.4},      // q = max component
        {V(0.1f, 0.9f, 0.1f), 0.9},      // max over channels, not luminance
        {V(0.01f, 0.0f, 0.0f), 0.05},    // lower clamp 0.05
        {V(2.0f, 1.0f, 1.0f), 0.95},     // upper clamp 0.95
    };
    const int N = 200000;
    for (auto& c : cases) {
        int alive = 0;
        for (uint32_t i = 0; i < (uint32_t)N; ++i) {
            PathState ps = fresh_path(make_path_key(19, i, 0));
            ps.depth = 3; ps.throughput = c.tp;
            double u = rng(ps.key, 3, kDimRoulette);   // spec: dimension kDimRoulette at the path's depth
            bool a = russian_roulette(ps);
            CHECK(a == (u < c.q));
            if (a) {
                ++alive;
                CHECK_NEAR(ps.throughput.x, c.tp.x / c.q, 1e-5 * (1 + c.tp.x / c.q));
                CHECK_NEAR(ps.throughput.y, c.tp.y / c.q, 1e-5 * (1 + c.tp.y / c.q));
            }
        }
        CHECK_NEAR((double)alive / N, c.q, 3 * std::sqrt(c.q * (1 - c.q) / N));
    }
}

TEST(russian_roulette_is_unbiased_for_the_expected_throughput) {
    // E[survive ? tp/q : 0] must equal tp. 1e6 trials, SE from the sample variance.
    struct Case { Vec3 tp; } cases[] = {{V(0.4f, 0.2f, 0.1f)}, {V(0.01f, 0.005f, 0.0f)}, {V(2.0f, 1.0f, 0.5f)}};
    for (auto& c : cases) {
        Acc a;
        for (uint32_t i = 0; i < 1000000; ++i) {
            PathState ps = fresh_path(make_path_key(20, i, 0));
            ps.depth = 4; ps.throughput = c.tp;
            a.add(russian_roulette(ps) ? ps.throughput.x : 0.0f);
        }
        CHECK_NEAR(a.mean, c.tp.x, 3 * a.se());
        CHECK(a.se() < 0.01 * c.tp.x + 1e-4);   // enough power to see a 1% bias
    }
}

// ------------------------------------------------------------------ whole-path oracles

namespace {

struct PathStats { Acc lum; Vec3 first{0, 0, 0}; bool all_equal = true; Acc r, g, b; };

// Mean radiance of N camera paths through the single-sphere world.
Acc trace_mean(const SceneView& v, int N, int max_depth, uint32_t seed, Acc* gch = nullptr, Acc* bch = nullptr) {
    Acc a;
    for (int i = 0; i < N; ++i) {
        Vec3 L = trace_path(v, i % v.width, (i / v.width) % v.height, (uint32_t)i, seed, max_depth);
        a.add(L.x);
        if (gch) gch->add(L.y);
        if (bch) bch->add(L.z);
    }
    return a;
}

// The furnace: one big emissive + Lambertian sphere, camera at its centre.
Mini furnace(float albedo, float Le) {
    Mini m;
    m.mats.push_back(mat(kLambert, V(albedo, albedo, albedo), V(Le, Le, Le)));
    m.sp.push_back(sph(V(0, 0, 0), 5, 0));
    m.lights.push_back(Light{0, 4.0f * kPi * 25.0f, {0, 0}});
    m.nodes = {BvhNode{V(-5.1f, -5.1f, -5.1f), -1, V(5.1f, 5.1f, 5.1f), 0, 1, {0, 0, 0}}};
    m.order = {0};
    m.cam = Camera{V(0, 0, 0), V(0, 0, -1), V(0, 1, 0), 60.0f};
    m.width = 16; m.height = 16;
    return m;
}

// A single sphere seen from outside, uniform environment radiance, no lights.
Mini outside_sphere(int type, Vec3 albedo, float ior, Vec3 env) {
    Mini m;
    m.mats.push_back(mat(type, albedo, V(0, 0, 0), ior));
    m.sp.push_back(sph(V(0, 0, 0), 1, 0));
    m.nodes = {BvhNode{V(-1.1f, -1.1f, -1.1f), -1, V(1.1f, 1.1f, 1.1f), 0, 1, {0, 0, 0}}};
    m.order = {0};
    m.env = env;
    m.cam = Camera{V(0, 0, -10), V(0, 0, 0), V(0, 1, 0), 6.0f};   // half-angle 3 deg: the pixel-square corners (4.3 deg) stay inside the silhouette (asin(1/10)=5.7 deg)
    m.width = 8; m.height = 8;
    return m;
}

}  // namespace

TEST(furnace_sphere_radiance_equals_Le_over_one_minus_rho) {
    // Closed Lambertian sphere, Le=1, rho=0.5, camera inside: L = Le/(1-rho) = 2 (max_depth 64 truncation is 0.5^64).
    Mini m = furnace(0.5f, 1.0f);
    SceneView v = m.view();
    Acc g, b;
    Acc a = trace_mean(v, 100000, 64, 7, &g, &b);
    CHECK_NEAR(a.mean, 2.0, 3 * a.se());
    CHECK_NEAR(g.mean, 2.0, 3 * g.se());
    CHECK_NEAR(b.mean, 2.0, 3 * b.se());
    CHECK(a.se() < 0.01);   // power: a 0.5% error in the mean is visible
    // A different albedo/emission pair: rho=0.8, Le=0.5 -> 2.5 (0.8^64 = 6e-7).
    Mini m2 = furnace(0.8f, 0.5f);
    SceneView v2 = m2.view();
    Acc a2 = trace_mean(v2, 100000, 64, 8);
    CHECK_NEAR(a2.mean, 2.5, 3 * a2.se());
    CHECK(a2.se() < 0.02);
}

TEST(furnace_sphere_truncated_at_max_depth_k_equals_Le_times_geometric_partial_sum) {
    // With max_depth = k the estimator collects bounces 0..k: E[L] = Le * sum_{j=0..k} rho^j.
    // k=0 -> 1 (exactly, for every path), k=1 -> 1.5, k=2 -> 1.75, k=3 (roulette active) -> 1.875, k=5 -> 1.96875.
    Mini m = furnace(0.5f, 1.0f);
    SceneView v = m.view();
    {
        Acc a = trace_mean(v, 2000, 0, 3);
        CHECK_NEAR(a.mean, 1.0, 1e-6);
        CHECK(a.var() < 1e-10);   // every path sees exactly the camera-visible emission
    }
    struct K { int k; double expect; } ks[] = {{1, 1.5}, {2, 1.75}, {3, 1.875}, {5, 1.96875}};
    for (auto& c : ks) {
        Acc a = trace_mean(v, 60000, c.k, 9 + c.k);
        if (std::fabs(a.mean - c.expect) > 3 * a.se() + 1e-5) std::printf("  depth %d mean %.5f expect %.5f se %.5f\n", c.k, a.mean, c.expect, a.se());
        CHECK_NEAR(a.mean, c.expect, 3 * a.se() + 1e-5);   // +1e-5: float rounding floor when the variance is ~0
        CHECK(a.se() < 0.01);
    }
}

TEST(furnace_sphere_with_black_walls_returns_exactly_the_emission) {
    // rho = 0: no scattered light at any depth -> every path returns Le exactly, whatever the estimator mix.
    Mini m = furnace(0.0f, 1.0f);
    SceneView v = m.view();
    for (int i = 0; i < 3000; ++i) {
        Vec3 L = trace_path(v, i % 16, (i / 16) % 16, (uint32_t)i, 5, 16);
        CHECK_NEAR(L.x, 1.0, 1e-6); CHECK_NEAR(L.y, 1.0, 1e-6); CHECK_NEAR(L.z, 1.0, 1e-6);
    }
}

TEST(escaping_paths_return_the_environment_radiance) {
    Mini m;   // no geometry at all: every camera ray escapes
    m.env = V(0.1f, 0.2f, 0.3f);
    m.width = 4; m.height = 4;
    SceneView v = m.view();
    Vec3 L = trace_path(v, 1, 2, 0, 1, 8);
    CHECK_NEAR(L.x, 0.1, 1e-7); CHECK_NEAR(L.y, 0.2, 1e-7); CHECK_NEAR(L.z, 0.3, 1e-7);
}

TEST(convex_diffuse_sphere_under_uniform_environment_reflects_albedo_times_env_exactly) {
    // Outside a convex object every scattered ray escapes: L = rho * env for each path that hit it.
    Mini m = outside_sphere(kLambert, V(0.5f, 0.4f, 0.3f), 1.5f, V(1, 1, 1));
    SceneView v = m.view();
    int bad = 0;
    for (int i = 0; i < 5000; ++i) {
        Vec3 L = trace_path(v, i % 8, (i / 8) % 8, (uint32_t)i, 3, 8);
        if (std::fabs(L.x - 0.5) > 1e-6 || std::fabs(L.y - 0.4) > 1e-6 || std::fabs(L.z - 0.3) > 1e-6) ++bad;
    }
    CHECK(bad == 0);
    // Depth limit: max_depth = 0 stops at the first hit before any scattering, so nothing is gathered.
    for (int i = 0; i < 200; ++i) {
        Vec3 L = trace_path(v, i % 8, (i / 8) % 8, (uint32_t)i, 3, 0);
        CHECK(L.x == 0.0f && L.y == 0.0f && L.z == 0.0f);
    }
}

TEST(convex_mirror_sphere_under_uniform_environment_returns_tint_times_env) {
    Mini m = outside_sphere(kMetal, V(0.9f, 0.8f, 0.7f), 1.5f, V(1, 2, 3));
    SceneView v = m.view();
    int bad = 0;
    for (int i = 0; i < 5000; ++i) {
        Vec3 L = trace_path(v, i % 8, (i / 8) % 8, (uint32_t)i, 4, 8);
        if (std::fabs(L.x - 0.9) > 1e-5 || std::fabs(L.y - 1.6) > 1e-5 || std::fabs(L.z - 2.1) > 1e-5) ++bad;
    }
    CHECK(bad == 0);
}

TEST(clear_glass_sphere_in_a_white_furnace_conserves_energy) {
    // Non-absorbing dielectric under constant env radiance 1: all light entering must leave, so E[L] = 1.
    // This exercises Fresnel selection, Snell refraction, total internal reflection and roulette together.
    Mini m = outside_sphere(kDielectric, V(1, 1, 1), 1.5f, V(1, 1, 1));
    SceneView v = m.view();
    Acc a = trace_mean(v, 60000, 64, 21);
    CHECK_NEAR(a.mean, 1.0, 3 * a.se());
    CHECK(a.se() < 0.01);
    // Same with a dense glass (n = 2.4): more total internal reflection, still conservative.
    Mini m2 = outside_sphere(kDielectric, V(1, 1, 1), 2.4f, V(1, 1, 1));
    SceneView v2 = m2.view();
    Acc a2 = trace_mean(v2, 60000, 64, 22);
    CHECK_NEAR(a2.mean, 1.0, 3 * a2.se());
    CHECK(a2.se() < 0.02);
}

TEST(trace_path_is_deterministic_and_independent_of_pixel_visit_order) {
    Mini m = furnace(0.5f, 1.0f);
    SceneView v = m.view();
    std::vector<Vec3> fwd, rev(256);
    for (int i = 0; i < 256; ++i) fwd.push_back(trace_path(v, i % 16, i / 16, 3, 11, 16));
    for (int i = 255; i >= 0; --i) rev[i] = trace_path(v, i % 16, i / 16, 3, 11, 16);
    int diff = 0;
    for (int i = 0; i < 256; ++i) diff += (fwd[i].x != rev[i].x || fwd[i].y != rev[i].y || fwd[i].z != rev[i].z);
    CHECK(diff == 0);
    // ...and a different seed changes (some) pixels, otherwise the seed is ignored.
    int changed = 0;
    for (int i = 0; i < 256; ++i) changed += trace_path(v, i % 16, i / 16, 3, 12, 16).x != fwd[i].x;
    CHECK(changed > 100);
}
