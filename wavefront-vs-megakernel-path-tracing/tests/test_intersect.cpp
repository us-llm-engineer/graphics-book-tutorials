// Tests for the header-only intersection code in include/wf/bvh.hpp:
// hit_sphere, hit_triangle, hit_box, intersect (skip-link traversal) and intersect_bruteforce.
// BVHs are built by hand so the skip links under test are written down independently of any builder.
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "test_framework.hpp"
#include "wf/bvh.hpp"
#include "wf/math.hpp"
#include "wf/scene.hpp"

using namespace wf;

namespace {

constexpr float kHuge = 3.402823466e+38f;

struct Rnd {
    std::mt19937_64 g;
    explicit Rnd(uint64_t s) : g(s) {}
    double u() { return (double)(g() >> 11) * (1.0 / 9007199254740992.0); }
    double range(double a, double b) { return a + (b - a) * u(); }
    Vec3 in_cube(double h) { return Vec3{(float)range(-h, h), (float)range(-h, h), (float)range(-h, h)}; }
    Vec3 unit() {
        double z = range(-1, 1), phi = range(0, 6.283185307179586), r = std::sqrt(1 - z * z);
        return Vec3{(float)(r * std::cos(phi)), (float)(r * std::sin(phi)), (float)z};
    }
};

Ray mk(Vec3 o, Vec3 d, float tmin = 1e-4f, float tmax = kHuge) { return Ray{o, tmin, d, tmax}; }
Vec3 V(float x, float y, float z) { return Vec3{x, y, z}; }
Sphere mksphere(Vec3 c, float r) { return Sphere{c, r, 0, {0, 0, 0}}; }
BvhNode leaf(Vec3 lo, Vec3 hi, int skip, int start, int count) { return BvhNode{lo, skip, hi, start, count, {0, 0, 0}}; }
BvhNode inner(Vec3 lo, Vec3 hi, int skip) { return BvhNode{lo, skip, hi, -1, 0, {0, 0, 0}}; }

// Owns the flat arrays and exposes a SceneView; only intersection-relevant fields are filled.
struct World {
    std::vector<Sphere> sp;
    std::vector<Vec3> vtx;
    std::vector<Triangle> tri;
    std::vector<BvhNode> nodes;
    std::vector<int> order;
    SceneView view() const {
        SceneView v{};
        v.spheres = sp.data(); v.vertices = vtx.data(); v.triangles = tri.data();
        v.nodes = nodes.data(); v.prim_order = order.data();
        v.n_spheres = (int)sp.size(); v.n_triangles = (int)tri.size(); v.n_nodes = (int)nodes.size();
        return v;
    }
    void add_tri(Vec3 a, Vec3 b, Vec3 c) {
        int i = (int)vtx.size();
        vtx.push_back(a); vtx.push_back(b); vtx.push_back(c);
        tri.push_back(Triangle{i, i + 1, i + 2, 0});
    }
    // bounds of primitive p (spheres first, then triangles), padded by `pad`
    void bounds(int p, Vec3* lo, Vec3* hi, float pad = 1e-3f) const {
        if (p < (int)sp.size()) {
            const Sphere& s = sp[p];
            *lo = s.center - V(s.radius + pad, s.radius + pad, s.radius + pad);
            *hi = s.center + V(s.radius + pad, s.radius + pad, s.radius + pad);
        } else {
            const Triangle& t = tri[p - sp.size()];
            *lo = vmin(vtx[t.i0], vmin(vtx[t.i1], vtx[t.i2])) - V(pad, pad, pad);
            *hi = vmax(vtx[t.i0], vmax(vtx[t.i1], vtx[t.i2])) + V(pad, pad, pad);
        }
    }
    float centroid_x(int p) const {
        if (p < (int)sp.size()) return sp[p].center.x;
        const Triangle& t = tri[p - sp.size()];
        return (vtx[t.i0].x + vtx[t.i1].x + vtx[t.i2].x) / 3.0f;
    }
    void bounds_of(const std::vector<int>& ids, Vec3* lo, Vec3* hi) const {
        *lo = V(kHuge, kHuge, kHuge); *hi = V(-kHuge, -kHuge, -kHuge);
        for (int p : ids) { Vec3 l, h; bounds(p, &l, &h); *lo = vmin(*lo, l); *hi = vmax(*hi, h); }
    }
};

World random_world(Rnd& rnd, int ns, int nt) {
    World w;
    for (int i = 0; i < ns; ++i) w.sp.push_back(mksphere(rnd.in_cube(3), (float)rnd.range(0.3, 1.0)));
    for (int i = 0; i < nt; ++i) w.add_tri(rnd.in_cube(3), rnd.in_cube(3), rnd.in_cube(3));
    return w;
}

// Single-leaf BVH over all primitives, identity prim_order.
void make_single_leaf(World& w) {
    int n = (int)(w.sp.size() + w.tri.size());
    w.order.clear();
    for (int i = 0; i < n; ++i) w.order.push_back(i);
    w.nodes = {leaf(V(-5.1f, -5.1f, -5.1f), V(5.1f, 5.1f, 5.1f), -1, 0, n)};
}

// Primitives sorted by centroid x and split into groups; returns ids per group (prim_order = concatenation).
std::vector<std::vector<int>> split_by_x(World& w, int groups) {
    int n = (int)(w.sp.size() + w.tri.size());
    std::vector<int> ids(n);
    for (int i = 0; i < n; ++i) ids[i] = i;
    std::sort(ids.begin(), ids.end(), [&](int a, int b) { return w.centroid_x(a) < w.centroid_x(b); });
    std::vector<std::vector<int>> out(groups);
    for (int i = 0; i < n; ++i) out[i * groups / n].push_back(ids[i]);
    w.order.clear();
    for (auto& g : out) for (int p : g) w.order.push_back(p);
    return out;
}

// Preorder tree: 0 = root, 1 = left leaf, 2 = right leaf; skip(1)=2, skip(2)=-1, skip(0)=-1.
void make_three_node(World& w) {
    auto g = split_by_x(w, 2);
    Vec3 llo, lhi, rlo, rhi, alo, ahi;
    w.bounds_of(g[0], &llo, &lhi); w.bounds_of(g[1], &rlo, &rhi); w.bounds_of(w.order, &alo, &ahi);
    w.nodes = {inner(alo, ahi, -1),
               leaf(llo, lhi, 2, 0, (int)g[0].size()),
               leaf(rlo, rhi, -1, (int)g[0].size(), (int)g[1].size())};
}

// Preorder 5-node tree: 0 root; 1 inner(left) with leaves 2,3; 4 right leaf.
// Skips: node1 -> 4 (after its subtree), node2 -> 3, node3 -> 4, node4 -> -1, root -> -1.
void make_five_node(World& w) {
    auto g = split_by_x(w, 3);
    Vec3 lo[3], hi[3], l01, h01, la, ha;
    for (int i = 0; i < 3; ++i) w.bounds_of(g[i], &lo[i], &hi[i]);
    std::vector<int> g01 = g[0]; g01.insert(g01.end(), g[1].begin(), g[1].end());
    w.bounds_of(g01, &l01, &h01); w.bounds_of(w.order, &la, &ha);
    int s0 = 0, s1 = (int)g[0].size(), s2 = s1 + (int)g[1].size();
    w.nodes = {inner(la, ha, -1), inner(l01, h01, 4),
               leaf(lo[0], hi[0], 3, s0, (int)g[0].size()),
               leaf(lo[1], hi[1], 4, s1, (int)g[1].size()),
               leaf(lo[2], hi[2], -1, s2, (int)g[2].size())};
}

Ray random_ray(Rnd& rnd) {
    Vec3 o = rnd.in_cube(6);
    Vec3 d = rnd.unit();
    int mode = (int)(rnd.u() * 10);
    if (mode == 0) d = V(rnd.u() < 0.5 ? 1.f : -1.f, 0, 0);       // axis-parallel: zero components -> inf slabs
    else if (mode == 1) d = V(0, rnd.u() < 0.5 ? 1.f : -1.f, 0);
    else if (mode == 2) d = V(0, 0, rnd.u() < 0.5 ? 1.f : -1.f);
    else if (mode == 3) d = V(d.x, d.y, 0);                        // one zero component
    if (mode == 3 && length2(d) > 0) d = normalize(d);
    else if (mode >= 5) { Vec3 target = rnd.in_cube(3); d = normalize(target - o); }  // aim into the scene
    if (length2(d) == 0) d = V(1, 0, 0);
    float tmax = rnd.u() < 0.3 ? (float)rnd.range(0.5, 12.0) : kHuge;
    return mk(o, d, 1e-4f, tmax);
}

// Compare traversal vs oracle on many rays; returns counts for vacuity checks.
struct Agree { int rays = 0, hits = 0, sphere_hits = 0, tri_hits = 0, mismatches = 0; };
Agree compare_on_random_rays(const SceneView& v, Rnd& rnd, int n) {
    Agree a;
    for (int i = 0; i < n; ++i) {
        Ray r = random_ray(rnd);
        HitRecord h1, h2;
        bool b1 = intersect(v, r, &h1), b2 = intersect_bruteforce(v, r, &h2);
        ++a.rays;
        bool ok = b1 == b2 && h1.prim == h2.prim;
        if (ok && b1) {
            ok = std::fabs(h1.t - h2.t) <= 1e-5f * std::fabs(h2.t) && std::fabs(h1.u - h2.u) <= 1e-5f &&
                 std::fabs(h1.v - h2.v) <= 1e-5f;
        }
        if (!ok) {
            if (a.mismatches < 3) std::printf("  mismatch ray %d: bvh prim=%d t=%g  brute prim=%d t=%g\n", i, h1.prim, h1.t, h2.prim, h2.t);
            ++a.mismatches;
        }
        if (b2) { ++a.hits; (h2.prim < v.n_spheres ? a.sphere_hits : a.tri_hits)++; }
    }
    return a;
}

// Independent double-precision nearest sphere root in (tmin, tmax); returns <0 if none.
double ref_sphere(Vec3 c, double rad, const Ray& r) {
    double ox = (double)r.o.x - c.x, oy = (double)r.o.y - c.y, oz = (double)r.o.z - c.z;
    double dx = r.d.x, dy = r.d.y, dz = r.d.z;
    double a = dx * dx + dy * dy + dz * dz, b = ox * dx + oy * dy + oz * dz, cc = ox * ox + oy * oy + oz * oz - rad * rad;
    double disc = b * b - a * cc;
    if (disc < 0) return -1;
    double s = std::sqrt(disc), t0 = (-b - s) / a, t1 = (-b + s) / a;
    if (t0 > r.tmin && t0 < r.tmax) return t0;
    if (t1 > r.tmin && t1 < r.tmax) return t1;
    return -1;
}

}  // namespace

// ------------------------------------------------------------------ sphere

TEST(sphere_ray_from_outside_hits_near_surface_at_analytic_distance) {
    // Unit sphere at origin, ray from z=-5 along +z: roots 4 and 6; nearest is 4.
    Sphere s = mksphere(V(0, 0, 0), 1);
    float t = 0;
    CHECK(hit_sphere(s, mk(V(0, 0, -5), V(0, 0, 1)), &t));
    CHECK_NEAR(t, 4.0, 1e-5);
    // Non-unit direction: t is measured in units of d, so |d|=2 halves it (catches a missing a=|d|^2).
    CHECK(hit_sphere(s, mk(V(0, 0, -5), V(0, 0, 2)), &t));
    CHECK_NEAR(t, 2.0, 1e-5);
    // Off-axis: o=(0.6,0,-5): z = -sqrt(1-0.36) = -0.8 -> t = 4.2
    CHECK(hit_sphere(s, mk(V(0.6f, 0, -5), V(0, 0, 1)), &t));
    CHECK_NEAR(t, 4.2, 1e-5);
    // Sphere not at origin: center (1,2,3), radius 2, ray along +x at y=2,z=3 from x=-10 -> near x=-1 -> t=9
    CHECK(hit_sphere(mksphere(V(1, 2, 3), 2), mk(V(-10, 2, 3), V(1, 0, 0)), &t));
    CHECK_NEAR(t, 9.0, 1e-5);
}

TEST(sphere_ray_from_inside_returns_far_root) {
    Sphere s = mksphere(V(0, 0, 0), 1);
    float t = 0;
    CHECK(hit_sphere(s, mk(V(0, 0, 0), V(0, 0, 1)), &t));  // roots -1 and +1
    CHECK_NEAR(t, 1.0, 1e-6);
    CHECK(hit_sphere(s, mk(V(0, 0, 0.5f), V(0, 0, 1)), &t));  // +0.5
    CHECK_NEAR(t, 0.5, 1e-6);
    CHECK(hit_sphere(s, mk(V(0, 0, 0.5f), V(0, 0, -1)), &t));  // +1.5 on the far side
    CHECK_NEAR(t, 1.5, 1e-6);
}

TEST(sphere_ray_pointing_away_or_past_does_not_hit) {
    Sphere s = mksphere(V(0, 0, 0), 1);
    float t = -123;
    CHECK(!hit_sphere(s, mk(V(0, 0, 5), V(0, 0, 1)), &t));     // sphere behind: roots -6, -4
    CHECK(!hit_sphere(s, mk(V(0, 0, -5), V(0, 0, -1)), &t));
    CHECK(!hit_sphere(s, mk(V(2, 0, -5), V(0, 0, 1)), &t));    // passes 1 unit outside
    CHECK(!hit_sphere(s, mk(V(0, 1.001f, -5), V(0, 0, 1)), &t));
    CHECK(t == -123.0f);                                        // a miss must not write the output
}

TEST(sphere_tangent_ray_is_hit_or_missed_consistently_with_a_nearby_pair) {
    Sphere s = mksphere(V(0, 0, 0), 1);
    float t = 0;
    // Just inside the silhouette: o.x = 0.999 -> z = -sqrt(1-0.998001) = -0.044710... -> t = 5 - 0.0447098
    CHECK(hit_sphere(s, mk(V(0.999f, 0, -5), V(0, 0, 1)), &t));
    CHECK_NEAR(t, 5.0 - std::sqrt(1 - 0.999 * 0.999), 1e-4);
    // Just outside: miss.
    CHECK(!hit_sphere(s, mk(V(1.001f, 0, -5), V(0, 0, 1)), &t));
    // Exactly tangent is measure zero: either answer is acceptable but a hit must be at the touch point t=5.
    float tt = 0;
    if (hit_sphere(s, mk(V(1, 0, -5), V(0, 0, 1)), &tt)) CHECK_NEAR(tt, 5.0, 1e-3);
}

TEST(sphere_tmin_and_tmax_clip_the_open_interval) {
    Sphere s = mksphere(V(0, 0, 0), 1);
    Vec3 o = V(0, 0, -5), d = V(0, 0, 1);  // roots exactly 4 and 6
    float t = 0;
    CHECK(hit_sphere(s, mk(o, d, 0.0f, 10.0f), &t)); CHECK_NEAR(t, 4.0, 1e-5);
    CHECK(hit_sphere(s, mk(o, d, 5.0f, 10.0f), &t)); CHECK_NEAR(t, 6.0, 1e-5);   // near clipped by tmin -> far root
    CHECK(hit_sphere(s, mk(o, d, 0.0f, 5.0f), &t));  CHECK_NEAR(t, 4.0, 1e-5);   // far clipped by tmax -> near root
    CHECK(!hit_sphere(s, mk(o, d, 4.5f, 5.5f), &t));                             // both roots outside
    CHECK(!hit_sphere(s, mk(o, d, 0.0f, 3.9f), &t));                             // tmax before entry
    CHECK(!hit_sphere(s, mk(o, d, 6.5f, 100.0f), &t));                           // tmin after exit
    // Open interval: root exactly at tmax / tmin is excluded (header: "(tmin, tmax)").
    CHECK(!hit_sphere(s, mk(o, d, 0.0f, 4.0f), &t));
    CHECK(!hit_sphere(s, mk(o, d, 6.0f, 100.0f), &t));
    CHECK(hit_sphere(s, mk(o, d, 0.0f, 4.001f), &t)); CHECK_NEAR(t, 4.0, 1e-5);
}

TEST(sphere_ray_starting_on_surface_ignores_its_own_origin_hit) {
    Sphere s = mksphere(V(0, 0, 0), 1);
    float t = 0;
    // From (0,0,-1) heading inward: roots are 0 (own origin) and 2. tmin=1e-4 excludes 0.
    CHECK(hit_sphere(s, mk(V(0, 0, -1), V(0, 0, 1)), &t));
    CHECK_NEAR(t, 2.0, 1e-5);
    // From the same point heading outward: roots 0 and -2 -> no hit at all.
    CHECK(!hit_sphere(s, mk(V(0, 0, -1), V(0, 0, -1)), &t));
    // Offset 1e-4 outside along the normal (what the estimator does) pointing outward: miss.
    CHECK(!hit_sphere(s, mk(V(0, 0, -1.0001f), V(0, 0, -1)), &t));
    // ... and offset inside pointing inward through the sphere: hit on the far side at ~1.9999.
    CHECK(hit_sphere(s, mk(V(0, 0, -0.9999f), V(0, 0, 1)), &t));
    CHECK_NEAR(t, 1.9999, 1e-4);
}

TEST(sphere_hit_agrees_with_double_precision_reference_on_random_rays) {
    Rnd rnd(10);
    int hits = 0, bad = 0;
    for (int i = 0; i < 100000; ++i) {
        Vec3 c = rnd.in_cube(2);
        float rad = (float)rnd.range(0.1, 2.0);
        Vec3 o = rnd.in_cube(6);
        Vec3 aim = (c + rnd.in_cube(rad * 1.5)) - o;  // aim near the sphere so hits are common
        Vec3 d = normalize(aim) * (float)rnd.range(0.5, 3.0);
        Ray r = mk(o, d, (float)rnd.range(0, 1), rnd.u() < 0.5 ? kHuge : (float)rnd.range(1, 15));
        double ref = ref_sphere(c, rad, r);
        float t = -1;
        bool h = hit_sphere(mksphere(c, rad), r, &t);
        // away from the interval and tangent borders the answers must agree exactly in hit/miss
        bool borderline = ref < 0 ? false : (std::fabs(ref - r.tmin) < 1e-3 || std::fabs(ref - r.tmax) < 1e-3);
        if (borderline) continue;
        bool ok = (h == (ref >= 0)) && (!h || std::fabs(t - ref) <= 1e-4 * (1 + ref));
        if (h) ++hits;
        if (!ok) ++bad;
    }
    CHECK(bad == 0);
    CHECK(hits > 10000);  // the comparison was not vacuous
}

// ------------------------------------------------------------------ triangle

namespace {
// Triangle in plane z=0: p0=(0,0,0) p1=(1,0,0) p2=(0,1,0). Ray from (x,y,1) along -z has t=1, u=x, v=y.
bool tri_at(float x, float y, float* t = nullptr, float* u = nullptr, float* v = nullptr, float dz = -1, float oz = 1) {
    float tt, uu, vv;
    bool h = hit_triangle(V(0, 0, 0), V(1, 0, 0), V(0, 1, 0), mk(V(x, y, oz), V(0, 0, dz)), &tt, &uu, &vv);
    if (h) { if (t) *t = tt; if (u) *u = uu; if (v) *v = vv; }
    return h;
}
}  // namespace

TEST(triangle_centre_hit_gives_t_and_barycentrics) {
    float t, u, v;
    CHECK(tri_at(1.0f / 3, 1.0f / 3, &t, &u, &v));
    CHECK_NEAR(t, 1.0, 1e-6); CHECK_NEAR(u, 1.0 / 3, 1e-6); CHECK_NEAR(v, 1.0 / 3, 1e-6);
    // Asymmetric point so swapped u/v is detectable: (0.25, 0.5) -> u=0.25 (weight of p1), v=0.5 (weight of p2)
    CHECK(tri_at(0.25f, 0.5f, &t, &u, &v));
    CHECK_NEAR(u, 0.25, 1e-6); CHECK_NEAR(v, 0.5, 1e-6);
    // Non-unit direction: |d|=2 halves t
    CHECK(tri_at(0.25f, 0.25f, &t, nullptr, nullptr, -2.0f)); CHECK_NEAR(t, 0.5, 1e-6);
}

TEST(triangle_barycentrics_for_oblique_ray_in_3d_match_construction) {
    // Triangle (0,0,2),(4,0,2),(0,4,2); P = p0 + 0.25*(p1-p0) + 0.5*(p2-p0) = (1,2,2).
    // Ray direction (0.5,0,1) reaches P at t=5 from o = P - 5d = (-1.5,2,-3).
    float t, u, v;
    CHECK(hit_triangle(V(0, 0, 2), V(4, 0, 2), V(0, 4, 2), mk(V(-1.5f, 2, -3), V(0.5f, 0, 1)), &t, &u, &v));
    CHECK_NEAR(t, 5.0, 1e-5); CHECK_NEAR(u, 0.25, 1e-6); CHECK_NEAR(v, 0.5, 1e-6);
}

TEST(triangle_constructed_points_return_their_barycentrics_on_random_triangles) {
    // Constructive oracle: choose barycentrics (b1,b2), form P, shoot a ray at P from a random origin.
    Rnd rnd(11);
    int tested = 0, bad = 0;
    for (int i = 0; i < 20000; ++i) {
        Vec3 p0 = rnd.in_cube(3), p1 = rnd.in_cube(3), p2 = rnd.in_cube(3);
        Vec3 n = cross(p1 - p0, p2 - p0);
        double area2 = length(n);
        if (area2 < 0.5) continue;
        n = n / (float)area2;
        double b1 = rnd.range(0.02, 0.96), b2 = rnd.range(0.02, 0.98 - b1);
        Vec3 P = p0 + (p1 - p0) * (float)b1 + (p2 - p0) * (float)b2;
        Vec3 o = P + rnd.unit() * (float)rnd.range(1, 6);
        Vec3 d = P - o;  // |d| = distance, so t should be 1
        if (std::fabs(dot(normalize(d), n)) < 0.2) continue;  // avoid near-parallel conditioning
        float t, u, v;
        bool h = hit_triangle(p0, p1, p2, mk(o, d, 1e-4f, 10.0f), &t, &u, &v);
        ++tested;
        if (!(h && std::fabs(t - 1) < 2e-3 && std::fabs(u - b1) < 2e-3 && std::fabs(v - b2) < 2e-3)) ++bad;
    }
    CHECK(tested > 5000);
    CHECK(bad == 0);
}

TEST(triangle_points_just_outside_each_edge_miss) {
    CHECK(!tri_at(0.6f, 0.6f));      // beyond hypotenuse (u+v>1)
    CHECK(!tri_at(-0.01f, 0.5f));    // left of edge p0-p2 (u<0)
    CHECK(!tri_at(0.5f, -0.01f));    // below edge p0-p1 (v<0)
    CHECK(!tri_at(1.01f, 0.0f));     // beyond vertex p1
    CHECK(!tri_at(-0.01f, -0.01f));  // beyond vertex p0
    CHECK(!tri_at(0.0f, 1.01f));     // beyond vertex p2
}

TEST(triangle_edges_and_vertices_are_watertight_across_a_shared_edge) {
    // Two triangles share the hypotenuse (1,0,0)-(0,1,0). A ray exactly on the shared edge must hit
    // at least one of them (no crack), and a ray just to one side must hit only that side.
    Vec3 a0 = V(0, 0, 0), a1 = V(1, 0, 0), a2 = V(0, 1, 0);
    Vec3 b0 = V(1, 0, 0), b1 = V(1, 1, 0), b2 = V(0, 1, 0);
    float t, u, v;
    int cracks = 0;
    for (float s = 0.0f; s <= 1.0f; s += 0.125f) {  // points (s,1-s) along the edge, exactly representable
        Ray r = mk(V(s, 1 - s, 1), V(0, 0, -1));
        bool ha = hit_triangle(a0, a1, a2, r, &t, &u, &v), hb = hit_triangle(b0, b1, b2, r, &t, &u, &v);
        if (!ha && !hb) ++cracks;
    }
    CHECK(cracks == 0);
    Ray left = mk(V(0.49f, 0.49f, 1), V(0, 0, -1)), right = mk(V(0.51f, 0.51f, 1), V(0, 0, -1));
    CHECK(hit_triangle(a0, a1, a2, left, &t, &u, &v));
    CHECK(!hit_triangle(b0, b1, b2, left, &t, &u, &v));
    CHECK(!hit_triangle(a0, a1, a2, right, &t, &u, &v));
    CHECK(hit_triangle(b0, b1, b2, right, &t, &u, &v));
}

TEST(triangle_vertices_are_hit) {
    // A vertex is on three edges; none of the three corner rays may slip through.
    CHECK(tri_at(0, 0)); CHECK(tri_at(1, 0)); CHECK(tri_at(0, 1));
}

TEST(triangle_parallel_rays_miss) {
    float t = -9, u, v;
    Vec3 p0 = V(0, 0, 0), p1 = V(1, 0, 0), p2 = V(0, 1, 0);
    CHECK(!hit_triangle(p0, p1, p2, mk(V(-1, 0.25f, 1), V(1, 0, 0)), &t, &u, &v));  // parallel, above plane
    CHECK(!hit_triangle(p0, p1, p2, mk(V(-1, 0.25f, 0), V(1, 0, 0)), &t, &u, &v));  // in-plane ray
    CHECK(!hit_triangle(p0, p1, p2, mk(V(0.2f, 0.2f, 1), V(1, 1, 0)), &t, &u, &v));
    CHECK(t == -9.0f);
}

TEST(triangle_is_hit_from_both_sides_with_same_t_and_barycentrics) {
    // Dielectrics need to be crossed in both directions, so back faces must not be culled.
    float tf, uf, vf, tb, ub, vb;
    CHECK(tri_at(0.25f, 0.5f, &tf, &uf, &vf, -1, 1));   // from +z (front, winding normal +z)
    CHECK(tri_at(0.25f, 0.5f, &tb, &ub, &vb, +1, -1));  // from -z (back face)
    CHECK_NEAR(tf, tb, 1e-6); CHECK_NEAR(uf, ub, 1e-6); CHECK_NEAR(vf, vb, 1e-6);
}

TEST(triangle_behind_the_origin_and_outside_tmin_tmax_miss) {
    Vec3 p0 = V(0, 0, 0), p1 = V(1, 0, 0), p2 = V(0, 1, 0);
    float t, u, v;
    CHECK(!hit_triangle(p0, p1, p2, mk(V(0.25f, 0.25f, -1), V(0, 0, -1)), &t, &u, &v));  // plane at t=-1
    // plane at t=1: open interval (tmin,tmax)
    Vec3 o = V(0.25f, 0.25f, 1), d = V(0, 0, -1);
    CHECK(hit_triangle(p0, p1, p2, mk(o, d, 0.99f, 1.01f), &t, &u, &v));
    CHECK(!hit_triangle(p0, p1, p2, mk(o, d, 1.0f, 2.0f), &t, &u, &v));
    CHECK(!hit_triangle(p0, p1, p2, mk(o, d, 0.0f, 1.0f), &t, &u, &v));
    CHECK(!hit_triangle(p0, p1, p2, mk(o, d, 1.5f, 3.0f), &t, &u, &v));
    CHECK(!hit_triangle(p0, p1, p2, mk(o, d, 0.0f, 0.5f), &t, &u, &v));
}

// ------------------------------------------------------------------ slab box

namespace {
Vec3 inv_of(Vec3 d) { return V(1.0f / d.x, 1.0f / d.y, 1.0f / d.z); }
bool box(Vec3 o, Vec3 d, float tmin = 1e-4f, float tmax = kHuge) {
    return hit_box(V(-1, -1, -1), V(1, 1, 1), o, inv_of(d), tmin, tmax);
}
}  // namespace

TEST(box_axis_parallel_ray_with_zero_components_uses_infinite_slabs) {
    CHECK(box(V(0, 0, -5), V(0, 0, 1)));          // d.x = d.y = 0 -> inv = +inf, origin inside both slabs
    CHECK(box(V(0.9f, -0.9f, 5), V(0, 0, -1)));
    CHECK(box(V(-5, 0.5f, 0.5f), V(1, 0, 0)));
    CHECK(!box(V(2, 0, -5), V(0, 0, 1)));          // zero x-direction but origin outside the x slab
    CHECK(!box(V(-2, 0, -5), V(0, 0, 1)));
    CHECK(!box(V(0, 1.5f, -5), V(0, 0, 1)));
    CHECK(!box(V(2, 0, -5), V(-0.0f, 0, 1)));       // negative zero -> inv = -inf, still a miss
    CHECK(!box(V(0, 0, 5), V(0, 0, 1)));            // pointing away
}

TEST(box_ray_starting_inside_hits_for_every_direction) {
    Rnd rnd(12);
    for (int i = 0; i < 2000; ++i) CHECK(box(V((float)rnd.range(-0.9, 0.9), (float)rnd.range(-0.9, 0.9), (float)rnd.range(-0.9, 0.9)), rnd.unit()));
    CHECK(box(V(0, 0, 0), V(1, 0, 0), 1e-4f, 1e-3f));  // inside: even a tiny tmax still overlaps the box
}

TEST(box_tmin_tmax_window_must_overlap_the_slab_interval) {
    Vec3 o = V(0, 0, -5), d = V(0, 0, 1);  // inside the box for t in [4,6]
    CHECK(box(o, d, 0.0f, 10.0f));
    CHECK(!box(o, d, 0.0f, 3.0f));      // window ends before entry
    CHECK(box(o, d, 0.0f, 4.5f));       // window ends inside the box
    CHECK(!box(o, d, 7.0f, 20.0f));     // window starts after exit
    CHECK(box(o, d, 5.5f, 20.0f));      // window starts inside the box
}

TEST(box_grazing_rays_are_never_falsely_culled) {
    // A BVH may report a false positive but must never report a false negative at a boundary.
    CHECK(box(V(1, 0, -5), V(0, 0, 1)));      // slides exactly along the x=+1 face
    CHECK(box(V(-1, 0, -5), V(0, 0, 1)));     // along x=-1 face (0*inf = NaN in the slab)
    CHECK(box(V(3, -1, 0), V(-1, 1, 0)));     // passes exactly through the edge x=1,y=1 at t=2
    CHECK(!box(V(1.001f, 0, -5), V(0, 0, 1)));  // just outside, parallel
    CHECK(box(V(0.999f, 0, -5), V(0, 0, 1)));   // just inside, parallel
    CHECK(!box(V(3, -0.99f, 0), V(-1, 1, 0)));  // passes the corner at y=1.01: miss
}

TEST(box_agrees_with_double_precision_slab_on_random_rays) {
    Rnd rnd(13);
    int hits = 0, misses = 0, bad = 0;
    for (int i = 0; i < 100000; ++i) {
        Vec3 o = rnd.in_cube(3), d = rnd.unit();
        double t0 = 1e-4, t1 = 1e30;
        double oo[3] = {o.x, o.y, o.z}, dd[3] = {d.x, d.y, d.z};
        for (int k = 0; k < 3; ++k) {
            double a = (-1 - oo[k]) / dd[k], b = (1 - oo[k]) / dd[k];
            if (a > b) std::swap(a, b);
            t0 = std::max(t0, a); t1 = std::min(t1, b);
        }
        double margin = t1 - t0;
        if (std::fabs(margin) < 1e-3) continue;  // borderline
        bool h = box(o, d);
        if (h != (margin > 0)) ++bad;
        (h ? hits : misses)++;
    }
    CHECK(bad == 0);
    CHECK(hits > 5000 && misses > 5000);
}

// ------------------------------------------------------------------ traversal vs oracle

TEST(single_leaf_traversal_matches_bruteforce_on_20000_random_rays) {
    Rnd rnd(14);
    World w = random_world(rnd, 6, 8);
    make_single_leaf(w);
    Agree a = compare_on_random_rays(w.view(), rnd, 20000);
    CHECK(a.mismatches == 0);
    // Non-vacuity: the rays must actually hit both primitive kinds a lot, and also miss a lot.
    CHECK(a.hits > 2000);
    CHECK(a.sphere_hits > 500);
    CHECK(a.tri_hits > 300);
    CHECK(a.rays - a.hits > 2000);
}

TEST(three_node_preorder_bvh_matches_bruteforce_on_20000_random_rays) {
    Rnd rnd(15);
    World w = random_world(rnd, 6, 8);
    make_three_node(w);
    CHECK(w.nodes.size() == 3u);
    Agree a = compare_on_random_rays(w.view(), rnd, 20000);
    CHECK(a.mismatches == 0);
    CHECK(a.hits > 2000 && a.sphere_hits > 500 && a.tri_hits > 300);
}

TEST(five_node_preorder_bvh_with_nested_skip_links_matches_bruteforce) {
    Rnd rnd(16);
    World w = random_world(rnd, 9, 12);
    make_five_node(w);
    CHECK(w.nodes.size() == 5u);
    CHECK(w.nodes[1].skip == 4 && w.nodes[2].skip == 3 && w.nodes[3].skip == 4 && w.nodes[4].skip == -1);
    Agree a = compare_on_random_rays(w.view(), rnd, 20000);
    CHECK(a.mismatches == 0);
    CHECK(a.hits > 2000 && a.sphere_hits > 500 && a.tri_hits > 300);
}

namespace {
// Two unit spheres at x=-3 (prim 0, in the left leaf) and x=+3 (prim 1, in the right leaf).
World two_sphere_tree() {
    World w;
    w.sp = {mksphere(V(-3, 0, 0), 1), mksphere(V(3, 0, 0), 1)};
    w.order = {0, 1};
    w.nodes = {inner(V(-4, -1, -1), V(4, 1, 1), -1),
               leaf(V(-4, -1, -1), V(-2, 1, 1), 2, 0, 1),
               leaf(V(2, -1, -1), V(4, 1, 1), -1, 1, 1)};
    return w;
}
}  // namespace

TEST(skip_link_from_left_leaf_reaches_right_leaf_when_left_box_is_missed) {
    World w = two_sphere_tree();
    SceneView v = w.view();
    HitRecord h;
    CHECK(intersect(v, mk(V(3, 0, -10), V(0, 0, 1)), &h));  // only the right sphere is on this ray
    CHECK(h.prim == 1);
    CHECK_NEAR(h.t, 9.0, 1e-5);
}

TEST(skip_link_from_left_leaf_reaches_right_leaf_after_a_left_hit) {
    World w = two_sphere_tree();
    SceneView v = w.view();
    HitRecord h;
    // Ray along +x from x=-10 passes both: left at t=6, right at t=12. Nearest wins.
    CHECK(intersect(v, mk(V(-10, 0, 0), V(1, 0, 0)), &h));
    CHECK(h.prim == 0); CHECK_NEAR(h.t, 6.0, 1e-5);
    // Ray along -x from x=+10: right is nearer although the left leaf is visited first in preorder.
    CHECK(intersect(v, mk(V(10, 0, 0), V(-1, 0, 0)), &h));
    CHECK(h.prim == 1); CHECK_NEAR(h.t, 6.0, 1e-5);
}

TEST(traversal_terminates_with_a_clean_miss_record) {
    World w = two_sphere_tree();
    SceneView v = w.view();
    HitRecord h;
    h.t = 1; h.prim = 99; h.u = 5; h.v = 5;  // stale garbage must be overwritten
    CHECK(!intersect(v, mk(V(0, 5, 0), V(1, 0, 0)), &h));
    CHECK(h.prim == -1);
    CHECK_NEAR(h.t, kHuge, 0);
    CHECK(!intersect(v, mk(V(-10, 0, 0), V(1, 0, 0), 1e-4f, 5.0f), &h));  // tmax=5 stops short of x=-4
    CHECK(h.prim == -1); CHECK_NEAR(h.t, 5.0, 0);
}

TEST(anyhit_returns_a_valid_primitive_and_agrees_with_closest_hit_on_occlusion) {
    Rnd rnd(17);
    World w = random_world(rnd, 6, 8);
    make_three_node(w);
    SceneView v = w.view();
    int agree = 0, total = 0, occluded = 0;
    for (int i = 0; i < 5000; ++i) {
        Ray r = random_ray(rnd);
        HitRecord ha, hc;
        bool a = intersect(v, r, &ha, true), c = intersect(v, r, &hc, false);
        ++total; occluded += c;
        if (a == c) ++agree;
        if (a) {  // any-hit must report some primitive that is genuinely hit, never nearer than the closest
            CHECK(ha.prim >= 0 && ha.prim < v.n_spheres + v.n_triangles);
            CHECK(ha.t >= hc.t - 1e-5f * hc.t);
            CHECK(ha.t < r.tmax);
        }
    }
    CHECK(agree == total);
    CHECK(occluded > 500);
}

TEST(bruteforce_prefers_the_nearest_and_breaks_exact_ties_towards_the_lower_id) {
    // Spheres 0 and 1 coincide (same t); strictly-nearer-wins means prim 0 is reported.
    World w;
    w.sp = {mksphere(V(0, 0, 0), 1), mksphere(V(0, 0, 0), 1), mksphere(V(0, 0, 3), 1)};
    make_single_leaf(w);
    SceneView v = w.view();
    HitRecord h;
    CHECK(intersect_bruteforce(v, mk(V(0, 0, -5), V(0, 0, 1)), &h));
    CHECK(h.prim == 0); CHECK_NEAR(h.t, 4.0, 1e-5);
    HitRecord h2;
    CHECK(intersect(v, mk(V(0, 0, -5), V(0, 0, 1)), &h2));
    CHECK(h2.prim == 0); CHECK_NEAR(h2.t, 4.0, 1e-5);  // BVH traversal uses the same tie rule
    // Triangle ids come after sphere ids: a triangle in front of every sphere is reported with id n_spheres.
    w.add_tri(V(-1, -1, -2), V(1, -1, -2), V(0, 1, -2));
    make_single_leaf(w);
    SceneView v2 = w.view();
    CHECK(intersect_bruteforce(v2, mk(V(0, 0, -5), V(0, 0, 1)), &h));
    CHECK(h.prim == 3); CHECK_NEAR(h.t, 3.0, 1e-5);
}

TEST(empty_scene_never_hits) {
    World w;
    SceneView v = w.view();
    HitRecord h;
    CHECK(!intersect(v, mk(V(0, 0, 0), V(0, 0, 1)), &h));
    CHECK(h.prim == -1);
    CHECK(!intersect_bruteforce(v, mk(V(0, 0, 0), V(0, 0, 1)), &h));
    CHECK(h.prim == -1);
}
