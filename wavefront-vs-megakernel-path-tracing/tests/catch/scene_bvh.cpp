// oracle correctness and boundaries for scenes, primitive intersection and the BVH.
// Oracles: brute force over all primitives (independent of the BVH), analytic roots/barycentrics
// computed by hand, an independent subtree-walk for skip links, metamorphic hash relations.
// Needs src/scenes.cpp (make_scene, finalize_scene, scene_hash) and src/bvh_build.cpp to link.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "wf/bvh.hpp"
#include "wf/path_core.hpp"
#include "wf/scene.hpp"

using namespace wf;
using Catch::Matchers::WithinAbs;

namespace {

// Independent, deterministic generator (xorshift64*), not the project's RNG.
struct Gen {
    uint64_t s;
    explicit Gen(uint64_t seed) : s(seed * 2685821657736338717ULL + 88172645463325252ULL) {}
    float next() {  // [0,1)
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return (float)((s * 2685821657736338717ULL) >> 40) * (1.0f / 16777216.0f);
    }
};

Vec3 inv_of(Vec3 d) { return Vec3{1.0f / d.x, 1.0f / d.y, 1.0f / d.z}; }

Material make_mat(int type, Vec3 albedo) {
    Material m{};
    m.type = type; m.ior = 1.5f; m.albedo = albedo; m.emission = Vec3{0, 0, 0};
    return m;
}

struct Agreement {
    int rays = 0, hits = 0, misses = 0, mismatches = 0, anyhit_bad = 0;
    std::string first;
};

// Camera rays through the rig plus random rays (random origin in the grown root box, random
// direction, half with a finite tmax). Compares stackless traversal with brute force.
Agreement compare_traversal(const Scene& sc, int n_random, uint64_t seed) {
    SceneView v = sc.view();
    Agreement a;
    std::vector<Ray> rays;
    for (int py = 0; py < v.height; py += 2)
        for (int px = 0; px < v.width; px += 2) {
            Vec3 d = v.camera.lower_left + v.camera.horizontal * (((float)px + 0.5f) / v.width) +
                     v.camera.vertical * (((float)py + 0.5f) / v.height) - v.camera.origin;
            rays.push_back(Ray{v.camera.origin, 1e-4f, normalize(d), kInf});
        }
    Vec3 lo = sc.nodes[0].lo, hi = sc.nodes[0].hi, ext = hi - lo;
    float diag = length(ext);
    Gen g(seed);
    for (int i = 0; i < n_random; ++i) {
        Vec3 o{lo.x - 0.2f * ext.x + 1.4f * ext.x * g.next(), lo.y - 0.2f * ext.y + 1.4f * ext.y * g.next(),
               lo.z - 0.2f * ext.z + 1.4f * ext.z * g.next()};
        Vec3 d;
        do { d = Vec3{2 * g.next() - 1, 2 * g.next() - 1, 2 * g.next() - 1}; } while (length2(d) > 1.0f || length2(d) < 1e-4f);
        float tmax = (i % 2) ? kInf : diag * (0.05f + g.next());
        rays.push_back(Ray{o, 1e-4f, normalize(d), tmax});
    }
    for (size_t i = 0; i < rays.size(); ++i) {
        const Ray& r = rays[i];
        HitRecord hs{}, hb{}, ha{};
        bool s = intersect(v, r, &hs, false);
        bool b = intersect_bruteforce(v, r, &hb);
        bool any = intersect(v, r, &ha, true);
        ++a.rays;
        (b ? a.hits : a.misses)++;
        bool same = (s == b) && hs.prim == hb.prim && hs.t == hb.t;
        if (!same) {
            ++a.mismatches;
            if (a.first.empty())
                a.first = "ray " + std::to_string(i) + ": stackless prim=" + std::to_string(hs.prim) +
                          " t=" + std::to_string(hs.t) + " vs brute prim=" + std::to_string(hb.prim) +
                          " t=" + std::to_string(hb.t);
        }
        // anyhit: same hit/miss verdict; a returned primitive really is hit inside the window and
        // can never be nearer than the true nearest hit.
        bool any_ok = (any == b);
        if (any && b) {
            float t, u, w;
            any_ok = any_ok && ha.prim >= 0 && hit_prim(v, ha.prim, r, &t, &u, &w) && t == ha.t &&
                     ha.t >= hb.t;
        }
        if (!any_ok) {
            ++a.anyhit_bad;
            if (a.first.empty()) a.first = "anyhit disagrees on ray " + std::to_string(i);
        }
    }
    return a;
}

void check_scene_agrees(const char* name, int w, int h, int n_random) {
    Scene sc = make_scene(name, w, h);
    Agreement a = compare_traversal(sc, n_random, 77);
    INFO("scene " << name << " rays " << a.rays << " first mismatch: " << a.first);
    // Oracle: the comparison must be non-vacuous (rays that hit and rays that miss both occur).
    CHECK(a.hits >= a.rays / 5);
    CHECK(a.misses >= a.rays / 50);
    CHECK(a.mismatches == 0);
    CHECK(a.anyhit_bad == 0);
}

// Independent end-of-subtree computation: the first node index after the subtree rooted at n.
int subtree_end(const Scene& s, int n) {
    const BvhNode& nd = s.nodes[n];
    if (nd.prim_count > 0) return n + 1;
    int left = n + 1;
    int right = s.nodes[left].skip;  // skip of a left child is its right sibling
    return subtree_end(s, right);
}

}  // namespace

TEST_CASE("stackless traversal equals brute force on cornell", "[scene_bvh]") {
    // Given the cornell scene, When 2000 random plus all camera rays are traced both ways,
    // Then nearest hit (primitive and t bit-exact) and anyhit verdicts agree.
    check_scene_agrees("cornell", 64, 36, 2000);
}

TEST_CASE("stackless traversal equals brute force on divergence", "[scene_bvh]") {
    check_scene_agrees("divergence", 64, 36, 2000);
}

TEST_CASE("stackless traversal equals brute force on trianglehv with reduced rays", "[scene_bvh]") {
    // 100k+ triangles make brute force expensive; a small image and few random rays keep it fast.
    Scene sc = make_scene("trianglehv", 16, 9);
    REQUIRE(sc.triangles.size() >= 100000);  // spec: triangle-heavy, 100k+ triangles
    Agreement a = compare_traversal(sc, 300, 78);
    INFO("first mismatch: " << a.first);
    CHECK(a.hits >= a.rays / 5);
    CHECK(a.mismatches == 0);
    CHECK(a.anyhit_bad == 0);
}

TEST_CASE("hit_sphere gives stable roots far away and no self-hit from the surface", "[scene_bvh]") {
    // Oracle: analytic roots t = Z -/+ r for a ray along +z through the centre.
    // Moderate distance: both roots are well within float precision.
    {
        Sphere s{}; s.center = Vec3{0, 0, 100.0f}; s.radius = 0.5f;
        Ray r{Vec3{0, 0, 0}, 0.0f, Vec3{0, 0, 1}, kInf};
        float t = 0;
        REQUIRE(hit_sphere(s, r, &t));
        CHECK_THAT(t, WithinAbs(99.5, 1e-3));
        Ray late{Vec3{0, 0, 0}, 99.6f, Vec3{0, 0, 1}, kInf};  // near root excluded by tmin
        REQUIRE(hit_sphere(s, late, &t));
        CHECK_THAT(t, WithinAbs(100.5, 1e-3));
        Ray cut{Vec3{0, 0, 0}, 0.0f, Vec3{0, 0, 1}, 99.4f};   // window ends before the sphere
        CHECK_FALSE(hit_sphere(s, cut, &t));
    }
    // Tiny sphere far away: |oc|^2 - r^2 cancels catastrophically when computed naively
    // (1e6 - 0.0625 has ulp 0.0625). Truth: near root 1000 - 0.25 = 999.75. A tolerance of 0.05
    // admits float rounding of the answer but not a lost radius.
    {
        Sphere s{}; s.center = Vec3{0, 0, 1000.0f}; s.radius = 0.25f;
        Ray r{Vec3{0, 0, 0}, 0.0f, Vec3{0, 0, 1}, kInf};
        float t = 0;
        INFO("tiny far sphere: near root must be 999.75 (cancellation in |oc|^2 - r^2 loses the radius)");
        REQUIRE(hit_sphere(s, r, &t));
        CHECK_THAT(t, WithinAbs(999.75, 0.05));
    }
    // Ray starting exactly on the surface of the unit sphere (3-4-5 point is exact enough that
    // c is within ~1e-7 of 0): going outward never self-hits; going inward finds the far side.
    {
        Sphere s{}; s.center = Vec3{0, 0, 0}; s.radius = 1.0f;
        Vec3 p{0.6f, 0.0f, 0.8f};
        float t = 0;
        Ray out{p, 1e-4f, Vec3{0.6f, 0.0f, 0.8f}, kInf};
        CHECK_FALSE(hit_sphere(s, out, &t));
        Ray in{p, 1e-4f, Vec3{-0.6f, 0.0f, -0.8f}, kInf};  // chord through the centre, length 2
        REQUIRE(hit_sphere(s, in, &t));
        CHECK_THAT(t, WithinAbs(2.0, 1e-4));
        Ray chord{p, 1e-4f, Vec3{-1.0f, 0.0f, 0.0f}, kInf};  // to (-0.6,0,0.8): length 1.2
        REQUIRE(hit_sphere(s, chord, &t));
        CHECK_THAT(t, WithinAbs(1.2, 1e-4));
        Ray axis_out{Vec3{0, 0, 1.0f}, 1e-4f, Vec3{0, 0, 1.0f}, kInf};  // exact surface point
        CHECK_FALSE(hit_sphere(s, axis_out, &t));
        Ray axis_in{Vec3{0, 0, 1.0f}, 1e-4f, Vec3{0, 0, -1.0f}, kInf};
        REQUIRE(hit_sphere(s, axis_in, &t));
        CHECK_THAT(t, WithinAbs(2.0, 1e-5));
    }
}

TEST_CASE("triangle edge, vertex and parallel rays and the slab test with zero direction components", "[scene_bvh]") {
    // Triangle (0,0,0),(1,0,0),(0,1,0); a ray down -z from (x,y,1) hits at t=1 with (u,v)=(x,y).
    // All coordinates are dyadic so barycentrics are exact in float.
    Vec3 p0{0, 0, 0}, p1{1, 0, 0}, p2{0, 1, 0};
    auto shoot = [&](float x, float y, float* t, float* u, float* v) {
        Ray r{Vec3{x, y, 1.0f}, 0.0f, Vec3{0, 0, -1}, kInf};
        return hit_triangle(p0, p1, p2, r, t, u, v);
    };
    float t = 0, u = 0, v = 0;
    REQUIRE(shoot(0.25f, 0.25f, &t, &u, &v));
    CHECK_THAT(t, WithinAbs(1.0, 1e-6)); CHECK_THAT(u, WithinAbs(0.25, 1e-6)); CHECK_THAT(v, WithinAbs(0.25, 1e-6));
    // Hypotenuse edge (u+v == 1) and the three vertices are on the closed triangle.
    REQUIRE(shoot(0.5f, 0.5f, &t, &u, &v));
    CHECK_THAT(u + v, WithinAbs(1.0, 1e-6));
    REQUIRE(shoot(0.0f, 0.0f, &t, &u, &v));
    CHECK_THAT(u, WithinAbs(0.0, 1e-6)); CHECK_THAT(v, WithinAbs(0.0, 1e-6));
    REQUIRE(shoot(1.0f, 0.0f, &t, &u, &v));
    CHECK_THAT(u, WithinAbs(1.0, 1e-6)); CHECK_THAT(v, WithinAbs(0.0, 1e-6));
    REQUIRE(shoot(0.0f, 1.0f, &t, &u, &v));
    CHECK_THAT(u, WithinAbs(0.0, 1e-6)); CHECK_THAT(v, WithinAbs(1.0, 1e-6));
    // Just outside each edge (1/64 away): a miss.
    CHECK_FALSE(shoot(0.5f + 1.0f / 64, 0.5f, &t, &u, &v));
    CHECK_FALSE(shoot(-1.0f / 64, 0.25f, &t, &u, &v));
    CHECK_FALSE(shoot(0.25f, -1.0f / 64, &t, &u, &v));
    // Back face is hit too (path tracing needs two-sided triangles for dielectrics).
    {
        Ray r{Vec3{0.25f, 0.25f, -1.0f}, 0.0f, Vec3{0, 0, 1}, kInf};
        REQUIRE(hit_triangle(p0, p1, p2, r, &t, &u, &v));
        CHECK_THAT(t, WithinAbs(1.0, 1e-6));
    }
    // Parallel to the plane (det == 0): above the plane and lying in the plane, both a miss.
    {
        Ray above{Vec3{0.25f, 0.25f, 1.0f}, 0.0f, Vec3{1, 0, 0}, kInf};
        CHECK_FALSE(hit_triangle(p0, p1, p2, above, &t, &u, &v));
        Ray inplane{Vec3{-1.0f, 0.25f, 0.0f}, 0.0f, Vec3{1, 0, 0}, kInf};
        CHECK_FALSE(hit_triangle(p0, p1, p2, inplane, &t, &u, &v));
    }
    // Ray window [tmin,tmax] excludes the t=1 hit on either side.
    {
        Ray late{Vec3{0.25f, 0.25f, 1.0f}, 1.5f, Vec3{0, 0, -1}, kInf};
        CHECK_FALSE(hit_triangle(p0, p1, p2, late, &t, &u, &v));
        Ray early{Vec3{0.25f, 0.25f, 1.0f}, 0.0f, Vec3{0, 0, -1}, 0.5f};
        CHECK_FALSE(hit_triangle(p0, p1, p2, early, &t, &u, &v));
    }

    // Slab test, box [-1,1]^3. Along +z from z=-5 the interval is t in [4,6]. Zero direction
    // components give inv_d = +-inf; the verdict is decided by whether the origin lies in that slab.
    Vec3 lo{-1, -1, -1}, hi{1, 1, 1};
    auto box = [&](Vec3 o, Vec3 d, float tmin, float tmax) { return hit_box(lo, hi, o, inv_of(d), tmin, tmax); };
    CHECK(box(Vec3{0, 0, -5}, Vec3{0, 0, 1}, 0.0f, kInf));              // both x,y zero, inside slabs
    CHECK_FALSE(box(Vec3{2, 0, -5}, Vec3{0, 0, 1}, 0.0f, kInf));        // outside the x slab
    CHECK_FALSE(box(Vec3{-2, 0, -5}, Vec3{0, 0, 1}, 0.0f, kInf));       // outside, other side
    CHECK_FALSE(box(Vec3{0, 2, -5}, Vec3{0, 0, 1}, 0.0f, kInf));        // outside the y slab
    CHECK_FALSE(box(Vec3{0, -2, -5}, Vec3{0, 0, 1}, 0.0f, kInf));
    CHECK(box(Vec3{0.5f, 0.5f, -5}, Vec3{-0.0f, -0.0f, 1}, 0.0f, kInf)); // negative zero => -inf
    CHECK_FALSE(box(Vec3{2, 0, -5}, Vec3{-0.0f, 0, 1}, 0.0f, kInf));
    CHECK(box(Vec3{-3, 0, 0}, Vec3{1, 0, 0}, 0.0f, kInf));              // y and z zero, along x
    CHECK_FALSE(box(Vec3{-3, 0, 1.5f}, Vec3{1, 0, 0}, 0.0f, kInf));     // z origin outside
    CHECK_FALSE(box(Vec3{0, 0, -5}, Vec3{0, 0, -1}, 0.0f, kInf));       // pointing away
    // Window: entry at t=4, exit at t=6.
    CHECK_FALSE(box(Vec3{0, 0, -5}, Vec3{0, 0, 1}, 0.0f, 3.0f));
    CHECK(box(Vec3{0, 0, -5}, Vec3{0, 0, 1}, 0.0f, 4.5f));
    CHECK_FALSE(box(Vec3{0, 0, -5}, Vec3{0, 0, 1}, 7.0f, kInf));
    CHECK(box(Vec3{0, 0, -5}, Vec3{0, 0, 1}, 5.5f, kInf));
    // Diagonal through the corner (-1,-1,-1) at t=2 and out through (1,1,1) at t=4.
    CHECK(box(Vec3{-3, -3, -3}, Vec3{1, 1, 1}, 0.0f, kInf));
    CHECK_FALSE(box(Vec3{-3, -3, -3}, Vec3{1, 1, 1}, 0.0f, 1.5f));
}

TEST_CASE("BVH structural invariants hold on five identical spheres", "[scene_bvh]") {
    // All centroids coincide, so the split heuristic has no spatial separation to use and must
    // fall back to a count split instead of building a leaf with more than 4 primitives.
    Scene s;
    s.name = "degenerate";
    s.materials.push_back(make_mat(kLambert, Vec3{0.5f, 0.5f, 0.5f}));
    for (int i = 0; i < 5; ++i) {
        Sphere sp{}; sp.center = Vec3{1, 2, 3}; sp.radius = 0.5f; sp.material = 0;
        s.spheres.push_back(sp);
    }
    build_bvh(s);
    const int n = (int)s.nodes.size();
    REQUIRE(n >= 3);  // 5 > 4 forces at least one internal node and two leaves
    // Every primitive exactly once.
    {
        std::vector<int> order = s.prim_order;
        std::sort(order.begin(), order.end());
        REQUIRE(order.size() == 5);
        for (int i = 0; i < 5; ++i) CHECK(order[i] == i);
    }
    int leaf_prims = 0;
    for (int i = 0; i < n; ++i) {
        const BvhNode& nd = s.nodes[i];
        INFO("node " << i);
        // skip: oracle is the independent subtree walk, -1 when the subtree ends the array.
        int end = subtree_end(s, i);
        CHECK(nd.skip == (end == n ? -1 : end));
        if (nd.prim_count > 0) {
            CHECK(nd.prim_count <= 4);
            CHECK(nd.prim_start >= 0);
            CHECK(nd.prim_start + nd.prim_count <= 5);
            leaf_prims += nd.prim_count;
            for (int k = 0; k < nd.prim_count; ++k) {  // the box contains each sphere's own box
                const Sphere& sp = s.spheres[s.prim_order[nd.prim_start + k]];
                CHECK(nd.lo.x <= sp.center.x - sp.radius); CHECK(nd.hi.x >= sp.center.x + sp.radius);
                CHECK(nd.lo.y <= sp.center.y - sp.radius); CHECK(nd.hi.y >= sp.center.y + sp.radius);
                CHECK(nd.lo.z <= sp.center.z - sp.radius); CHECK(nd.hi.z >= sp.center.z + sp.radius);
            }
        } else {
            CHECK(nd.prim_count == 0);
            CHECK(nd.prim_start == -1);
            int left = i + 1, right = s.nodes[left].skip;
            REQUIRE(left < n);
            REQUIRE(right > left);
            REQUIRE(right < n);
            for (int c : {left, right}) {  // children inside parent
                CHECK(nd.lo.x <= s.nodes[c].lo.x); CHECK(nd.hi.x >= s.nodes[c].hi.x);
                CHECK(nd.lo.y <= s.nodes[c].lo.y); CHECK(nd.hi.y >= s.nodes[c].hi.y);
                CHECK(nd.lo.z <= s.nodes[c].lo.z); CHECK(nd.hi.z >= s.nodes[c].hi.z);
            }
        }
    }
    CHECK(leaf_prims == 5);  // leaves partition the primitives
    CHECK(s.nodes[0].skip == -1);
    // The finished tree still answers queries: a ray through the common centre hits at 3 - ... t=
    // distance to the shared surface; all five tie so only t is asserted. Origin (1,2,0), +z:
    // near surface at z = 3 - 0.5.
    finalize_scene(s);
    SceneView v = s.view();
    HitRecord h{};
    Ray r{Vec3{1, 2, 0}, 0.0f, Vec3{0, 0, 1}, kInf};
    REQUIRE(intersect(v, r, &h, false));
    CHECK_THAT(h.t, WithinAbs(2.5, 1e-5));
}

TEST_CASE("divergence scene has at least four material types and neighbouring pixels differ", "[scene_bvh]") {
    // Oracle: the material types actually referenced by primitives, and the fraction of
    // horizontally adjacent pixel pairs (both hitting) whose primitive material TYPES differ,
    // measured against the cornell scene (few materials) as a control.
    auto measure = [](const char* name, std::set<int>* types_used) {
        Scene sc = make_scene(name, 128, 72);
        SceneView v = sc.view();
        for (int i = 0; i < v.n_spheres + v.n_triangles; ++i)
            types_used->insert(v.materials[prim_material(v, i)].type);
        std::vector<int> row_prev;
        long pairs = 0, diff = 0;
        for (int py = 0; py < v.height; ++py) {
            int prev = -2;
            for (int px = 0; px < v.width; ++px) {
                Vec3 d = v.camera.lower_left + v.camera.horizontal * (((float)px + 0.5f) / v.width) +
                         v.camera.vertical * (((float)(v.height - 1 - py) + 0.5f) / v.height) - v.camera.origin;
                HitRecord h{};
                int type = -1;
                if (intersect(v, Ray{v.camera.origin, 1e-4f, normalize(d), kInf}, &h))
                    type = v.materials[prim_material(v, h.prim)].type;
                if (prev >= 0 && type >= 0) { ++pairs; diff += (prev != type); }
                prev = type;
            }
        }
        REQUIRE(pairs > 0);
        return (double)diff / (double)pairs;
    };
    std::set<int> div_types, corn_types;
    double fdiv = measure("divergence", &div_types);
    double fcorn = measure("cornell", &corn_types);
    INFO("divergence neighbour-differs fraction " << fdiv << ", cornell " << fcorn);
    CHECK(div_types.size() >= 4);     // Lambert, metal, dielectric, emissive all present
    CHECK(fdiv >= 0.08);              // "often": at least 8% of adjacent hit pairs change type
    CHECK(fdiv > 3.0 * fcorn);        // and far more than the few-material control scene
}

TEST_CASE("scene_hash is stable and sensitive; camera rig centre ray points at look_at", "[scene_bvh]") {
    // ---- scene_hash on a hand scene ----
    auto base = [] {
        Scene s;
        s.name = "h";
        s.materials.push_back(make_mat(kLambert, Vec3{0.5f, 0.6f, 0.7f}));
        s.materials.push_back(make_mat(kMetal, Vec3{0.9f, 0.8f, 0.1f}));
        Sphere a{}; a.center = Vec3{0, 0, 0}; a.radius = 1.0f; a.material = 0;
        Sphere b{}; b.center = Vec3{3, 0, 0}; b.radius = 0.5f; b.material = 1;
        s.spheres = {a, b};
        s.vertices = {Vec3{0, 0, -2}, Vec3{1, 0, -2}, Vec3{0, 1, -2}};
        s.triangles.push_back(Triangle{0, 1, 2, 0});
        s.env = Vec3{0.1f, 0.2f, 0.3f};
        s.camera = Camera{Vec3{0, 0, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 40.0f};
        s.width = 64; s.height = 36;
        return s;
    };
    Scene ref = base();
    uint64_t h0 = scene_hash(ref);
    CHECK(scene_hash(base()) == h0);   // same content built twice
    CHECK(scene_hash(ref) == h0);      // repeated call on one object
    {   // Padding bytes are not content: scribbling on them must not move the hash.
        Scene s = base();
        s.spheres[0].pad[0] = 123; s.spheres[0].pad[2] = -9;
        s.materials[0].pad0 = 5.0f; s.materials[1].pad3 = 7.0f;
        CHECK(scene_hash(s) == h0);
    }
    auto nudge = [](float x) { return std::nextafter(x, 1e9f); };
    std::vector<std::pair<const char*, std::function<void(Scene&)>>> edits = {
        {"sphere centre x one ulp", [&](Scene& s) { s.spheres[1].center.x = nudge(s.spheres[1].center.x); }},
        {"sphere radius one ulp", [&](Scene& s) { s.spheres[0].radius = nudge(s.spheres[0].radius); }},
        {"sphere material index", [](Scene& s) { s.spheres[0].material = 1; }},
        {"sphere order swapped", [](Scene& s) { std::swap(s.spheres[0], s.spheres[1]); }},
        {"vertex z one ulp", [&](Scene& s) { s.vertices[1].z = nudge(s.vertices[1].z); }},
        {"triangle index", [](Scene& s) { s.triangles[0].i2 = 1; }},
        {"triangle material", [](Scene& s) { s.triangles[0].material = 1; }},
        {"material albedo", [&](Scene& s) { s.materials[0].albedo.y = nudge(s.materials[0].albedo.y); }},
        {"material emission", [](Scene& s) { s.materials[1].emission.z = 0.001f; }},
        {"material type", [](Scene& s) { s.materials[0].type = kDielectric; }},
        {"material ior", [](Scene& s) { s.materials[0].ior = 1.51f; }},
        {"env", [&](Scene& s) { s.env.x = nudge(s.env.x); }},
        {"camera position", [&](Scene& s) { s.camera.position.z = nudge(s.camera.position.z); }},
        {"camera look_at", [&](Scene& s) { s.camera.look_at.x = 0.001f; }},
        {"camera up", [](Scene& s) { s.camera.up = Vec3{0, 1, 0.01f}; }},
        {"camera vfov", [&](Scene& s) { s.camera.vfov_deg = nudge(s.camera.vfov_deg); }},
        {"width", [](Scene& s) { s.width = 65; }},
        {"height", [](Scene& s) { s.height = 37; }},
        {"width/height swapped", [](Scene& s) { std::swap(s.width, s.height); }},
    };
    std::set<uint64_t> seen{h0};
    for (auto& [what, edit] : edits) {
        Scene s = base();
        edit(s);
        uint64_t h = scene_hash(s);
        INFO("edit: " << what);
        CHECK(h != h0);
        seen.insert(h);
    }
    CHECK(seen.size() == edits.size() + 1);  // no two distinct edits collide either
    CHECK(scene_hash(make_scene("cornell", 32, 18)) == scene_hash(make_scene("cornell", 32, 18)));
    CHECK(scene_hash(make_scene("cornell", 32, 18)) != scene_hash(make_scene("divergence", 32, 18)));
    CHECK(scene_hash(make_scene("cornell", 32, 18)) != scene_hash(make_scene("cornell", 33, 18)));

    // ---- camera rig ----
    // Oracle: the centre pixel ray (lower_left + h/2 + v/2 - origin) is -w = normalize(look_at - pos).
    auto centre_dir = [](const Camera& c, int w, int hgt) {
        CameraRig r = make_camera_rig(c, w, hgt);
        return normalize(r.lower_left + r.horizontal * 0.5f + r.vertical * 0.5f - r.origin);
    };
    {
        Camera c{Vec3{1, 2, 3}, Vec3{1, 2, -7}, Vec3{0, 1, 0}, 60.0f};
        Vec3 d = centre_dir(c, 16, 9);
        CHECK_THAT(d.x, WithinAbs(0.0, 1e-6)); CHECK_THAT(d.y, WithinAbs(0.0, 1e-6)); CHECK_THAT(d.z, WithinAbs(-1.0, 1e-6));
    }
    {
        Camera c{Vec3{3, 4, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 35.0f};  // |p| = sqrt(50)
        Vec3 d = centre_dir(c, 7, 11);
        float n = std::sqrt(50.0f);
        CHECK_THAT(d.x, WithinAbs(-3.0 / n, 1e-6)); CHECK_THAT(d.y, WithinAbs(-4.0 / n, 1e-6)); CHECK_THAT(d.z, WithinAbs(-5.0 / n, 1e-6));
    }
    {   // fov 90 => |vertical| = 2, |horizontal| = 2*aspect; right-handed: +x right, +y up
        Camera c{Vec3{0, 0, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 90.0f};
        CameraRig r = make_camera_rig(c, 16, 9);
        CHECK_THAT(length(r.vertical), WithinAbs(2.0, 1e-5));
        CHECK_THAT(length(r.horizontal), WithinAbs(2.0 * 16.0 / 9.0, 1e-5));
        CHECK_THAT(dot(r.horizontal, r.vertical), WithinAbs(0.0, 1e-5));
        CHECK(r.horizontal.x > 0.0f);
        CHECK(r.vertical.y > 0.0f);
    }
}
