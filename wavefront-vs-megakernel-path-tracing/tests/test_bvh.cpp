// BVH tests: structure, stackless traversal vs brute force, any-hit, degenerate inputs.
// Needs src/scenes.cpp + src/bvh_build.cpp to link.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "wf/bvh.hpp"
#include "wf/path_core.hpp"
#include "wf/scene.hpp"

#include "test_framework.hpp"

using namespace wf;

namespace {

const char* const kNames[4] = {"cornell", "divergence", "trianglehv", "furnace"};

const Scene& cached(const std::string& name) {
    static std::map<std::string, Scene> cache;
    auto it = cache.find(name);
    if (it == cache.end()) it = cache.emplace(name, make_scene(name, 64, 36)).first;
    return it->second;
}

Material mat(int type, Vec3 albedo) {
    Material m{};
    m.type = type; m.ior = 1.5f; m.albedo = albedo; m.emission = Vec3{0, 0, 0};
    return m;
}
Sphere sph(Vec3 c, float r, int m) {
    Sphere s{};
    s.center = c; s.radius = r; s.material = m;
    return s;
}
Scene base_scene() {
    Scene s;
    s.name = "hand";
    s.materials.push_back(mat(kLambert, Vec3{0.5f, 0.5f, 0.5f}));
    s.camera = Camera{Vec3{0, 0, 20}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 45.0f};
    s.width = 16; s.height = 9;
    return s;
}

// ---- independent primitive bounds
struct Box { Vec3 lo, hi; };

Box prim_box(const Scene& s, int p) {
    if (p < (int)s.spheres.size()) {
        const Sphere& sp = s.spheres[p];
        Vec3 r{sp.radius, sp.radius, sp.radius};
        return Box{sp.center - r, sp.center + r};
    }
    const Triangle& t = s.triangles[p - (int)s.spheres.size()];
    Vec3 a = s.vertices[t.i0], b = s.vertices[t.i1], c = s.vertices[t.i2];
    return Box{vmin(a, vmin(b, c)), vmax(a, vmax(b, c))};
}
Box scene_box(const Scene& s) {
    Box b = prim_box(s, 0);
    for (int p = 1; p < s.n_prims(); ++p) {
        Box q = prim_box(s, p);
        b.lo = vmin(b.lo, q.lo); b.hi = vmax(b.hi, q.hi);
    }
    return b;
}
bool contains(const Box& outer, Vec3 lo, Vec3 hi) {
    auto ok = [](float olo, float ohi, float lo2, float hi2) {
        float slack = 1e-5f * (1.0f + std::fabs(olo) + std::fabs(ohi));
        return olo <= lo2 + slack && hi2 <= ohi + slack;
    };
    return ok(outer.lo.x, outer.hi.x, lo.x, hi.x) && ok(outer.lo.y, outer.hi.y, lo.y, hi.y) &&
           ok(outer.lo.z, outer.hi.z, lo.z, hi.z);
}

// Independent preorder / skip-link invariant checker. Returns "" if fine.
std::string preorder_violation(const Scene& s) {
    const int n = (int)s.nodes.size();
    const int np = s.n_prims();
    if (n <= 0) return "no nodes";
    if (s.nodes.back().skip != -1) return "last node skip is not -1";
    if ((int)s.prim_order.size() != np) return "prim_order size != number of primitives";
    std::vector<int> seen(np, 0);
    int leaves = 0;
    std::vector<int> slot_used(np, 0);
    Box root{s.nodes[0].lo, s.nodes[0].hi};
    for (int p = 0; p < np; ++p) {
        Box b = prim_box(s, p);
        if (!contains(root, b.lo, b.hi)) return "root box does not contain primitive " + std::to_string(p);
    }
    for (int i = 0; i < n; ++i) {
        const BvhNode& nd = s.nodes[i];
        if (nd.skip != -1 && nd.skip <= i) return "skip of node " + std::to_string(i) + " is not greater than the node index";
        if (nd.skip >= n) return "skip of node " + std::to_string(i) + " out of range";
        Box nb{nd.lo, nd.hi};
        if (nd.prim_count > 0) {
            ++leaves;
            if (nd.prim_count > 4) return "leaf " + std::to_string(i) + " has more than 4 primitives";
            if (nd.prim_start < 0 || nd.prim_start + nd.prim_count > np) return "leaf range out of prim_order";
            if (i + 1 < n && nd.skip == -1) return "leaf with skip -1 that is not the last node";
            for (int k = 0; k < nd.prim_count; ++k) {
                int slot = nd.prim_start + k;
                if (slot_used[slot]++) return "prim_order slot used by two leaves";
                int p = s.prim_order[slot];
                if (p < 0 || p >= np) return "prim_order entry out of range";
                ++seen[p];
                Box b = prim_box(s, p);
                if (!contains(nb, b.lo, b.hi)) return "leaf " + std::to_string(i) + " box does not contain primitive " + std::to_string(p);
            }
        } else {
            if (nd.prim_count != 0) return "negative prim_count";
            if (nd.prim_start != -1) return "internal node " + std::to_string(i) + " has prim_start != -1";
            int left = i + 1;
            if (left >= n) return "internal node without a left child";
            int right = s.nodes[left].skip;       // end of left subtree == start of right subtree
            if (right <= left || right >= n) return "right child index of node " + std::to_string(i) + " invalid";
            if (s.nodes[right].skip != nd.skip) return "right child of node " + std::to_string(i) + " does not share its skip";
            const BvhNode& l = s.nodes[left];
            const BvhNode& r = s.nodes[right];
            if (!contains(nb, l.lo, l.hi) || !contains(nb, r.lo, r.hi)) return "children of node " + std::to_string(i) + " escape the parent box";
        }
    }
    for (int p = 0; p < np; ++p)
        if (seen[p] != 1) return "primitive " + std::to_string(p) + " appears " + std::to_string(seen[p]) + " times";
    if (n != 2 * leaves - 1) return "not a full binary tree (nodes != 2*leaves-1)";
    return "";
}

void expect_empty(const std::string& msg, const char* what) {
    if (!msg.empty()) std::printf("  %s: %s\n", what, msg.c_str());
    CHECK(msg.empty());
}

// ---- random rays (own generator: splitmix64)
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uni() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    double gauss() {
        double u1 = uni(), u2 = uni();
        if (u1 < 1e-300) u1 = 1e-300;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

Ray random_ray(Rng& g, const Box& b, int i) {
    Vec3 ext = b.hi - b.lo;
    Vec3 mid = (b.lo + b.hi) * 0.5f;
    Vec3 o;
    if (i % 2 == 0) {   // inside the scene bounds
        o = Vec3{b.lo.x + (float)g.uni() * ext.x, b.lo.y + (float)g.uni() * ext.y, b.lo.z + (float)g.uni() * ext.z};
    } else {            // outside: sample the 3x enlarged box until it falls outside the bounds
        for (;;) {
            o = Vec3{mid.x + (float)(g.uni() - 0.5) * 3.0f * ext.x, mid.y + (float)(g.uni() - 0.5) * 3.0f * ext.y,
                     mid.z + (float)(g.uni() - 0.5) * 3.0f * ext.z};
            bool in = o.x >= b.lo.x && o.x <= b.hi.x && o.y >= b.lo.y && o.y <= b.hi.y && o.z >= b.lo.z && o.z <= b.hi.z;
            if (!in) break;
        }
    }
    Vec3 d;
    if (i % 10 == 3) {  // exactly axis-aligned: zero direction components -> infinite inverse
        int axis = (int)(g.next() % 3);
        float sgn = (g.next() & 1) ? 1.0f : -1.0f;
        d = Vec3{axis == 0 ? sgn : 0.0f, axis == 1 ? sgn : 0.0f, axis == 2 ? sgn : 0.0f};
    } else {
        d = Vec3{(float)g.gauss(), (float)g.gauss(), (float)g.gauss()};
        if (length(d) < 1e-6f) d = Vec3{0, 1, 0};
        d = normalize(d);
    }
    Ray r;
    r.o = o; r.d = d; r.tmin = 1e-4f;
    r.tmax = (i % 3 == 0) ? (float)(g.uni() * length(ext)) + 1e-3f : kInf;   // some finite far clips
    return r;
}

// Compares intersect / anyhit against brute force over n random rays; prints the first offenders.
void compare_scene(const Scene& s, int n, uint64_t seed, const char* name, bool expect_hits = true) {
    SceneView v = s.view();
    Box b = scene_box(s);
    Rng g(seed);
    int hits = 0, bad_hit = 0, bad_t = 0, bad_prim = 0, ties = 0, bad_any = 0, printed = 0;
    for (int i = 0; i < n; ++i) {
        Ray r = random_ray(g, b, i);
        HitRecord a, c, any;
        bool ha = intersect(v, r, &a);
        bool hb = intersect_bruteforce(v, r, &c);
        bool hy = intersect(v, r, &any, true);
        hits += hb;
        if (ha != hb) {
            ++bad_hit;
            if (printed++ < 3) std::printf("  %s ray %d: bvh hit=%d brute hit=%d\n", name, i, (int)ha, (int)hb);
            continue;
        }
        if (hy != hb) {
            ++bad_any;
            if (printed++ < 3) std::printf("  %s ray %d: anyhit=%d brute hit=%d\n", name, i, (int)hy, (int)hb);
        }
        if (!hb) continue;
        double rel = std::fabs((double)a.t - (double)c.t) / std::max(std::fabs((double)c.t), 1e-12);
        if (rel > 1e-5) {
            ++bad_t;
            if (printed++ < 3) std::printf("  %s ray %d: t bvh=%.9g brute=%.9g\n", name, i, a.t, c.t);
        }
        if (a.prim != c.prim) {
            // Allowed only for a genuine tie: both primitives really are hit at |t1-t2| < 1e-6 rel.
            float t1, u1, v1, t2, u2, v2;
            bool h1 = hit_prim(v, a.prim, r, &t1, &u1, &v1);
            bool h2 = hit_prim(v, c.prim, r, &t2, &u2, &v2);
            double tr = std::fabs((double)t1 - (double)t2) / std::max(std::fabs((double)t2), 1e-12);
            if (h1 && h2 && tr < 1e-6) ++ties;
            else {
                ++bad_prim;
                if (printed++ < 3) std::printf("  %s ray %d: prim bvh=%d brute=%d (t %.9g vs %.9g)\n", name, i, a.prim, c.prim, a.t, c.t);
            }
        } else if (a.prim >= v.n_spheres) {
            if (std::fabs(a.u - c.u) > 1e-4f || std::fabs(a.v - c.v) > 1e-4f) {
                ++bad_prim;
                if (printed++ < 3) std::printf("  %s ray %d: barycentrics differ\n", name, i);
            }
        }
        if (hy) {   // any-hit result: a real hit of the ray, not nearer than the nearest hit
            float t2, u2, v2;
            bool real = hit_prim(v, any.prim, r, &t2, &u2, &v2);
            if (!real || t2 != any.t || any.t < c.t * (1.0f - 1e-5f) || !(any.t > r.tmin) || !(any.t < r.tmax)) {
                ++bad_any;
                if (printed++ < 3) std::printf("  %s ray %d: anyhit record inconsistent (prim %d t %.9g nearest %.9g)\n", name, i, any.prim, any.t, c.t);
            }
        }
    }
    std::printf("  %s: %d rays, %d hits, %d ties\n", name, n, hits, ties);
    CHECK(bad_hit == 0);
    CHECK(bad_t == 0);
    CHECK(bad_prim == 0);
    CHECK(bad_any == 0);
    if (expect_hits) {
        CHECK(hits > n / 50);   // the comparison must not be vacuous
        CHECK(hits < n);        // and must contain misses (finite tmax / outside origins)
    }
}

}  // namespace

// ---------------------------------------------------------------- structure

TEST(bvh_validate_accepts_every_procedural_scene) {
    for (const char* name : kNames) expect_empty(validate_bvh(cached(name)), name);
}

TEST(bvh_preorder_and_skip_invariants_hold_on_every_procedural_scene) {
    // Independent re-check of: root = node 0 containing everything, left child N+1, skips strictly
    // forward or -1, last skip -1, right child shares its parent's skip, every primitive exactly once,
    // leaves <= 4, full binary tree.
    for (const char* name : kNames) expect_empty(preorder_violation(cached(name)), name);
}

TEST(bvh_rebuild_is_deterministic_and_replaces_previous_nodes) {
    Scene s = cached("cornell");
    std::vector<int> order = s.prim_order;
    size_t n = s.nodes.size();
    build_bvh(s);
    expect_empty(validate_bvh(s), "rebuild");
    CHECK(s.nodes.size() == n);          // nodes were replaced, not appended
    CHECK(s.prim_order == order);        // same input -> same layout
}

TEST(bvh_build_on_empty_scene_throws_runtime_error_and_leaves_scene_untouched) {
    Scene s = base_scene();
    bool threw = false;
    try { build_bvh(s); } catch (const std::runtime_error&) { threw = true; } catch (...) {}
    CHECK(threw);
    CHECK(s.nodes.empty());
    CHECK(s.prim_order.empty());
}

TEST(bvh_single_primitive_is_one_leaf) {
    Scene s = base_scene();
    s.spheres.push_back(sph(Vec3{1, 2, 3}, 2.0f, 0));
    build_bvh(s);
    CHECK(s.nodes.size() == 1u);
    CHECK(s.prim_order.size() == 1u);
    if (s.nodes.size() == 1 && s.prim_order.size() == 1) {
        CHECK(s.nodes[0].prim_count == 1);
        CHECK(s.nodes[0].skip == -1);
        CHECK(s.prim_order[0] == 0);
        // Box is the sphere's bounds: centre +- radius.
        CHECK_NEAR(s.nodes[0].lo.x, -1.0, 1e-5); CHECK_NEAR(s.nodes[0].hi.x, 3.0, 1e-5);
        CHECK_NEAR(s.nodes[0].lo.y, 0.0, 1e-5);  CHECK_NEAR(s.nodes[0].hi.y, 4.0, 1e-5);
        CHECK_NEAR(s.nodes[0].lo.z, 1.0, 1e-5);  CHECK_NEAR(s.nodes[0].hi.z, 5.0, 1e-5);
    }
    expect_empty(validate_bvh(s), "single sphere");
    expect_empty(preorder_violation(s), "single sphere");
}

TEST(bvh_five_identical_spheres_degenerate_sah_still_builds_a_valid_tree) {
    // Identical centroids: every SAH split has zero spread. Must terminate, split (leaf <= 4) and
    // keep every primitive exactly once.
    Scene s = base_scene();
    for (int i = 0; i < 5; ++i) s.spheres.push_back(sph(Vec3{0, 0, 0}, 1.0f, 0));
    build_bvh(s);
    expect_empty(validate_bvh(s), "5 identical spheres");
    expect_empty(preorder_violation(s), "5 identical spheres");
}

TEST(bvh_five_identical_spheres_traversal_agrees_with_brute_force) {
    Scene s = base_scene();
    for (int i = 0; i < 5; ++i) s.spheres.push_back(sph(Vec3{0, 0, 0}, 1.0f, 0));
    finalize_scene(s);
    SceneView v = s.view();
    // Analytic: from (0,0,10) toward -z the first surface is at t = 10 - 1 = 9 for all five (tie).
    Ray r{Vec3{0, 0, 10}, 1e-4f, Vec3{0, 0, -1}, kInf};
    HitRecord h;
    CHECK(intersect(v, r, &h));
    CHECK_NEAR(h.t, 9.0, 1e-4);
    CHECK(h.prim >= 0 && h.prim < 5);
    HitRecord any;
    CHECK(intersect(v, r, &any, true));
    // Miss: parallel offset by more than the radius.
    Ray miss{Vec3{1.5f, 0, 10}, 1e-4f, Vec3{0, 0, -1}, kInf};
    CHECK(!intersect(v, miss, &h));
    CHECK(h.prim == -1);
    compare_scene(s, 20000, 77, "five identical spheres");
}

TEST(bvh_single_triangle_scene_matches_hand_computed_hits) {
    // Triangle (0,0,0),(1,0,0),(0,1,0). Ray from (0.25,0.25,1) toward -z: t=1, u=v=0.25.
    Scene s = base_scene();
    s.vertices = {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}};
    s.triangles.push_back(Triangle{0, 1, 2, 0});
    finalize_scene(s);
    SceneView v = s.view();
    HitRecord h;
    CHECK(intersect(v, Ray{Vec3{0.25f, 0.25f, 1}, 1e-4f, Vec3{0, 0, -1}, kInf}, &h));
    CHECK_NEAR(h.t, 1.0, 1e-6); CHECK_NEAR(h.u, 0.25, 1e-6); CHECK_NEAR(h.v, 0.25, 1e-6);
    CHECK(h.prim == 0);   // n_spheres == 0 so the triangle id is 0
    // Back side: triangles are double sided.
    CHECK(intersect(v, Ray{Vec3{0.25f, 0.25f, -1}, 1e-4f, Vec3{0, 0, 1}, kInf}, &h));
    CHECK_NEAR(h.t, 1.0, 1e-6);
    // Outside the hypotenuse (u+v > 1), parallel to the plane, and clipped by tmax.
    CHECK(!intersect(v, Ray{Vec3{0.75f, 0.75f, 1}, 1e-4f, Vec3{0, 0, -1}, kInf}, &h));
    CHECK(!intersect(v, Ray{Vec3{-1, 0.25f, 1}, 1e-4f, Vec3{1, 0, 0}, kInf}, &h));
    CHECK(!intersect(v, Ray{Vec3{0.25f, 0.25f, 1}, 1e-4f, Vec3{0, 0, -1}, 0.5f}, &h));
    compare_scene(s, 20000, 5, "single triangle", false);   // flat bounds: hits are rare, hand cases above carry it
}

// ---------------------------------------------------------------- rays inside spheres

TEST(bvh_rays_starting_inside_a_sphere_exit_at_the_analytic_distance) {
    // Sphere centre c, radius R; origin o inside, unit d: t = -b + sqrt(b^2 - (|oc|^2 - R^2)),
    // b = dot(oc, d), computed in double. Origin at the centre gives t = R for every direction.
    Scene s = base_scene();
    const Vec3 c{1, 2, 3};
    const double R = 2.0;
    s.spheres.push_back(sph(c, (float)R, 0));
    finalize_scene(s);
    SceneView v = s.view();
    Rng g(99);
    int bad = 0;
    for (int i = 0; i < 5000; ++i) {
        Vec3 d{(float)g.gauss(), (float)g.gauss(), (float)g.gauss()};
        d = normalize(d);
        Vec3 off{0, 0, 0};
        if (i > 0) {   // i == 0: exactly the centre
            off = Vec3{(float)(g.uni() - 0.5), (float)(g.uni() - 0.5), (float)(g.uni() - 0.5)} * 3.4f;
            if (length(off) > 1.8f) off = off * (1.8f / length(off));
        }
        Vec3 o = c + off;
        double bq = (double)off.x * d.x + (double)off.y * d.y + (double)off.z * d.z;
        double cq = (double)off.x * off.x + (double)off.y * off.y + (double)off.z * off.z - R * R;
        double t_exp = -bq + std::sqrt(bq * bq - cq);
        HitRecord h, hb;
        Ray r{o, 1e-4f, d, kInf};
        bool ok = intersect(v, r, &h) && intersect_bruteforce(v, r, &hb);
        if (!ok || h.prim != 0 || std::fabs(h.t - t_exp) > 1e-4 * t_exp + 1e-5 || std::fabs(hb.t - t_exp) > 1e-4 * t_exp + 1e-5) {
            if (bad++ < 3) std::printf("  inside-sphere ray %d: t=%.9g expected %.9g\n", i, h.t, t_exp);
        }
    }
    CHECK(bad == 0);
}

TEST(bvh_nested_spheres_report_the_nearest_surface_from_inside) {
    // Outer R=2 (id 0), inner R=1 (id 1), same centre. From (1.5,0,0): inward hits the inner sphere
    // at t = 0.5 (x=1), outward hits the outer sphere at t = 0.5 (x=2).
    Scene s = base_scene();
    s.spheres.push_back(sph(Vec3{0, 0, 0}, 2.0f, 0));
    s.spheres.push_back(sph(Vec3{0, 0, 0}, 1.0f, 0));
    finalize_scene(s);
    SceneView v = s.view();
    HitRecord h;
    CHECK(intersect(v, Ray{Vec3{1.5f, 0, 0}, 1e-4f, Vec3{-1, 0, 0}, kInf}, &h));
    CHECK(h.prim == 1);
    CHECK_NEAR(h.t, 0.5, 1e-5);
    CHECK(intersect(v, Ray{Vec3{1.5f, 0, 0}, 1e-4f, Vec3{1, 0, 0}, kInf}, &h));
    CHECK(h.prim == 0);
    CHECK_NEAR(h.t, 0.5, 1e-5);
    // From the shared centre the inner sphere is hit first, at t = 1.
    CHECK(intersect(v, Ray{Vec3{0, 0, 0}, 1e-4f, Vec3{0, 1, 0}, kInf}, &h));
    CHECK(h.prim == 1);
    CHECK_NEAR(h.t, 1.0, 1e-5);
}

// ---------------------------------------------------------------- traversal vs brute force

TEST(bvh_traversal_matches_brute_force_on_cornell) { compare_scene(cached("cornell"), 100000, 11, "cornell"); }
TEST(bvh_traversal_matches_brute_force_on_divergence) { compare_scene(cached("divergence"), 100000, 12, "divergence"); }
TEST(bvh_traversal_matches_brute_force_on_furnace) { compare_scene(cached("furnace"), 100000, 14, "furnace"); }

TEST(bvh_traversal_matches_brute_force_on_trianglehv) {
    // 100k+ triangles: brute force is O(rays x triangles), so this scene uses 3000 rays to stay inside
    // the per-file time budget (the other three scenes use the full 100000).
    compare_scene(cached("trianglehv"), 3000, 13, "trianglehv");
}

// ---------------------------------------------------------------- validator is not vacuous

TEST(bvh_validate_rejects_duplicate_primitive_in_prim_order) {
    Scene s = cached("divergence");
    CHECK(s.prim_order.size() >= 2u);
    CHECK(s.prim_order[0] != s.prim_order[1]);
    s.prim_order[0] = s.prim_order[1];   // one primitive now appears twice, another not at all
    CHECK(!validate_bvh(s).empty());
}

TEST(bvh_validate_rejects_backward_skip_link) {
    Scene s = cached("divergence");
    CHECK(s.nodes.size() >= 3u);
    s.nodes[1].skip = 0;   // points backward
    CHECK(!validate_bvh(s).empty());
}

TEST(bvh_validate_rejects_root_box_that_does_not_contain_its_children) {
    Scene s = cached("divergence");
    CHECK(s.nodes.size() >= 3u);
    s.nodes[0].hi = s.nodes[0].lo;   // collapse the root box to a point
    CHECK(!validate_bvh(s).empty());
}

TEST(bvh_validate_rejects_last_node_with_a_forward_skip_out_of_range) {
    Scene s = cached("divergence");
    s.nodes.back().skip = (int)s.nodes.size();   // beyond the end instead of -1
    CHECK(!validate_bvh(s).empty());
}
