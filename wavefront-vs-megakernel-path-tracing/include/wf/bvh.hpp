// BVH build (CPU), brute-force oracle and stackless traversal.
#pragma once
#include "wf/math.hpp"
#include "wf/scene.hpp"

namespace wf {

// Builds scene.nodes and scene.prim_order over all primitives with binned SAH (leaf <= 4).
// Preorder layout with skip links; node 0 is the root; skip of the last node on the right-most
// path is -1. Throws std::runtime_error if the scene has no primitives.
// A leaf never holds more than 4 primitives, also for identical primitives. The skip of a left
// child is its right sibling; a right child shares its parent's skip.
void build_bvh(Scene& scene);

// Checks structural invariants: every primitive appears exactly once in prim_order, node boxes
// contain children and primitives, skip links point forward (or -1), preorder left = N+1.
// Returns an empty string if valid, otherwise the first violated invariant.
std::string validate_bvh(const Scene& scene);

// ---- primitive intersection (host + device, header-only) ----

// Ray-sphere with a projected-distance discriminant: w = oc - d*(oc.d/a) is the offset of the
// sphere centre from the ray line, disc = r^2 - |w|^2, t = -(oc.d)/a +- sqrt(disc/a). This avoids
// the cancellation of |oc|^2 - r^2 for small or distant spheres. Nearest t in (tmin, tmax).
WF_HD bool hit_sphere(const Sphere& s, const Ray& r, float* t_out) {
    Vec3 oc = r.o - s.center;
    float a = dot(r.d, r.d);
    float b = dot(oc, r.d);
    Vec3 w = oc - r.d * (b / a);
    float disc = s.radius * s.radius - dot(w, w);
    if (disc < 0.0f) return false;
    float mid = -b / a;
    float sq = sqrtf(disc / a);
    float t = mid - sq;
    if (!(t > r.tmin && t < r.tmax)) {
        t = mid + sq;
        if (!(t > r.tmin && t < r.tmax)) return false;
    }
    *t_out = t;
    return true;
}

// Moller-Trumbore ray-triangle; u,v are barycentrics of v1 and v2.
WF_HD bool hit_triangle(Vec3 p0, Vec3 p1, Vec3 p2, const Ray& r, float* t_out, float* u_out,
                        float* v_out) {
    Vec3 e1 = p1 - p0, e2 = p2 - p0;
    Vec3 pv = cross(r.d, e2);
    float det = dot(e1, pv);
    if (det == 0.0f) return false;
    float inv = 1.0f / det;
    Vec3 tv = r.o - p0;
    float u = dot(tv, pv) * inv;
    if (u < 0.0f || u > 1.0f) return false;
    Vec3 qv = cross(tv, e1);
    float v = dot(r.d, qv) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = dot(e2, qv) * inv;
    if (!(t > r.tmin && t < r.tmax)) return false;
    *t_out = t; *u_out = u; *v_out = v;
    return true;
}

// Slab test against [tmin, tmax].
WF_HD bool hit_box(Vec3 lo, Vec3 hi, Vec3 o, Vec3 inv_d, float tmin, float tmax) {
    float t0 = tmin, t1 = tmax;
    float tn = (lo.x - o.x) * inv_d.x, tf = (hi.x - o.x) * inv_d.x;
    if (tn > tf) { float s = tn; tn = tf; tf = s; }
    t0 = tn > t0 ? tn : t0; t1 = tf < t1 ? tf : t1;
    if (t0 > t1) return false;
    tn = (lo.y - o.y) * inv_d.y; tf = (hi.y - o.y) * inv_d.y;
    if (tn > tf) { float s = tn; tn = tf; tf = s; }
    t0 = tn > t0 ? tn : t0; t1 = tf < t1 ? tf : t1;
    if (t0 > t1) return false;
    tn = (lo.z - o.z) * inv_d.z; tf = (hi.z - o.z) * inv_d.z;
    if (tn > tf) { float s = tn; tn = tf; tf = s; }
    t0 = tn > t0 ? tn : t0; t1 = tf < t1 ? tf : t1;
    return t0 <= t1;
}

WF_HD bool hit_prim(const SceneView& s, int prim, const Ray& r, float* t, float* u, float* v) {
    if (prim < s.n_spheres) {
        *u = 0.0f; *v = 0.0f;
        return hit_sphere(s.spheres[prim], r, t);
    }
    const Triangle& tri = s.triangles[prim - s.n_spheres];
    return hit_triangle(s.vertices[tri.i0], s.vertices[tri.i1], s.vertices[tri.i2], r, t, u, v);
}

// Nearest hit via stackless skip-link traversal. Returns true on a hit; fills *h.
// anyhit=true returns as soon as any primitive is hit (used for shadow rays).
WF_HD bool intersect(const SceneView& s, Ray r, HitRecord* h, bool anyhit = false) {
    h->t = r.tmax; h->prim = -1; h->u = 0.0f; h->v = 0.0f;
    Vec3 inv_d{1.0f / r.d.x, 1.0f / r.d.y, 1.0f / r.d.z};
    int n = 0;
    while (n >= 0 && n < s.n_nodes) {
        const BvhNode& node = s.nodes[n];
        if (hit_box(node.lo, node.hi, r.o, inv_d, r.tmin, r.tmax)) {
            if (node.prim_count > 0) {
                for (int i = 0; i < node.prim_count; ++i) {
                    int p = s.prim_order[node.prim_start + i];
                    float t, u, v;
                    if (hit_prim(s, p, r, &t, &u, &v)) {
                        r.tmax = t; h->t = t; h->prim = p; h->u = u; h->v = v;
                        if (anyhit) return true;
                    }
                }
                n = node.skip;
            } else {
                n = n + 1;
            }
        } else {
            n = node.skip;
        }
    }
    return h->prim >= 0;
}

// Oracle: tests every primitive. Same tie-breaking (strictly nearer wins, lower id first).
WF_HD bool intersect_bruteforce(const SceneView& s, Ray r, HitRecord* h) {
    h->t = r.tmax; h->prim = -1; h->u = 0.0f; h->v = 0.0f;
    int total = s.n_spheres + s.n_triangles;
    for (int p = 0; p < total; ++p) {
        float t, u, v;
        if (hit_prim(s, p, r, &t, &u, &v)) {
            r.tmax = t; h->t = t; h->prim = p; h->u = u; h->v = v;
        }
    }
    return h->prim >= 0;
}

}  // namespace wf
