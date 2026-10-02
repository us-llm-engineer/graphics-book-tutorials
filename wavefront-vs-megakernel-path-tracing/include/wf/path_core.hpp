// The single estimator. CPU renderer, megakernel and every wavefront stage call these functions.
// Order of operations per bounce is documented in DESIGN.md section 4.
#pragma once
#include "wf/bvh.hpp"
#include "wf/math.hpp"
#include "wf/rng.hpp"
#include "wf/scene.hpp"

namespace wf {

constexpr float kRayEps = 1e-4f;

struct PathState {
    Ray ray;
    Vec3 throughput;
    Vec3 L;
    int depth;
    int specular;     // 1 if the last scattering event was a delta event (or at the camera)
    float prev_pdf;   // solid-angle pdf of the last BSDF sample, 0 for delta
    PathKey key;
};

struct Surface {
    Vec3 p;
    Vec3 n;       // geometric normal facing against the ray
    Vec3 n_out;   // geometric outward normal (sphere: radial; triangle: winding)
    int inside;   // 1 if the ray arrives from behind n_out
    int material;
};

struct LightSample {
    Vec3 p;
    Vec3 n;
    float pdf_solid;  // includes the 1/n_lights selection probability; 0 if invalid
    Vec3 Le;
};

WF_HD float power_heuristic(float a, float b) {
    float a2 = a * a, b2 = b * b;
    float s = a2 + b2;
    return s > 0.0f ? a2 / s : 0.0f;
}

WF_HD float prim_area(const SceneView& s, int prim) {
    if (prim < s.n_spheres) {
        float r = s.spheres[prim].radius;
        return 4.0f * kPi * r * r;
    }
    const Triangle& t = s.triangles[prim - s.n_spheres];
    Vec3 e1 = s.vertices[t.i1] - s.vertices[t.i0];
    Vec3 e2 = s.vertices[t.i2] - s.vertices[t.i0];
    return 0.5f * length(cross(e1, e2));
}

WF_HD int prim_material(const SceneView& s, int prim) {
    return prim < s.n_spheres ? s.spheres[prim].material : s.triangles[prim - s.n_spheres].material;
}

WF_HD Ray camera_ray(const SceneView& s, int px, int py, PathKey key) {
    float jx = rng(key, 0, kDimPixelX), jy = rng(key, 0, kDimPixelY);
    float u = ((float)px + jx) / (float)s.width;
    float v = ((float)(s.height - 1 - py) + jy) / (float)s.height;
    Vec3 target = s.camera.lower_left + s.camera.horizontal * u + s.camera.vertical * v;
    Ray r;
    r.o = s.camera.origin;
    r.d = normalize(target - s.camera.origin);
    r.tmin = kRayEps;
    r.tmax = kInf;
    return r;
}

WF_HD PathState init_path(const SceneView& s, int px, int py, uint32_t sample, uint32_t seed) {
    PathState ps;
    ps.key = make_path_key(seed, (uint32_t)(py * s.width + px), sample);
    ps.ray = camera_ray(s, px, py, ps.key);
    ps.throughput = Vec3{1.0f, 1.0f, 1.0f};
    ps.L = Vec3{0.0f, 0.0f, 0.0f};
    ps.depth = 0;
    ps.specular = 1;
    ps.prev_pdf = 0.0f;
    return ps;
}

WF_HD Surface make_surface(const SceneView& s, const Ray& ray, const HitRecord& h) {
    Surface f;
    f.p = ray.o + ray.d * h.t;
    if (h.prim < s.n_spheres) {
        const Sphere& sp = s.spheres[h.prim];
        f.n_out = (f.p - sp.center) / sp.radius;
        f.material = sp.material;
    } else {
        const Triangle& t = s.triangles[h.prim - s.n_spheres];
        Vec3 e1 = s.vertices[t.i1] - s.vertices[t.i0];
        Vec3 e2 = s.vertices[t.i2] - s.vertices[t.i0];
        f.n_out = normalize(cross(e1, e2));
        f.material = t.material;
    }
    f.inside = dot(f.n_out, ray.d) > 0.0f ? 1 : 0;
    f.n = f.inside ? -f.n_out : f.n_out;
    return f;
}

// Uniform point on a light primitive, converted to a solid-angle pdf from point `from`.
WF_HD LightSample sample_light(const SceneView& s, Vec3 from, PathKey key, uint32_t depth) {
    LightSample ls;
    ls.pdf_solid = 0.0f;
    ls.p = from; ls.n = Vec3{0, 0, 1}; ls.Le = Vec3{0, 0, 0};
    if (s.n_lights <= 0) return ls;
    int idx = (int)(rng(key, depth, kDimLightPick) * (float)s.n_lights);
    if (idx >= s.n_lights) idx = s.n_lights - 1;
    int prim = s.lights[idx].prim;
    float u1 = rng(key, depth, kDimLightU), u2 = rng(key, depth, kDimLightV);
    float area = prim_area(s, prim);
    if (prim < s.n_spheres) {
        const Sphere& sp = s.spheres[prim];
        float z = 1.0f - 2.0f * u1;
        float r = sqrtf(fmaxf3(0.0f, 1.0f - z * z));
        float phi = 2.0f * kPi * u2;
        Vec3 dir{r * cosf(phi), r * sinf(phi), z};
        ls.n = dir;
        ls.p = sp.center + dir * sp.radius;
        ls.Le = s.materials[sp.material].emission;
    } else {
        const Triangle& t = s.triangles[prim - s.n_spheres];
        float su = sqrtf(u1);
        float b0 = 1.0f - su, b1 = u2 * su;
        Vec3 p0 = s.vertices[t.i0], p1 = s.vertices[t.i1], p2 = s.vertices[t.i2];
        ls.p = p0 * (1.0f - b0 - b1) + p1 * b0 + p2 * b1;
        ls.n = normalize(cross(p1 - p0, p2 - p0));
        ls.Le = s.materials[t.material].emission;
    }
    Vec3 d = ls.p - from;
    float dist2 = length2(d);
    float dist = sqrtf(dist2);
    if (dist <= 0.0f) return ls;
    float cosl = fabsf(dot(ls.n, d)) / dist;
    if (cosl <= 1e-6f) return ls;
    ls.pdf_solid = (dist2 / (cosl * area)) / (float)s.n_lights;
    return ls;
}

// Solid-angle light pdf (with selection probability) for a BSDF-sampled hit on light `prim`.
WF_HD float light_pdf_at_hit(const SceneView& s, int prim, float t, Vec3 dir, Vec3 n_out) {
    float cosl = fabsf(dot(n_out, dir));
    if (cosl <= 1e-6f) return 0.0f;
    return ((t * t) / (cosl * prim_area(s, prim))) / (float)s.n_lights;
}

// Steps 1-3 of a bounce. Returns the material type to shade (0..3 for scattering types), or -1 if
// the path terminated (miss, pure emitter, or depth limit). Adds emission to ps.L.
WF_HD int classify(const SceneView& s, PathState& ps, const HitRecord& h, int max_depth,
                   Surface* surf) {
    if (h.prim < 0) {
        ps.L += ps.throughput * s.env;
        return -1;
    }
    *surf = make_surface(s, ps.ray, h);
    const Material& m = s.materials[surf->material];
    if (m.emission.x > 0.0f || m.emission.y > 0.0f || m.emission.z > 0.0f) {
        float w = 1.0f;
        if (!ps.specular && ps.prev_pdf > 0.0f) {
            float pl = light_pdf_at_hit(s, h.prim, h.t, ps.ray.d, surf->n_out);
            w = power_heuristic(ps.prev_pdf, pl);
        }
        ps.L += ps.throughput * m.emission * w;
    }
    if (m.type == kEmissive) return -1;
    if (ps.depth >= max_depth) return -1;
    return m.type;
}

// Russian roulette (step 6). Returns false if the path dies.
WF_HD bool russian_roulette(PathState& ps) {
    if (ps.depth < 3) return true;
    float q = clampf(max_component(ps.throughput), 0.05f, 0.95f);
    if (rng(ps.key, (uint32_t)ps.depth, kDimRoulette) >= q) return false;
    ps.throughput = ps.throughput / q;
    return true;
}

WF_HD void set_next_ray(PathState& ps, Vec3 origin, Vec3 dir) {
    ps.ray.o = origin;
    ps.ray.d = dir;
    ps.ray.tmin = kRayEps;
    ps.ray.tmax = kInf;
}

// Lambertian: next-event estimation (shadow ray + contribution, not yet occluded-tested), then
// cosine-sampled scatter, then roulette. Returns true if the path continues.
// *has_shadow is set when a shadow ray must be traced; if it is unoccluded the caller adds
// *contrib to ps.L BEFORE the next bounce's classify.
WF_HD bool shade_lambert(const SceneView& s, PathState& ps, const Surface& f, Ray* shadow,
                         Vec3* contrib, int* has_shadow) {
    const Material& m = s.materials[f.material];
    uint32_t d = (uint32_t)ps.depth;
    *has_shadow = 0;
    Vec3 org = f.p + f.n * kRayEps;
    if (s.n_lights > 0) {
        LightSample ls = sample_light(s, org, ps.key, d);
        if (ls.pdf_solid > 0.0f) {
            Vec3 to = ls.p - org;
            float dist = length(to);
            Vec3 wi = to / dist;
            float cos_s = dot(f.n, wi);
            if (cos_s > 0.0f) {
                float pb = cos_s * kInvPi;
                float w = power_heuristic(ls.pdf_solid, pb);
                *contrib = ps.throughput * m.albedo * ls.Le *
                           (cos_s * kInvPi * w / ls.pdf_solid);
                shadow->o = org; shadow->d = wi;
                shadow->tmin = kRayEps; shadow->tmax = dist * (1.0f - 1e-3f);
                *has_shadow = 1;
            }
        }
    }
    float u1 = rng(ps.key, d, kDimBsdfU), u2 = rng(ps.key, d, kDimBsdfV);
    float r = sqrtf(u1), phi = 2.0f * kPi * u2;
    Vec3 t, b;
    make_basis(f.n, &t, &b);
    float z = sqrtf(fmaxf3(0.0f, 1.0f - u1));
    Vec3 dir = normalize(t * (r * cosf(phi)) + b * (r * sinf(phi)) + f.n * z);
    ps.throughput = ps.throughput * m.albedo;
    ps.specular = 0;
    ps.prev_pdf = dot(dir, f.n) * kInvPi;
    ps.depth += 1;
    set_next_ray(ps, org, dir);
    return russian_roulette(ps);
}

WF_HD bool shade_metal(const SceneView& s, PathState& ps, const Surface& f) {
    const Material& m = s.materials[f.material];
    Vec3 dir = reflect(ps.ray.d, f.n);
    if (dot(dir, f.n) <= 0.0f) return false;
    ps.throughput = ps.throughput * m.albedo;
    ps.specular = 1;
    ps.prev_pdf = 0.0f;
    ps.depth += 1;
    set_next_ray(ps, f.p + f.n * kRayEps, normalize(dir));
    return russian_roulette(ps);
}

// Exact dielectric Fresnel (unpolarised), cos_i > 0 against the facing normal.
WF_HD float fresnel_dielectric(float cos_i, float eta_i, float eta_t) {
    float sin_t = eta_i / eta_t * sqrtf(fmaxf3(0.0f, 1.0f - cos_i * cos_i));
    if (sin_t >= 1.0f) return 1.0f;
    float cos_t = sqrtf(fmaxf3(0.0f, 1.0f - sin_t * sin_t));
    float rpar = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    float rperp = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    return 0.5f * (rpar * rpar + rperp * rperp);
}

WF_HD bool shade_dielectric(const SceneView& s, PathState& ps, const Surface& f) {
    const Material& m = s.materials[f.material];
    float eta_i = f.inside ? m.ior : 1.0f;
    float eta_t = f.inside ? 1.0f : m.ior;
    float cos_i = clampf(-dot(ps.ray.d, f.n), 0.0f, 1.0f);
    float F = fresnel_dielectric(cos_i, eta_i, eta_t);
    float u = rng(ps.key, (uint32_t)ps.depth, kDimDielectric);
    Vec3 org, dir;
    if (u < F) {
        dir = reflect(ps.ray.d, f.n);
        org = f.p + f.n * kRayEps;
    } else {
        float eta = eta_i / eta_t;
        float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
        float cos_t = sqrtf(fmaxf3(0.0f, 1.0f - sin2_t));
        dir = ps.ray.d * eta + f.n * (eta * cos_i - cos_t);
        org = f.p - f.n * kRayEps;
    }
    ps.throughput = ps.throughput * m.albedo;
    ps.specular = 1;
    ps.prev_pdf = 0.0f;
    ps.depth += 1;
    set_next_ray(ps, org, normalize(dir));
    return russian_roulette(ps);
}

// Complete path for one (pixel, sample). Used by the CPU renderer and the megakernel.
WF_HD Vec3 trace_path(const SceneView& s, int px, int py, uint32_t sample, uint32_t seed,
                      int max_depth) {
    PathState ps = init_path(s, px, py, sample, seed);
    for (;;) {
        HitRecord h;
        intersect(s, ps.ray, &h);
        Surface f;
        int mat = classify(s, ps, h, max_depth, &f);
        if (mat < 0) break;
        bool alive = false;
        if (mat == kLambert) {
            Ray sh; Vec3 c; int has;
            alive = shade_lambert(s, ps, f, &sh, &c, &has);
            if (has) {
                HitRecord sh_hit;
                if (!intersect(s, sh, &sh_hit, true)) ps.L += c;
            }
        } else if (mat == kMetal) {
            alive = shade_metal(s, ps, f);
        } else {
            alive = shade_dielectric(s, ps, f);
        }
        if (!alive) break;
    }
    return ps.L;
}

}  // namespace wf
