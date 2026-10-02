// Scene description (host, owning) and SceneView (plain pointers, usable in kernels).
#pragma once
#include <string>
#include <vector>
#include "wf/math.hpp"

namespace wf {

enum MaterialType : int { kLambert = 0, kMetal = 1, kDielectric = 2, kEmissive = 3 };
constexpr int kNumMaterialTypes = 4;

struct Material {
    int type;
    float ior;       // dielectric index of refraction
    float pad0, pad1;
    Vec3 albedo;     // reflectance / tint
    float pad2;
    Vec3 emission;   // radiance, may be non-zero for any type
    float pad3;
};

struct Sphere {
    Vec3 center;
    float radius;
    int material;
    int pad[3];
};

struct Triangle {  // indexed triangle
    int i0, i1, i2;
    int material;
};

struct Camera {
    Vec3 position;
    Vec3 look_at;
    Vec3 up;
    float vfov_deg;
};

// Derived camera for ray generation.
struct CameraRig {
    Vec3 origin;
    Vec3 lower_left;
    Vec3 horizontal;
    Vec3 vertical;
};

struct Light {
    int prim;        // primitive id in the shared primitive space
    float area;
    float pad[2];
};

struct BvhNode {  // 48 bytes, preorder layout: left child of node N is N+1
    Vec3 lo;
    int skip;       // node to visit on a miss or after a leaf; -1 terminates
    Vec3 hi;
    int prim_start; // leaf: first entry in prim_order; internal: -1
    int prim_count; // leaf: >0; internal: 0
    int pad[3];
};

struct HitRecord {
    float t;
    int prim;   // -1 for a miss
    float u, v; // barycentrics for triangles (u,v), unused for spheres
};

// Non-owning view over flat arrays; the same struct is used on the host and in kernels.
struct SceneView {
    const Sphere* spheres;
    const Vec3* vertices;
    const Triangle* triangles;
    const Material* materials;
    const BvhNode* nodes;
    const int* prim_order;   // BVH leaf entries -> primitive ids
    const Light* lights;
    int n_spheres, n_triangles, n_materials, n_nodes, n_lights;
    Vec3 env;                // environment radiance for escaped rays
    CameraRig camera;
    int width, height;
};

struct Scene {
    std::string name;
    std::vector<Sphere> spheres;
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<BvhNode> nodes;
    std::vector<int> prim_order;
    std::vector<Light> lights;
    Vec3 env{0.0f, 0.0f, 0.0f};
    Camera camera{};
    int width = 1280, height = 720;

    int n_prims() const { return (int)spheres.size() + (int)triangles.size(); }
    SceneView view() const;   // requires finalize() to have been called
};

// Computes lights from emissive primitives, builds the BVH, derives the camera rig.
// Throws std::runtime_error on invalid input (see validate_scene).
void finalize_scene(Scene& scene);

// Returns an empty string if valid, otherwise a human-readable reason (bad material index,
// out-of-range vertex index, non-finite vertex, degenerate sphere radius, no primitives).
std::string validate_scene(const Scene& scene);

// Procedural scenes: "cornell", "divergence", "trianglehv", "furnace".
// Throws std::invalid_argument for an unknown name. Result is already finalized.
Scene make_scene(const std::string& name, int width, int height);

// Wavefront OBJ (v, vn ignored, f with 3+ indices, optional "usemtl" ignored; one Lambertian
// material, grey 0.7). Throws std::runtime_error with a message that names the file and the
// line for: unreadable file, malformed numbers, face index out of range, no faces.
// The returned scene is finalized. Line numbers are 1-based and count comment and blank lines.
// Numbers are parsed strictly ("2abc" is an error). For a missing/empty/no-face file the message
// names the file only. finalize_scene leaves nodes and lights empty if it throws.
Scene load_obj(const std::string& path, int width, int height);

// 64-bit FNV-1a over the canonical bytes of the scene (spheres, vertices, triangles,
// materials, env, camera, resolution). Same scene -> same hash on every run.
uint64_t scene_hash(const Scene& scene);

// Pinhole rig: w = normalize(position - look_at), u = normalize(cross(up, w)), v = cross(w, u),
// h = tan(vfov/2), horizontal = 2*h*aspect*u, vertical = 2*h*v,
// lower_left = origin - horizontal/2 - vertical/2 - w.
WF_HD CameraRig make_camera_rig(const Camera& cam, int width, int height) {
    float aspect = (float)width / (float)height;
    float h = tanf(cam.vfov_deg * 0.5f * kPi / 180.0f);
    Vec3 w = normalize(cam.position - cam.look_at);
    Vec3 u = normalize(cross(cam.up, w));
    Vec3 v = cross(w, u);
    CameraRig r;
    r.origin = cam.position;
    r.horizontal = u * (2.0f * h * aspect);
    r.vertical = v * (2.0f * h);
    r.lower_left = r.origin - r.horizontal * 0.5f - r.vertical * 0.5f - w;
    return r;
}

}  // namespace wf
