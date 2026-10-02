#include "wf/scene.hpp"
#include "wf/bvh.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace wf {

std::string validate_scene(const Scene& scene) {
    // Check no primitives
    if (scene.spheres.empty() && scene.triangles.empty()) {
        return "no primitives";
    }

    // Check materials
    if (scene.materials.empty()) {
        return "no materials";
    }

    // Check sphere material indices and properties
    for (size_t i = 0; i < scene.spheres.size(); ++i) {
        const Sphere& s = scene.spheres[i];
        if (s.material < 0 || s.material >= (int)scene.materials.size()) {
            return "sphere material index out of range";
        }
        if (s.radius <= 0.0f || !is_finite1(s.radius)) {
            return "sphere has invalid radius";
        }
        if (!is_finite3(s.center)) {
            return "sphere centre is non-finite";
        }
    }

    // Check triangle material indices and vertex indices
    for (size_t i = 0; i < scene.triangles.size(); ++i) {
        const Triangle& t = scene.triangles[i];
        if (t.material < 0 || t.material >= (int)scene.materials.size()) {
            return "triangle material index out of range";
        }
        if (t.i0 < 0 || t.i0 >= (int)scene.vertices.size() ||
            t.i1 < 0 || t.i1 >= (int)scene.vertices.size() ||
            t.i2 < 0 || t.i2 >= (int)scene.vertices.size()) {
            return "triangle vertex index out of range";
        }
    }

    // Check vertex finiteness
    for (size_t i = 0; i < scene.vertices.size(); ++i) {
        if (!is_finite3(scene.vertices[i])) {
            return "vertex is non-finite";
        }
    }

    return "";  // valid
}

void finalize_scene(Scene& scene) {
    // Validate first
    std::string err = validate_scene(scene);
    if (!err.empty()) {
        // Clear partial state
        scene.nodes.clear();
        scene.lights.clear();
        throw std::runtime_error(err);
    }

    // Build lights from emissive primitives
    scene.lights.clear();
    for (int i = 0; i < (int)scene.spheres.size(); ++i) {
        const Material& m = scene.materials[scene.spheres[i].material];
        if (m.emission.x > 0.0f || m.emission.y > 0.0f || m.emission.z > 0.0f) {
            const Sphere& s = scene.spheres[i];
            float area = 4.0f * 3.14159265358979323846f * s.radius * s.radius;
            scene.lights.push_back(Light{i, area, {0, 0}});
        }
    }
    for (int i = 0; i < (int)scene.triangles.size(); ++i) {
        const Material& m = scene.materials[scene.triangles[i].material];
        if (m.emission.x > 0.0f || m.emission.y > 0.0f || m.emission.z > 0.0f) {
            const Triangle& t = scene.triangles[i];
            int prim_id = (int)scene.spheres.size() + i;
            const Vec3& p0 = scene.vertices[t.i0];
            const Vec3& p1 = scene.vertices[t.i1];
            const Vec3& p2 = scene.vertices[t.i2];
            Vec3 e1 = p1 - p0;
            Vec3 e2 = p2 - p0;
            float area = 0.5f * length(cross(e1, e2));
            scene.lights.push_back(Light{prim_id, area, {0, 0}});
        }
    }

    // Build BVH
    try {
        build_bvh(scene);
    } catch (...) {
        scene.nodes.clear();
        scene.lights.clear();
        throw;
    }

    // Derive camera rig
    // (the camera rig is computed on-demand in Scene::view())
}

// FNV-1a 64-bit hash
uint64_t scene_hash(const Scene& scene) {
    uint64_t hash = 14695981039346656037ULL;  // FNV-1a offset basis

    auto update = [&](const void* data, size_t len) {
        const uint8_t* bytes = (const uint8_t*)data;
        for (size_t i = 0; i < len; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;  // FNV-1a prime
        }
    };

    // Hash spheres (only actual data, not padding)
    for (const auto& s : scene.spheres) {
        update(&s.center, sizeof(Vec3));
        update(&s.radius, sizeof(float));
        update(&s.material, sizeof(int));
    }

    // Hash vertices
    for (const auto& v : scene.vertices) {
        update(&v, sizeof(Vec3));
    }

    // Hash triangles (excluding padding)
    for (const auto& t : scene.triangles) {
        update(&t.i0, sizeof(int));
        update(&t.i1, sizeof(int));
        update(&t.i2, sizeof(int));
        update(&t.material, sizeof(int));
    }

    // Hash materials (only actual data, not padding)
    for (const auto& m : scene.materials) {
        update(&m.type, sizeof(int));
        update(&m.ior, sizeof(float));
        update(&m.albedo, sizeof(Vec3));
        update(&m.emission, sizeof(Vec3));
    }

    // Hash environment
    update(&scene.env, sizeof(Vec3));

    // Hash camera (only data, not padding)
    update(&scene.camera.position, sizeof(Vec3));
    update(&scene.camera.look_at, sizeof(Vec3));
    update(&scene.camera.up, sizeof(Vec3));
    update(&scene.camera.vfov_deg, sizeof(float));

    // Hash resolution
    update(&scene.width, sizeof(int));
    update(&scene.height, sizeof(int));

    return hash;
}

}  // namespace wf
