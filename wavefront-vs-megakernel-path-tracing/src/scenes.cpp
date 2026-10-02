#include "wf/scene.hpp"
#include <stdexcept>

namespace wf {

// Scene::view() returns a SceneView (non-owning pointers)
SceneView Scene::view() const {
    SceneView sv;
    sv.spheres = spheres.empty() ? nullptr : spheres.data();
    sv.vertices = vertices.empty() ? nullptr : vertices.data();
    sv.triangles = triangles.empty() ? nullptr : triangles.data();
    sv.materials = materials.empty() ? nullptr : materials.data();
    sv.nodes = nodes.empty() ? nullptr : nodes.data();
    sv.prim_order = prim_order.empty() ? nullptr : prim_order.data();
    sv.lights = lights.empty() ? nullptr : lights.data();
    sv.n_spheres = (int)spheres.size();
    sv.n_triangles = (int)triangles.size();
    sv.n_materials = (int)materials.size();
    sv.n_nodes = (int)nodes.size();
    sv.n_lights = (int)lights.size();
    sv.env = env;
    sv.camera = make_camera_rig(camera, width, height);
    sv.width = width;
    sv.height = height;
    return sv;
}

// Cornell box: left and right walls are spheres, back/front/floor/ceiling/light are triangles
Scene make_scene(const std::string& name, int width, int height) {
    if (name == "cornell") {
        Scene s;
        s.name = "cornell";
        s.width = width;
        s.height = height;

        // Materials: Lambertian, metal, dielectric, emissive
        Material lambert{};
        lambert.type = kLambert;
        lambert.albedo = Vec3{0.7f, 0.7f, 0.7f};
        lambert.emission = Vec3{0, 0, 0};
        s.materials.push_back(lambert);

        // Red wall (left sphere)
        Material red{};
        red.type = kLambert;
        red.albedo = Vec3{0.7f, 0.2f, 0.2f};
        red.emission = Vec3{0, 0, 0};
        s.materials.push_back(red);

        // Green wall (right sphere)
        Material green{};
        green.type = kLambert;
        green.albedo = Vec3{0.2f, 0.7f, 0.2f};
        green.emission = Vec3{0, 0, 0};
        s.materials.push_back(green);

        // Light (emissive)
        Material light{};
        light.type = kLambert;
        light.albedo = Vec3{0.7f, 0.7f, 0.7f};
        light.emission = Vec3{5.0f, 5.0f, 5.0f};
        s.materials.push_back(light);

        // Left red sphere (x = -1.5, y = 0, z = -2, radius = 0.3)
        Sphere left{};
        left.center = Vec3{-1.5f, 0.0f, -2.0f};
        left.radius = 0.3f;
        left.material = 1;
        s.spheres.push_back(left);

        // Right green sphere (x = 1.5, y = 0, z = -2, radius = 0.3)
        Sphere right{};
        right.center = Vec3{1.5f, 0.0f, -2.0f};
        right.radius = 0.3f;
        right.material = 2;
        s.spheres.push_back(right);

        // Back wall triangle at z = -4
        s.vertices.push_back(Vec3{-3.0f, -2.0f, -4.0f}); // 0
        s.vertices.push_back(Vec3{3.0f, -2.0f, -4.0f});   // 1
        s.vertices.push_back(Vec3{3.0f, 2.0f, -4.0f});    // 2
        s.vertices.push_back(Vec3{-3.0f, 2.0f, -4.0f});   // 3
        Triangle back1{0, 1, 2, 0};
        Triangle back2{0, 2, 3, 0};
        s.triangles.push_back(back1);
        s.triangles.push_back(back2);

        // Floor at y = -2
        s.vertices.push_back(Vec3{-3.0f, -2.0f, 0.0f});   // 4
        s.vertices.push_back(Vec3{3.0f, -2.0f, 0.0f});    // 5
        Triangle floor1{0, 4, 5};
        Triangle floor2{0, 5, 1};
        floor1.material = 0;
        floor2.material = 0;
        s.triangles.push_back(floor1);
        s.triangles.push_back(floor2);

        // Ceiling at y = 2
        Triangle ceil1{3, 2, 6};  // Will need more vertices
        Triangle ceil2{3, 6, 7};
        s.vertices.push_back(Vec3{-3.0f, 2.0f, 0.0f});    // 6
        s.vertices.push_back(Vec3{3.0f, 2.0f, 0.0f});     // 7
        ceil1 = Triangle{3, 2, 7, 0};
        ceil2 = Triangle{3, 7, 6, 0};
        s.triangles.push_back(ceil1);
        s.triangles.push_back(ceil2);

        // Light quad at top (emissive rectangle on ceiling, smaller)
        s.vertices.push_back(Vec3{-1.0f, 1.99f, -2.5f});  // 8
        s.vertices.push_back(Vec3{1.0f, 1.99f, -2.5f});   // 9
        s.vertices.push_back(Vec3{1.0f, 1.99f, -1.5f});   // 10
        s.vertices.push_back(Vec3{-1.0f, 1.99f, -1.5f});  // 11
        Triangle light1{8, 9, 10, 3};
        Triangle light2{8, 10, 11, 3};
        s.triangles.push_back(light1);
        s.triangles.push_back(light2);

        // Camera
        s.camera.position = Vec3{0.0f, 0.0f, 2.5f};
        s.camera.look_at = Vec3{0.0f, 0.0f, -2.0f};
        s.camera.up = Vec3{0.0f, 1.0f, 0.0f};
        s.camera.vfov_deg = 40.0f;

        s.env = Vec3{0.0f, 0.0f, 0.0f};
        finalize_scene(s);
        return s;
    }

    if (name == "divergence") {
        Scene s;
        s.name = "divergence";
        s.width = width;
        s.height = height;

        // Material 0: Lambertian grey
        Material lambert0{};
        lambert0.type = kLambert;
        lambert0.albedo = Vec3{0.5f, 0.5f, 0.5f};
        lambert0.emission = Vec3{0, 0, 0};
        s.materials.push_back(lambert0);

        // Material 1: Another Lambertian with different albedo (for variety, though same type)
        Material lambert1{};
        lambert1.type = kLambert;
        lambert1.albedo = Vec3{0.7f, 0.7f, 0.7f};
        lambert1.emission = Vec3{0, 0, 0};
        s.materials.push_back(lambert1);

        // Material 2: Lambertian with red tint
        Material lambert2{};
        lambert2.type = kLambert;
        lambert2.albedo = Vec3{0.7f, 0.3f, 0.3f};
        lambert2.emission = Vec3{0, 0, 0};
        s.materials.push_back(lambert2);

        // Material 3: Lambertian with blue tint
        Material lambert3{};
        lambert3.type = kLambert;
        lambert3.albedo = Vec3{0.3f, 0.3f, 0.7f};
        lambert3.emission = Vec3{0, 0, 0};
        s.materials.push_back(lambert3);

        // Material 4: Metal
        Material metal{};
        metal.type = kMetal;
        metal.albedo = Vec3{0.9f, 0.9f, 0.9f};
        metal.emission = Vec3{0, 0, 0};
        s.materials.push_back(metal);

        // Material 5: Dielectric
        Material dielectric{};
        dielectric.type = kDielectric;
        dielectric.ior = 1.5f;
        dielectric.albedo = Vec3{1.0f, 1.0f, 1.0f};
        dielectric.emission = Vec3{0, 0, 0};
        s.materials.push_back(dielectric);

        // Material 6: Emissive
        Material emissive{};
        emissive.type = kEmissive;
        emissive.albedo = Vec3{0.0f, 0.0f, 0.0f};
        emissive.emission = Vec3{1.0f, 1.0f, 1.0f};
        s.materials.push_back(emissive);

        // Ground plane (large flat rectangle at y = -1), Lambertian
        s.vertices.push_back(Vec3{-15.0f, -1.0f, -15.0f});
        s.vertices.push_back(Vec3{15.0f, -1.0f, -15.0f});
        s.vertices.push_back(Vec3{15.0f, -1.0f, 15.0f});
        s.vertices.push_back(Vec3{-15.0f, -1.0f, 15.0f});
        s.triangles.push_back(Triangle{0, 1, 2, 0});  // Ground is Lambertian (material 0)
        s.triangles.push_back(Triangle{0, 2, 3, 0});

        // Maximum-density 30x30 grid of small spheres with strong material alternation
        float spacing = 0.33f;
        float radius = 0.16f;
        for (int i = 0; i < 30; ++i) {
            for (int j = 0; j < 30; ++j) {
                Sphere sp{};
                sp.center = Vec3{-4.95f + i * spacing, 0.2f, -4.95f + j * spacing};
                sp.radius = radius;
                // Checkerboard pattern: alternating material types ensures divergence
                int parity = (i + j) & 1;
                if (parity == 0) {
                    // Even: cycle through Lambertian materials (0-3)
                    sp.material = (i + j) % 4;
                } else {
                    // Odd: alternate between Metal (4) and Dielectric (5)
                    sp.material = (i & 1) ? 5 : 4;
                }
                s.spheres.push_back(sp);
            }
        }

        // Small light sphere (emissive) at top center
        Sphere light{};
        light.center = Vec3{0.0f, 5.0f, 0.0f};
        light.radius = 0.2f;
        light.material = 6;
        s.spheres.push_back(light);

        s.camera.position = Vec3{0.0f, 2.0f, 6.0f};
        s.camera.look_at = Vec3{0.0f, 0.0f, 0.0f};
        s.camera.up = Vec3{0.0f, 1.0f, 0.0f};
        s.camera.vfov_deg = 45.0f;

        s.env = Vec3{0.05f, 0.05f, 0.05f};
        finalize_scene(s);
        return s;
    }

    if (name == "trianglehv") {
        Scene s;
        s.name = "trianglehv";
        s.width = width;
        s.height = height;

        // Material
        Material lambert{};
        lambert.type = kLambert;
        lambert.albedo = Vec3{0.7f, 0.7f, 0.7f};
        lambert.emission = Vec3{0, 0, 0};
        s.materials.push_back(lambert);

        Material metal{};
        metal.type = kMetal;
        metal.albedo = Vec3{0.9f, 0.9f, 0.9f};
        metal.emission = Vec3{0, 0, 0};
        s.materials.push_back(metal);

        Material light{};
        light.type = kEmissive;
        light.albedo = Vec3{0.0f, 0.0f, 0.0f};
        light.emission = Vec3{3.0f, 3.0f, 3.0f};
        s.materials.push_back(light);

        // Procedural heightfield: increase grid size to get > 100000 triangles
        // Previous: 225x225 = 50625 grid cells = ~101250 triangles
        // Now: 240x240 = 57600 grid cells = ~115200 triangles
        int grid_size = 240;
        float scale = 10.0f;
        float height_scale = 1.0f;

        // Generate vertices
        for (int i = 0; i <= grid_size; ++i) {
            for (int j = 0; j <= grid_size; ++j) {
                float x = (i - grid_size / 2) * scale / (grid_size / 2);
                float z = (j - grid_size / 2) * scale / (grid_size / 2);
                float h = height_scale * (sinf(x) * cosf(z) + 0.5f * (x * x + z * z) / 100.0f);
                s.vertices.push_back(Vec3{x, h, z});
            }
        }

        // Generate triangles (fan triangulation)
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                int v0 = i * (grid_size + 1) + j;
                int v1 = v0 + 1;
                int v2 = v0 + (grid_size + 1);
                int v3 = v2 + 1;
                s.triangles.push_back(Triangle{v0, v1, v2, 0});
                s.triangles.push_back(Triangle{v1, v3, v2, 0});
            }
        }

        // Add a few spheres
        Sphere s1{};
        s1.center = Vec3{0.0f, 2.0f, 0.0f};
        s1.radius = 0.5f;
        s1.material = 1;
        s.spheres.push_back(s1);

        Sphere s2{};
        s2.center = Vec3{3.0f, 1.5f, -2.0f};
        s2.radius = 0.4f;
        s2.material = 1;
        s.spheres.push_back(s2);

        // Light sphere at top
        Sphere light_sphere{};
        light_sphere.center = Vec3{0.0f, 5.0f, 0.0f};
        light_sphere.radius = 0.3f;
        light_sphere.material = 2;
        s.spheres.push_back(light_sphere);

        s.camera.position = Vec3{5.0f, 3.0f, 5.0f};
        s.camera.look_at = Vec3{0.0f, 1.0f, 0.0f};
        s.camera.up = Vec3{0.0f, 1.0f, 0.0f};
        s.camera.vfov_deg = 50.0f;

        s.env = Vec3{0.05f, 0.05f, 0.1f};
        finalize_scene(s);
        return s;
    }

    if (name == "furnace") {
        Scene s;
        s.name = "furnace";
        s.width = width;
        s.height = height;

        // One material: Lambertian with emission 1 and albedo 0.5
        Material furnace_mat{};
        furnace_mat.type = kLambert;
        furnace_mat.albedo = Vec3{0.5f, 0.5f, 0.5f};
        furnace_mat.emission = Vec3{1.0f, 1.0f, 1.0f};
        s.materials.push_back(furnace_mat);

        // One large sphere at origin, radius 5
        Sphere sphere{};
        sphere.center = Vec3{0.0f, 0.0f, 0.0f};
        sphere.radius = 5.0f;
        sphere.material = 0;
        s.spheres.push_back(sphere);

        // Camera at origin looking forward
        s.camera.position = Vec3{0.0f, 0.0f, 0.0f};
        s.camera.look_at = Vec3{0.0f, 0.0f, -1.0f};
        s.camera.up = Vec3{0.0f, 1.0f, 0.0f};
        s.camera.vfov_deg = 90.0f;

        s.env = Vec3{0.0f, 0.0f, 0.0f};
        finalize_scene(s);
        return s;
    }

    throw std::invalid_argument("Unknown scene name: " + name);
}

}  // namespace wf
