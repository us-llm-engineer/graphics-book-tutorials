// Scene module tests: procedural scenes, validation, finalisation, hashing, camera rig.
// Needs src/scenes.cpp (+ bvh_build.cpp) to link.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "wf/bvh.hpp"
#include "wf/path_core.hpp"
#include "wf/scene.hpp"

#include "test_framework.hpp"

using namespace wf;

namespace {

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInfF = std::numeric_limits<float>::infinity();

Material mat(int type, Vec3 albedo, Vec3 emission = Vec3{0.0f, 0.0f, 0.0f}, float ior = 1.5f) {
    Material m{};
    m.type = type; m.ior = ior; m.albedo = albedo; m.emission = emission;
    return m;
}
Sphere sph(Vec3 c, float r, int m) {
    Sphere s{};
    s.center = c; s.radius = r; s.material = m;
    return s;
}
Triangle tri(int a, int b, int c, int m) {
    Triangle t{};
    t.i0 = a; t.i1 = b; t.i2 = c; t.material = m;
    return t;
}

// One grey Lambertian material, camera on +z looking at the origin, no primitives.
Scene base_scene() {
    Scene s;
    s.name = "hand";
    s.materials.push_back(mat(kLambert, Vec3{0.5f, 0.5f, 0.5f}));
    s.camera = Camera{Vec3{0.0f, 0.0f, 5.0f}, Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 1.0f, 0.0f}, 45.0f};
    s.width = 16; s.height = 9;
    return s;
}
Scene sphere_scene() {
    Scene s = base_scene();
    s.spheres.push_back(sph(Vec3{0.0f, 0.0f, 0.0f}, 1.0f, 0));
    return s;
}
Scene tri_scene() {
    Scene s = base_scene();
    s.vertices = {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}};
    s.triangles.push_back(tri(0, 1, 2, 0));
    return s;
}

template <class E, class F>
bool throws_as(F f, std::string* msg = nullptr) {
    try { f(); } catch (const E& e) { if (msg) *msg = e.what(); return true; } catch (...) { return false; }
    return false;
}

const Scene& cached(const std::string& name) {
    static std::map<std::string, Scene> cache;
    auto it = cache.find(name);
    if (it == cache.end()) it = cache.emplace(name, make_scene(name, 64, 36)).first;
    return it->second;
}

const char* const kNames[4] = {"cornell", "divergence", "trianglehv", "furnace"};

bool emissive(const Material& m) { return m.emission.x > 0.0f || m.emission.y > 0.0f || m.emission.z > 0.0f; }

// Rejection helper: validate_scene reports a reason, finalize_scene throws runtime_error and
// leaves derived data (BVH, lights) untouched.
void expect_rejected(const Scene& bad, const char* what) {
    std::string reason = validate_scene(bad);
    if (reason.empty()) std::printf("  %s: validate_scene accepted an invalid scene\n", what);
    CHECK(!reason.empty());
    Scene copy = bad;
    bool threw = throws_as<std::runtime_error>([&] { finalize_scene(copy); });
    if (!threw) std::printf("  %s: finalize_scene did not throw std::runtime_error\n", what);
    CHECK(threw);
    CHECK(copy.nodes.empty());   // rejection must not half-apply
    CHECK(copy.lights.empty());
}

}  // namespace

// ---------------------------------------------------------------- make_scene

TEST(make_scene_returns_valid_finalized_scene_for_every_name) {
    // Given each documented name / When make_scene / Then validation and BVH validation are clean.
    for (const char* name : kNames) {
        const Scene& s = cached(name);
        std::string v = validate_scene(s);
        if (!v.empty()) std::printf("  %s: validate_scene: %s\n", name, v.c_str());
        CHECK(v.empty());
        std::string b = validate_bvh(s);
        if (!b.empty()) std::printf("  %s: validate_bvh: %s\n", name, b.c_str());
        CHECK(b.empty());
        CHECK(s.n_prims() > 0);
        CHECK(!s.nodes.empty());
        SceneView view = s.view();
        CHECK(view.n_nodes == (int)s.nodes.size());   // view reflects the finalised arrays
        CHECK(view.n_spheres == (int)s.spheres.size());
        CHECK(view.n_triangles == (int)s.triangles.size());
        CHECK(view.n_lights == (int)s.lights.size());
    }
}

TEST(make_scene_lights_are_exactly_the_emissive_primitives) {
    // Oracle: independent scan of materials. A light is any primitive whose material has non-zero
    // emission; prim id of triangle j is n_spheres + j; area is 4 pi r^2 / half |e1 x e2|.
    for (const char* name : kNames) {
        const Scene& s = cached(name);
        std::vector<int> expect;
        std::map<int, double> area;
        for (int i = 0; i < (int)s.spheres.size(); ++i)
            if (emissive(s.materials[s.spheres[i].material])) {
                expect.push_back(i);
                area[i] = 4.0 * 3.14159265358979 * (double)s.spheres[i].radius * s.spheres[i].radius;
            }
        for (int j = 0; j < (int)s.triangles.size(); ++j) {
            const Triangle& t = s.triangles[j];
            if (!emissive(s.materials[t.material])) continue;
            int id = (int)s.spheres.size() + j;
            expect.push_back(id);
            Vec3 e1 = s.vertices[t.i1] - s.vertices[t.i0], e2 = s.vertices[t.i2] - s.vertices[t.i0];
            area[id] = 0.5 * length(cross(e1, e2));
        }
        std::vector<int> got;
        for (const Light& l : s.lights) got.push_back(l.prim);
        std::sort(got.begin(), got.end());
        if (got != expect) std::printf("  %s: %zu lights, expected %zu\n", name, got.size(), expect.size());
        CHECK(got == expect);
        for (const Light& l : s.lights) {
            auto it = area.find(l.prim);
            if (it != area.end()) CHECK_NEAR(l.area, it->second, 1e-4 * it->second + 1e-9);
        }
    }
    // Emission exists in these three by construction (Cornell light, emissive sphere, furnace shell).
    CHECK(!cached("cornell").lights.empty());
    CHECK(!cached("divergence").lights.empty());
    CHECK(!cached("furnace").lights.empty());
}

TEST(make_scene_unknown_name_throws_invalid_argument) {
    std::string msg;
    CHECK(throws_as<std::invalid_argument>([] { make_scene("no_such_scene", 16, 9); }, &msg));
    CHECK(!msg.empty());
    CHECK(throws_as<std::invalid_argument>([] { make_scene("", 16, 9); }));
}

TEST(make_scene_propagates_width_and_height_to_scene_and_camera_rig) {
    // Non-square, non-default sizes so a hard-coded 1280x720 or a swapped pair is caught.
    for (const char* name : {"cornell", "furnace"}) {
        Scene s = make_scene(name, 48, 20);
        CHECK(s.width == 48);
        CHECK(s.height == 20);
        SceneView v = s.view();
        CHECK(v.width == 48);
        CHECK(v.height == 20);
        // Oracle: the rig is make_camera_rig of the scene camera at that aspect ratio.
        CameraRig r = make_camera_rig(s.camera, 48, 20);
        CHECK_NEAR(v.camera.origin.x, r.origin.x, 1e-5);
        CHECK_NEAR(v.camera.origin.y, r.origin.y, 1e-5);
        CHECK_NEAR(v.camera.origin.z, r.origin.z, 1e-5);
        CHECK_NEAR(v.camera.horizontal.x, r.horizontal.x, 1e-4);
        CHECK_NEAR(v.camera.horizontal.y, r.horizontal.y, 1e-4);
        CHECK_NEAR(v.camera.horizontal.z, r.horizontal.z, 1e-4);
        CHECK_NEAR(v.camera.vertical.x, r.vertical.x, 1e-4);
        CHECK_NEAR(v.camera.vertical.y, r.vertical.y, 1e-4);
        CHECK_NEAR(v.camera.vertical.z, r.vertical.z, 1e-4);
        CHECK_NEAR(v.camera.lower_left.x, r.lower_left.x, 1e-4);
        CHECK_NEAR(v.camera.lower_left.y, r.lower_left.y, 1e-4);
        CHECK_NEAR(v.camera.lower_left.z, r.lower_left.z, 1e-4);
    }
}

TEST(divergence_scene_uses_at_least_four_material_types) {
    const Scene& s = cached("divergence");
    std::set<int> types;
    for (const Sphere& sp : s.spheres) types.insert(s.materials[sp.material].type);
    for (const Triangle& t : s.triangles) types.insert(s.materials[t.material].type);
    CHECK((int)types.size() >= 4);   // design: Lambertian, metal, dielectric, emissive
}

TEST(divergence_neighbouring_pixels_often_hit_different_material_types) {
    // Given a 64x36 grid of pixel-centre camera rays / When intersect + prim_material
    // Then >= 30% of horizontally and vertically adjacent pixel pairs hit different material TYPES
    // (a miss never counts as "different"; both pixels must hit).
    const Scene& s = cached("divergence");
    SceneView v = s.view();
    const int W = 64, H = 36;
    std::vector<int> type(W * H, -1);
    for (int py = 0; py < H; ++py)
        for (int px = 0; px < W; ++px) {
            // Independent ray construction from the rig (pixel centre, row 0 at the top).
            float u = ((float)px + 0.5f) / (float)W;
            float w = ((float)(H - 1 - py) + 0.5f) / (float)H;
            Vec3 target = v.camera.lower_left + v.camera.horizontal * u + v.camera.vertical * w;
            Ray r;
            r.o = v.camera.origin; r.d = normalize(target - v.camera.origin);
            r.tmin = 1e-4f; r.tmax = kInf;
            HitRecord h;
            if (intersect(v, r, &h)) type[py * W + px] = v.materials[prim_material(v, h.prim)].type;
        }
    int pairs = 0, diff = 0, hits = 0;
    for (int i = 0; i < W * H; ++i) hits += type[i] >= 0;
    for (int py = 0; py < H; ++py)
        for (int px = 0; px < W; ++px) {
            if (px + 1 < W) { ++pairs; diff += (type[py * W + px] >= 0 && type[py * W + px + 1] >= 0 && type[py * W + px] != type[py * W + px + 1]); }
            if (py + 1 < H) { ++pairs; diff += (type[py * W + px] >= 0 && type[(py + 1) * W + px] >= 0 && type[py * W + px] != type[(py + 1) * W + px]); }
        }
    double frac = (double)diff / (double)pairs;
    std::printf("  divergence: %d/%d pixels hit, %.3f of adjacent pairs differ in material type\n", hits, W * H, frac);
    CHECK(hits > W * H / 2);   // the camera actually looks at the scene
    CHECK(frac >= 0.30);
}

TEST(trianglehv_has_at_least_100000_triangles) {
    const Scene& s = cached("trianglehv");
    CHECK(s.triangles.size() >= 100000u);
    CHECK(!s.vertices.empty());
}

TEST(furnace_camera_sees_a_closed_scene) {
    // Closed furnace: every primary ray must hit something (otherwise the oracle Le/(1-rho) leaks).
    const Scene& s = cached("furnace");
    SceneView v = s.view();
    int misses = 0;
    for (int py = 0; py < s.height; ++py)
        for (int px = 0; px < s.width; ++px) {
            float u = ((float)px + 0.5f) / (float)s.width;
            float w = ((float)(s.height - 1 - py) + 0.5f) / (float)s.height;
            Vec3 target = v.camera.lower_left + v.camera.horizontal * u + v.camera.vertical * w;
            Ray r{v.camera.origin, 1e-4f, normalize(target - v.camera.origin), kInf};
            HitRecord h;
            misses += !intersect(v, r, &h);
        }
    CHECK(misses == 0);
    CHECK(s.env.x == 0.0f && s.env.y == 0.0f && s.env.z == 0.0f);
}

// ---------------------------------------------------------------- scene_hash

TEST(scene_hash_is_stable_across_calls_and_across_builds) {
    const Scene& a = cached("divergence");
    CHECK(scene_hash(a) == scene_hash(a));
    Scene b = make_scene("divergence", 64, 36);
    CHECK(scene_hash(a) == scene_hash(b));   // procedural builds are deterministic
}

TEST(scene_hash_identical_for_two_separately_built_equal_scenes) {
    Scene a = sphere_scene(), b = sphere_scene();
    CHECK(scene_hash(a) == scene_hash(b));
    finalize_scene(a);
    finalize_scene(b);
    CHECK(scene_hash(a) == scene_hash(b));
}

TEST(scene_hash_ignores_derived_acceleration_data) {
    // The hash covers spheres, vertices, triangles, materials, env, camera, resolution: not nodes.
    Scene a = sphere_scene();
    uint64_t before = scene_hash(a);
    finalize_scene(a);
    CHECK(scene_hash(a) == before);
}

TEST(scene_hash_changes_when_one_sphere_radius_changes) {
    Scene a = make_scene("divergence", 64, 36);
    CHECK(!a.spheres.empty());
    uint64_t h0 = scene_hash(a);
    a.spheres[a.spheres.size() / 2].radius *= 1.01f;
    CHECK(scene_hash(a) != h0);
}

TEST(scene_hash_changes_when_one_vertex_changes) {
    Scene a = make_scene("trianglehv", 64, 36);
    CHECK(!a.vertices.empty());
    uint64_t h0 = scene_hash(a);
    a.vertices[a.vertices.size() / 2].y += 0.5f;
    CHECK(scene_hash(a) != h0);
}

TEST(scene_hash_changes_when_environment_changes) {
    Scene a = make_scene("cornell", 64, 36);
    uint64_t h0 = scene_hash(a);
    a.env = Vec3{0.1f, 0.0f, 0.0f};
    CHECK(scene_hash(a) != h0);
}

TEST(scene_hash_changes_when_camera_changes) {
    Scene a = make_scene("cornell", 64, 36);
    uint64_t h0 = scene_hash(a);
    Scene b = a;
    b.camera.position.x += 0.1f;
    CHECK(scene_hash(b) != h0);
    Scene c = a;
    c.camera.vfov_deg += 1.0f;
    CHECK(scene_hash(c) != h0);
}

TEST(scene_hash_changes_with_resolution_and_materials_and_triangle_indices) {
    Scene a = tri_scene();
    a.vertices.push_back(Vec3{1, 1, 0});
    uint64_t h0 = scene_hash(a);
    Scene r = a; r.width += 1;
    CHECK(scene_hash(r) != h0);
    Scene m = a; m.materials[0].albedo.x = 0.6f;
    CHECK(scene_hash(m) != h0);
    Scene t = a; t.triangles[0].i2 = 3;   // different but valid triangle
    CHECK(scene_hash(t) != h0);
}

TEST(scene_hash_depends_on_primitive_order) {
    // Canonical bytes are sequential: swapping two different spheres changes the hash.
    Scene a = make_scene("divergence", 64, 36);
    CHECK(a.spheres.size() >= 2);
    size_t i = 0, j = a.spheres.size() - 1;
    CHECK(a.spheres[i].center.x != a.spheres[j].center.x || a.spheres[i].center.z != a.spheres[j].center.z);
    uint64_t h0 = scene_hash(a);
    std::swap(a.spheres[i], a.spheres[j]);
    CHECK(scene_hash(a) != h0);
}

// ---------------------------------------------------------------- validate_scene / finalize_scene

TEST(validate_scene_accepts_minimal_valid_scenes_including_boundaries) {
    // These baselines pin the rejection tests below: each rejection differs by exactly one defect.
    CHECK(validate_scene(sphere_scene()).empty());
    CHECK(validate_scene(tri_scene()).empty());
    Scene tiny = sphere_scene(); tiny.spheres[0].radius = 1e-3f;
    CHECK(validate_scene(tiny).empty());                       // tiny but positive radius is legal
    Scene last_mat = sphere_scene();
    last_mat.materials.push_back(mat(kMetal, Vec3{0.9f, 0.9f, 0.9f}));
    last_mat.spheres[0].material = 1;                          // index == size-1 is legal
    CHECK(validate_scene(last_mat).empty());
    Scene last_vtx = tri_scene();
    last_vtx.vertices.push_back(Vec3{1, 1, 0});
    last_vtx.triangles[0].i2 = 3;                              // index == size-1 is legal
    CHECK(validate_scene(last_vtx).empty());
}

TEST(validate_scene_rejects_sphere_material_index_equal_to_material_count) {
    Scene s = sphere_scene(); s.spheres[0].material = 1;   // off by one: only material 0 exists
    expect_rejected(s, "sphere material == size");
}

TEST(validate_scene_rejects_negative_sphere_material_index) {
    Scene s = sphere_scene(); s.spheres[0].material = -1;
    expect_rejected(s, "sphere material -1");
}

TEST(validate_scene_rejects_triangle_material_index_out_of_range) {
    Scene s = tri_scene(); s.triangles[0].material = 5;
    expect_rejected(s, "triangle material 5");
}

TEST(validate_scene_rejects_vertex_index_equal_to_vertex_count) {
    Scene s = tri_scene(); s.triangles[0].i1 = 3;   // off by one: vertices are 0..2
    expect_rejected(s, "vertex index == size");
}

TEST(validate_scene_rejects_negative_vertex_index) {
    Scene s = tri_scene(); s.triangles[0].i0 = -1;
    expect_rejected(s, "vertex index -1");
}

TEST(validate_scene_rejects_nan_vertex) {
    Scene s = tri_scene(); s.vertices[1].y = kNaN;
    expect_rejected(s, "NaN vertex");
}

TEST(validate_scene_rejects_infinite_vertex) {
    Scene s = tri_scene(); s.vertices[2].x = kInfF;
    expect_rejected(s, "inf vertex");
}

TEST(validate_scene_rejects_zero_sphere_radius) {
    Scene s = sphere_scene(); s.spheres[0].radius = 0.0f;
    expect_rejected(s, "radius 0");
}

TEST(validate_scene_rejects_negative_sphere_radius) {
    Scene s = sphere_scene(); s.spheres[0].radius = -1.0f;
    expect_rejected(s, "radius -1");
}

TEST(validate_scene_rejects_nan_sphere_radius) {
    Scene s = sphere_scene(); s.spheres[0].radius = kNaN;
    expect_rejected(s, "radius NaN");
}

TEST(validate_scene_rejects_scene_without_primitives) {
    Scene s = base_scene();   // has a material, camera, but nothing to render
    expect_rejected(s, "no primitives");
    Scene vertices_only = base_scene();
    vertices_only.vertices = {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}};   // vertices are not primitives
    expect_rejected(vertices_only, "vertices but no triangles");
}

TEST(validate_scene_rejects_primitive_when_there_are_no_materials) {
    Scene s = sphere_scene(); s.materials.clear();
    expect_rejected(s, "no materials");
}

TEST(finalize_scene_builds_valid_bvh_and_rig_for_a_valid_scene) {
    Scene s = sphere_scene();
    finalize_scene(s);
    CHECK(s.nodes.size() == 1u);   // a single primitive is a single leaf
    CHECK(validate_bvh(s).empty());
    CHECK(s.lights.empty());       // nothing emissive
    SceneView v = s.view();
    CameraRig r = make_camera_rig(s.camera, 16, 9);
    CHECK_NEAR(v.camera.lower_left.x, r.lower_left.x, 1e-5);
    CHECK_NEAR(v.camera.horizontal.x, r.horizontal.x, 1e-5);
    CHECK_NEAR(v.camera.vertical.y, r.vertical.y, 1e-5);
    CHECK(v.width == 16 && v.height == 9);
}

TEST(finalize_scene_lights_use_shared_primitive_id_space_with_areas) {
    // Sphere 0 plain, triangle emissive (right triangle legs 3 and 4 => area 6).
    Scene s = base_scene();
    s.materials.push_back(mat(kEmissive, Vec3{0, 0, 0}, Vec3{5, 5, 5}));
    s.spheres.push_back(sph(Vec3{0, 0, 0}, 1.0f, 0));
    s.vertices = {Vec3{0, 0, 3}, Vec3{3, 0, 3}, Vec3{0, 4, 3}};
    s.triangles.push_back(tri(0, 1, 2, 1));
    finalize_scene(s);
    CHECK(s.lights.size() == 1u);
    if (s.lights.size() == 1) {
        CHECK(s.lights[0].prim == 1);   // n_spheres (1) + triangle index (0)
        CHECK_NEAR(s.lights[0].area, 6.0, 1e-4);
    }
}

TEST(finalize_scene_counts_emissive_lambertian_sphere_with_one_nonzero_channel) {
    // Design: any material may carry emission; any non-zero channel makes the primitive a light.
    Scene s = base_scene();
    s.materials.push_back(mat(kLambert, Vec3{0.5f, 0.5f, 0.5f}, Vec3{0.0f, 0.5f, 0.0f}));
    s.spheres.push_back(sph(Vec3{0, 0, 0}, 2.0f, 1));
    s.spheres.push_back(sph(Vec3{10, 0, 0}, 1.0f, 0));
    finalize_scene(s);
    CHECK(s.lights.size() == 1u);
    if (s.lights.size() == 1) {
        CHECK(s.lights[0].prim == 0);
        CHECK_NEAR(s.lights[0].area, 16.0 * 3.14159265358979, 1e-3);   // 4 pi r^2, r = 2
    }
}

TEST(finalize_scene_throws_runtime_error_on_empty_scene_without_side_effects) {
    Scene s = base_scene();
    CHECK(throws_as<std::runtime_error>([&] { finalize_scene(s); }));
    CHECK(s.nodes.empty());
    CHECK(s.lights.empty());
}

// ---------------------------------------------------------------- make_camera_rig

namespace {
Vec3 ray_dir(const CameraRig& r, float u, float v) {   // u,v in [0,1], unnormalised target - origin
    return r.lower_left + r.horizontal * u + r.vertical * v - r.origin;
}
}  // namespace

TEST(camera_rig_centre_pixel_looks_down_minus_z) {
    // Hand computation: w=(0,0,1), u=(1,0,0), v=(0,1,0), h=tan 45 = 1 =>
    // lower_left=(-1,-1,4), horizontal=(2,0,0), vertical=(0,2,0); centre target (0,0,4).
    Camera c{Vec3{0, 0, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 90.0f};
    CameraRig r = make_camera_rig(c, 100, 100);
    Vec3 d = normalize(ray_dir(r, 0.5f, 0.5f));
    CHECK_NEAR(d.x, 0.0, 1e-6);
    CHECK_NEAR(d.y, 0.0, 1e-6);
    CHECK_NEAR(d.z, -1.0, 1e-6);
    CHECK_NEAR(r.origin.z, 5.0, 1e-6);
}

TEST(camera_rig_corner_directions_have_45_degree_extents_with_correct_signs) {
    Camera c{Vec3{0, 0, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 90.0f};
    CameraRig r = make_camera_rig(c, 100, 100);
    const double s3 = 1.0 / std::sqrt(3.0);
    Vec3 ll = normalize(ray_dir(r, 0.0f, 0.0f));   // bottom-left
    CHECK_NEAR(ll.x, -s3, 1e-5); CHECK_NEAR(ll.y, -s3, 1e-5); CHECK_NEAR(ll.z, -s3, 1e-5);
    Vec3 ur = normalize(ray_dir(r, 1.0f, 1.0f));   // top-right
    CHECK_NEAR(ur.x, s3, 1e-5); CHECK_NEAR(ur.y, s3, 1e-5); CHECK_NEAR(ur.z, -s3, 1e-5);
    // Half-angles: atan(|x| / -z) == 45 degrees horizontally and vertically.
    const double pi = 3.14159265358979;
    CHECK_NEAR(std::atan2((double)ur.x, -(double)ur.z) * 180.0 / pi, 45.0, 1e-3);
    CHECK_NEAR(std::atan2((double)ur.y, -(double)ur.z) * 180.0 / pi, 45.0, 1e-3);
}

TEST(camera_rig_aspect_ratio_widens_horizontal_extent_only) {
    // 200x100, vfov 90: h=1, aspect 2 => horizontal = (4,0,0), vertical = (0,2,0). Square images hide
    // an aspect that is swapped or ignored.
    Camera c{Vec3{0, 0, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 90.0f};
    CameraRig r = make_camera_rig(c, 200, 100);
    CHECK_NEAR(r.horizontal.x, 4.0, 1e-5);
    CHECK_NEAR(r.vertical.y, 2.0, 1e-5);
    Vec3 right = ray_dir(r, 1.0f, 0.5f);
    CHECK_NEAR(right.x / -right.z, 2.0, 1e-5);   // tan of the horizontal half angle
    Vec3 top = ray_dir(r, 0.5f, 1.0f);
    CHECK_NEAR(top.y / -top.z, 1.0, 1e-5);
}

TEST(camera_rig_looking_along_plus_x_has_plus_z_on_the_right) {
    // Camera at origin looking at +x, up y: w=(-1,0,0), u=cross(y,w)=(0,0,1), v=(0,1,0).
    // vfov 60 => h=tan30; aspect 1.
    Camera c{Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}, 60.0f};
    CameraRig r = make_camera_rig(c, 50, 50);
    Vec3 centre = normalize(ray_dir(r, 0.5f, 0.5f));
    CHECK_NEAR(centre.x, 1.0, 1e-6);
    CHECK_NEAR(centre.y, 0.0, 1e-6);
    CHECK_NEAR(centre.z, 0.0, 1e-6);
    const double h = std::tan(30.0 * 3.14159265358979 / 180.0);
    CHECK_NEAR(r.horizontal.z, 2.0 * h, 1e-5);   // +z is screen-right
    CHECK_NEAR(r.horizontal.x, 0.0, 1e-6);
    CHECK_NEAR(r.vertical.y, 2.0 * h, 1e-5);
}
