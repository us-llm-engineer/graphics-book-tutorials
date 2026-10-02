// invalid input and failure mapping for scenes and OBJ assets.
// Every negative case starts from a valid control and changes exactly one thing; every
// rejection also asserts the documented post-state (nothing half-built).
// Needs src/scenes.cpp (validate_scene, finalize_scene, make_scene, load_obj) and bvh_build.cpp.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "wf/bvh.hpp"
#include "wf/path_core.hpp"
#include "wf/scene.hpp"

using namespace wf;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
namespace fs = std::filesystem;

namespace {

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInfF = std::numeric_limits<float>::infinity();

Material mat(int type, Vec3 emission = Vec3{0, 0, 0}) {
    Material m{};
    m.type = type; m.ior = 1.5f; m.albedo = Vec3{0.5f, 0.5f, 0.5f}; m.emission = emission;
    return m;
}

// Valid control scene: material 0 Lambert, material 1 emissive; sphere 0 (r=0.5, emissive) at the
// origin; triangle (z=-2 plane, area 0.5, emissive). Primitive ids: sphere 0, triangle 1.
Scene valid_scene() {
    Scene s;
    s.name = "control";
    s.materials = {mat(kLambert), mat(kEmissive, Vec3{4, 4, 4})};
    Sphere sp{}; sp.center = Vec3{0, 0, 0}; sp.radius = 0.5f; sp.material = 1;
    s.spheres.push_back(sp);
    s.vertices = {Vec3{-2, 0, -2}, Vec3{-1, 0, -2}, Vec3{-2, 1, -2}};
    s.triangles.push_back(Triangle{0, 1, 2, 1});
    s.camera = Camera{Vec3{0, 0, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 45.0f};
    s.width = 16; s.height = 9;
    return s;
}

// Temporary directory with RAII cleanup; file names contain no digits.
struct TmpDir {
    fs::path dir;
    TmpDir() {
        static int counter = 0;
        dir = fs::temp_directory_path() / ("wf_r11_assets_" + std::to_string((long)getpid()) + "_" + std::to_string(counter++));
        fs::create_directories(dir);
    }
    ~TmpDir() { std::error_code ec; fs::remove_all(dir, ec); }
    std::string write(const std::string& name, const std::string& text) const {
        fs::path p = dir / name;
        std::ofstream(p, std::ios::binary) << text;
        return p.string();
    }
    std::string path(const std::string& name) const { return (dir / name).string(); }
};

// Calls load_obj in exactly one try block; returns the exception message, or "<no throw>".
std::string obj_error(const std::string& path, bool* is_runtime_error = nullptr) {
    if (is_runtime_error) *is_runtime_error = false;
    try {
        Scene s = load_obj(path, 8, 8);
        (void)s;
    } catch (const std::runtime_error& e) {
        if (is_runtime_error) *is_runtime_error = true;
        return e.what();
    } catch (...) {
        return "<other exception type>";
    }
    return "<no throw>";
}

const char* kObjHead =       // 6 lines: comment, vertex, blank, vertex, vertex, normal
    "# header comment\n"     // 1
    "v 0 0 0\n"              // 2
    "\n"                     // 3
    "v 1 0 0\n"              // 4
    "v 0 1 0\n"              // 5
    "vn 0 0 1\n";            // 6   -> the defect line is line 7

}  // namespace

TEST_CASE("validate_scene rejects each single defect and accepts the control", "[assets_loading]") {
    // Given a valid control scene, When exactly one defect is applied, Then validate_scene
    // returns a non-empty reason; the control itself returns an empty string.
    CHECK(validate_scene(valid_scene()).empty());
    using Edit = std::function<void(Scene&)>;
    std::vector<std::pair<std::string, Edit>> defects = {
        {"sphere material index == n_materials", [](Scene& s) { s.spheres[0].material = 2; }},
        {"sphere material index negative", [](Scene& s) { s.spheres[0].material = -1; }},
        {"triangle material index == n_materials", [](Scene& s) { s.triangles[0].material = 2; }},
        {"triangle vertex index == n_vertices", [](Scene& s) { s.triangles[0].i2 = 3; }},
        {"triangle vertex index negative", [](Scene& s) { s.triangles[0].i0 = -1; }},
        {"NaN vertex", [](Scene& s) { s.vertices[1].y = kNaN; }},
        {"infinite vertex", [](Scene& s) { s.vertices[2].x = kInfF; }},
        {"zero radius", [](Scene& s) { s.spheres[0].radius = 0.0f; }},
        {"negative radius", [](Scene& s) { s.spheres[0].radius = -1.0f; }},
        {"NaN radius", [](Scene& s) { s.spheres[0].radius = kNaN; }},
        {"no primitives at all", [](Scene& s) { s.spheres.clear(); s.triangles.clear(); }},
        {"no materials with primitives present", [](Scene& s) { s.materials.clear(); }},
    };
    for (auto& [what, edit] : defects) {
        Scene s = valid_scene();
        edit(s);
        INFO("defect not rejected: " << what);
        CHECK_FALSE(validate_scene(s).empty());
    }
    // A scene whose only primitive is a sphere (no vertices) is valid: unused vertex storage
    // must not be required, and a triangle-only scene is valid too (no spheres required).
    Scene only_sphere = valid_scene();
    only_sphere.triangles.clear(); only_sphere.vertices.clear();
    CHECK(validate_scene(only_sphere).empty());
    Scene only_tri = valid_scene();
    only_tri.spheres.clear();
    CHECK(validate_scene(only_tri).empty());
}

TEST_CASE("finalize_scene throws on invalid scenes, leaves nothing built, and derives lights on valid ones", "[assets_loading]") {
    Scene bad = valid_scene();
    bad.spheres[0].material = 7;   // the single defect
    REQUIRE_THROWS_AS(finalize_scene(bad), std::runtime_error);
    // Documented post-state: nothing half-built.
    CHECK(bad.nodes.empty());
    CHECK(bad.lights.empty());

    Scene empty = valid_scene();
    empty.spheres.clear(); empty.triangles.clear();
    REQUIRE_THROWS_AS(finalize_scene(empty), std::runtime_error);
    CHECK(empty.nodes.empty());
    CHECK(empty.lights.empty());

    // Positive counterpart: same scene minus the defect finalizes; lights are exactly the
    // emissive primitives with analytic areas 4*pi*0.25 = pi and 0.5.
    Scene good = valid_scene();
    REQUIRE_NOTHROW(finalize_scene(good));
    CHECK_FALSE(good.nodes.empty());
    REQUIRE(good.lights.size() == 2);
    for (const Light& l : good.lights) {
        if (l.prim == 0) CHECK_THAT(l.area, WithinAbs(3.14159265, 1e-5));
        else { CHECK(l.prim == 1); CHECK_THAT(l.area, WithinAbs(0.5, 1e-6)); }
    }
    CHECK(good.lights[0].prim != good.lights[1].prim);
    // A scene with no emissive primitive has no lights.
    Scene dark = valid_scene();
    dark.materials[1] = mat(kLambert);
    REQUIRE_NOTHROW(finalize_scene(dark));
    CHECK(dark.lights.empty());
}

TEST_CASE("load_obj reads a triangle and triangulates a quad", "[assets_loading]") {
    TmpDir tmp;
    // Comments, blank lines, vn and usemtl are ignored.
    std::string tri = tmp.write("tri.obj",
        "# a single triangle\nv 0 0 0\nv 1 0 0\n\nv 0 1 0\nvn 0 0 1\nusemtl ignored\nf 1 2 3\n");
    Scene t = load_obj(tri, 12, 7);
    REQUIRE(t.vertices.size() == 3);
    REQUIRE(t.triangles.size() == 1);
    CHECK(t.spheres.empty());
    CHECK(t.triangles[0].i0 == 0); CHECK(t.triangles[0].i1 == 1); CHECK(t.triangles[0].i2 == 2);  // 1-based file -> 0-based
    CHECK_THAT(t.vertices[1].x, WithinAbs(1.0, 0.0));
    CHECK_FALSE(t.nodes.empty());                       // returned scene is finalized
    CHECK(validate_scene(t).empty());
    // Behaviour check through the BVH: a ray from above hits primitive 0 at t = 2.
    {
        SceneView v = t.view();
        HitRecord h{};
        REQUIRE(intersect(v, Ray{Vec3{0.25f, 0.25f, 2.0f}, 0.0f, Vec3{0, 0, -1}, kInf}, &h));
        CHECK(h.prim == 0);
        CHECK_THAT(h.t, WithinAbs(2.0, 1e-5));
    }

    // Unit square quad: must become two triangles that cover all four vertices with total area 1
    // (oracle: area of the unit square), regardless of which diagonal is chosen.
    std::string quad = tmp.write("quad.obj", "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n");
    Scene q = load_obj(quad, 8, 8);
    REQUIRE(q.vertices.size() == 4);
    REQUIRE(q.triangles.size() == 2);
    double area = 0;
    std::vector<int> used(4, 0);
    for (const Triangle& tr : q.triangles) {
        REQUIRE(tr.i0 >= 0); REQUIRE(tr.i0 < 4); REQUIRE(tr.i1 >= 0); REQUIRE(tr.i1 < 4);
        REQUIRE(tr.i2 >= 0); REQUIRE(tr.i2 < 4);
        used[tr.i0] = used[tr.i1] = used[tr.i2] = 1;
        area += 0.5 * length(cross(q.vertices[tr.i1] - q.vertices[tr.i0], q.vertices[tr.i2] - q.vertices[tr.i0]));
    }
    CHECK_THAT(area, WithinAbs(1.0, 1e-6));
    CHECK(used[0] + used[1] + used[2] + used[3] == 4);
    // A pentagon (5 indices) gives 3 triangles: n-2.
    std::string pent = tmp.write("pent.obj", "v 0 0 0\nv 2 0 0\nv 3 1 0\nv 1 2 0\nv -1 1 0\nf 1 2 3 4 5\n");
    CHECK(load_obj(pent, 8, 8).triangles.size() == 3);
}

TEST_CASE("load_obj errors name the file and the 1-based line", "[assets_loading]") {
    TmpDir tmp;
    // Each file is kObjHead (3 vertices over 6 lines) followed by one defect on line 7. Digits
    // used by the defect are never 7, and the file path is stripped before searching for "7".
    struct Defect { std::string name, line; };
    std::vector<Defect> defects = {
        {"non-numeric coordinate", "v 0 abc 0\n"},
        {"trailing garbage in number", "v 1 2abc 3\n"},
        {"face index 0", "f 0 1 2\n"},
        {"face index beyond vertex count", "f 1 2 4\n"},
        {"face with 2 indices", "f 1 2\n"},
    };
    const std::regex line7("(^|[^0-9])7([^0-9]|$)");
    for (const Defect& d : defects) {
        std::string path = tmp.write("bad_" + std::string(1, (char)('a' + (&d - &defects[0]))) + ".obj",
                                     std::string(kObjHead) + d.line + (d.line[0] == 'v' ? "f 1 2 3\n" : ""));
        bool is_rt = false;
        std::string msg = obj_error(path, &is_rt);
        INFO("defect: " << d.name << "  message: " << msg);
        REQUIRE(msg != "<no throw>");
        CHECK(is_rt);
        CHECK_THAT(msg, ContainsSubstring(path));
        std::string rest = msg;
        for (size_t p; (p = rest.find(path)) != std::string::npos;) rest.erase(p, path.size());
        CHECK(std::regex_search(rest, line7));   // 1-based, counting comment and blank lines
    }
    // The same defect one line later must report the later line: the number follows the defect
    // rather than being a constant (face index 0 on line 8 after a trailing comment).
    std::string p8 = tmp.write("late.obj", std::string(kObjHead) + "# pad\nf 0 1 2\n");
    std::string m8 = obj_error(p8);
    std::string rest8 = m8;
    for (size_t p; (p = rest8.find(p8)) != std::string::npos;) rest8.erase(p, p8.size());
    INFO("message: " << m8);
    CHECK(std::regex_search(rest8, std::regex("(^|[^0-9])8([^0-9]|$)")));
    CHECK_FALSE(std::regex_search(rest8, line7));
}

TEST_CASE("load_obj rejects missing, empty and face-less files naming the file", "[assets_loading]") {
    TmpDir tmp;
    struct Case { std::string name, path; };
    std::vector<Case> cases = {
        {"missing file", tmp.path("does_not_exist.obj")},
        {"empty file", tmp.write("empty.obj", "")},
        {"only comments and blanks", tmp.write("blank.obj", "# nothing\n\n\n")},
        {"vertices but no faces", tmp.write("nofaces.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\n")},
    };
    for (const Case& c : cases) {
        bool is_rt = false;
        std::string msg = obj_error(c.path, &is_rt);
        INFO("case: " << c.name << "  message: " << msg);
        REQUIRE(msg != "<no throw>");
        CHECK(is_rt);
        CHECK_THAT(msg, ContainsSubstring(c.path));
    }
    // Control: the same vertices plus one face load fine.
    std::string ok = tmp.write("ok.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
    CHECK(obj_error(ok) == "<no throw>");
}

TEST_CASE("build_bvh throws on a scene with no primitives and builds a single leaf for one sphere", "[assets_loading]") {
    Scene empty;
    empty.materials.push_back(mat(kLambert));
    empty.camera = Camera{Vec3{0, 0, 5}, Vec3{0, 0, 0}, Vec3{0, 1, 0}, 45.0f};
    REQUIRE_THROWS_AS(build_bvh(empty), std::runtime_error);
    CHECK(empty.nodes.empty());        // nothing half-built
    CHECK(empty.prim_order.empty());

    // Vertices without triangles are still "no primitives".
    Scene verts_only = empty;
    verts_only.vertices = {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0}};
    REQUIRE_THROWS_AS(build_bvh(verts_only), std::runtime_error);
    CHECK(verts_only.nodes.empty());

    // Positive counterpart: one sphere gives one leaf whose box is centre +- radius, skip -1.
    Scene one = empty;
    Sphere sp{}; sp.center = Vec3{1, 2, 3}; sp.radius = 0.5f; sp.material = 0;
    one.spheres.push_back(sp);
    REQUIRE_NOTHROW(build_bvh(one));
    REQUIRE(one.nodes.size() == 1);
    const BvhNode& n = one.nodes[0];
    CHECK(n.prim_count == 1);
    CHECK(n.skip == -1);
    CHECK_THAT(n.lo.x, WithinAbs(0.5, 1e-6)); CHECK_THAT(n.hi.x, WithinAbs(1.5, 1e-6));
    CHECK_THAT(n.lo.y, WithinAbs(1.5, 1e-6)); CHECK_THAT(n.hi.y, WithinAbs(2.5, 1e-6));
    CHECK_THAT(n.lo.z, WithinAbs(2.5, 1e-6)); CHECK_THAT(n.hi.z, WithinAbs(3.5, 1e-6));
    REQUIRE(one.prim_order.size() == 1);
    CHECK(one.prim_order[0] == 0);
}

TEST_CASE("make_scene throws std::invalid_argument for unknown names and succeeds for known ones", "[assets_loading]") {
    // Each rejection is its own statement inside its own expect-failure block.
    CHECK_THROWS_AS(make_scene("nope", 16, 9), std::invalid_argument);
    CHECK_THROWS_AS(make_scene("", 16, 9), std::invalid_argument);
    CHECK_THROWS_AS(make_scene("Cornell", 16, 9), std::invalid_argument);   // names are exact
    CHECK_THROWS_AS(make_scene("cornell ", 16, 9), std::invalid_argument);
    // Positive counterpart: the documented names build and come back finalized and valid.
    for (const char* name : {"cornell", "divergence", "furnace"}) {
        Scene s = make_scene(name, 16, 9);
        INFO("scene " << name);
        CHECK_FALSE(s.nodes.empty());
        CHECK(validate_scene(s).empty());
        if (std::string(name) != "furnace") CHECK_FALSE(s.lights.empty());  // lit scenes have lights
    }
}

TEST_CASE("width and height propagate to scenes, views and the camera aspect", "[assets_loading]") {
    // Oracle: stated dimensions, and |horizontal| / |vertical| of the rig equals width / height.
    TmpDir tmp;
    std::string tri = tmp.write("t.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
    struct Dim { int w, h; };
    for (Dim d : {Dim{48, 27}, Dim{7, 11}, Dim{1, 1}, Dim{100, 3}}) {
        std::vector<std::pair<std::string, Scene>> scenes;
        scenes.emplace_back("cornell", make_scene("cornell", d.w, d.h));
        scenes.emplace_back("furnace", make_scene("furnace", d.w, d.h));
        scenes.emplace_back("obj", load_obj(tri, d.w, d.h));
        for (auto& [name, s] : scenes) {
            INFO(name << " at " << d.w << "x" << d.h);
            CHECK(s.width == d.w);
            CHECK(s.height == d.h);
            SceneView v = s.view();
            CHECK(v.width == d.w);
            CHECK(v.height == d.h);
            double ratio = (double)length(v.camera.horizontal) / (double)length(v.camera.vertical);
            CHECK_THAT(ratio, WithinAbs((double)d.w / (double)d.h, 1e-4 * d.w / d.h));
        }
    }
}
