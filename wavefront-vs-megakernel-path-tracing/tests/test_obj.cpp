// OBJ loader tests. Needs src/obj_loader.cpp to link.
// Temp files live under /tmp. File names are deliberately digit-free so that a line number found in
// an error message cannot be explained by digits of the file name.
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "wf/scene.hpp"

#include "test_framework.hpp"

using namespace wf;

namespace {

std::string letters(unsigned long long v) {
    std::string s;
    do { s.push_back((char)('a' + v % 26)); v /= 26; } while (v);
    return s;
}

struct TmpFile {
    std::string path, base;
    explicit TmpFile(const std::string& content) {
        static unsigned long long counter = 0;
        unsigned long long now = (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count();
        base = "wfobj_" + letters((unsigned long long)getpid()) + "_" + letters(counter++) + "_" + letters(now) + ".obj";
        path = "/tmp/" + base;
        std::ofstream f(path, std::ios::binary);
        f << content;
    }
    ~TmpFile() { std::remove(path.c_str()); }
    TmpFile(const TmpFile&) = delete;
    TmpFile& operator=(const TmpFile&) = delete;
};

std::string join(const std::vector<std::string>& lines) {
    std::string s;
    for (const auto& l : lines) s += l + "\n";
    return s;
}

// Thrown message must be std::runtime_error, name the file, and (line > 0) the 1-based line.
void expect_load_error(const std::string& path, int line, const char* what) {
    std::string msg;
    bool runtime = false, other = false;
    try { (void)load_obj(path, 8, 8); }
    catch (const std::runtime_error& e) { runtime = true; msg = e.what(); }
    catch (const std::exception& e) { other = true; msg = e.what(); }
    if (!runtime) std::printf("  %s: expected std::runtime_error, got %s\n", what, other ? "another exception type" : "no exception");
    CHECK(runtime);
    if (!runtime) return;
    std::string base = path.substr(path.find_last_of('/') + 1);
    bool has_name = msg.find(base) != std::string::npos;
    if (!has_name) std::printf("  %s: message lacks file name: '%s'\n", what, msg.c_str());
    CHECK(has_name);
    if (line > 0) {
        bool has_line = msg.find(std::to_string(line)) != std::string::npos;
        if (!has_line) std::printf("  %s: message lacks line %d: '%s'\n", what, line, msg.c_str());
        CHECK(has_line);
    }
}

// `pad` comment lines first, then `body`; defective line is body[defect_index] (0-based).
// Checks the message at two different padding amounts so the line number cannot be a constant.
void expect_error_at_body_line(const std::vector<std::string>& body, int defect_index, const char* what) {
    for (int pad : {2, 4}) {
        std::vector<std::string> lines;
        for (int i = 0; i < pad; ++i) lines.push_back("# pad");
        lines.insert(lines.end(), body.begin(), body.end());
        TmpFile f(join(lines));
        expect_load_error(f.path, pad + defect_index + 1, what);
    }
}

Vec3 vtx(const Scene& s, int tri, int corner) {
    const Triangle& t = s.triangles[tri];
    int idx = corner == 0 ? t.i0 : (corner == 1 ? t.i1 : t.i2);
    return s.vertices[idx];
}
bool same(Vec3 a, Vec3 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
double area_z(Vec3 a, Vec3 b, Vec3 c) {   // signed area projected on z (CCW > 0)
    return 0.5 * ((double)(b.x - a.x) * (c.y - a.y) - (double)(b.y - a.y) * (c.x - a.x));
}

}  // namespace

TEST(obj_single_triangle_loads_with_absolute_one_based_indices) {
    // Oracle: "f 2 3 4" refers to the 2nd, 3rd and 4th "v" lines (1-based). Zero-based or shifted
    // indexing would pick different positions.
    TmpFile f(join({"v 0 0 0", "v 1 0 0", "v 0 2 0", "v 0 0 3", "f 2 3 4"}));
    Scene s = load_obj(f.path, 20, 10);
    CHECK(s.triangles.size() == 1u);
    CHECK(s.spheres.empty());
    if (s.triangles.size() == 1) {
        CHECK(same(vtx(s, 0, 0), Vec3{1, 0, 0}));
        CHECK(same(vtx(s, 0, 1), Vec3{0, 2, 0}));
        CHECK(same(vtx(s, 0, 2), Vec3{0, 0, 3}));
    }
    CHECK(s.width == 20);
    CHECK(s.height == 10);
    CHECK(validate_scene(s).empty());
}

TEST(obj_uses_one_grey_lambertian_material) {
    TmpFile f(join({"v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 3"}));
    Scene s = load_obj(f.path, 8, 8);
    CHECK(s.materials.size() == 1u);
    if (s.materials.size() == 1) {
        CHECK(s.materials[0].type == kLambert);
        CHECK_NEAR(s.materials[0].albedo.x, 0.7, 1e-6);
        CHECK_NEAR(s.materials[0].albedo.y, 0.7, 1e-6);
        CHECK_NEAR(s.materials[0].albedo.z, 0.7, 1e-6);
        CHECK(s.materials[0].emission.x == 0.0f && s.materials[0].emission.y == 0.0f && s.materials[0].emission.z == 0.0f);
    }
    CHECK(s.triangles[0].material == 0);
}

TEST(obj_parses_negative_and_fractional_and_exponent_coordinates_exactly) {
    // All values exactly representable in binary: 1e0 = 1, -2.5, 0.25, 1.5e1 = 15.
    TmpFile f(join({"v 1e0 -2.5 0.25", "v 1.5e1 0 0", "v 0 0 -0.5", "f 1 2 3"}));
    Scene s = load_obj(f.path, 8, 8);
    CHECK(s.triangles.size() == 1u);
    if (s.triangles.size() == 1) {
        CHECK(same(vtx(s, 0, 0), Vec3{1.0f, -2.5f, 0.25f}));
        CHECK(same(vtx(s, 0, 1), Vec3{15.0f, 0.0f, 0.0f}));
        CHECK(same(vtx(s, 0, 2), Vec3{0.0f, 0.0f, -0.5f}));
    }
}

TEST(obj_quad_is_triangulated_into_two_triangles_covering_it_once) {
    // Unit square CCW. Either diagonal is a valid triangulation: {123,134} or {124,234}.
    const Vec3 P[4] = {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{1, 1, 0}, Vec3{0, 1, 0}};
    TmpFile f(join({"v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0", "f 1 2 3 4"}));
    Scene s = load_obj(f.path, 8, 8);
    CHECK(s.triangles.size() == 2u);
    if (s.triangles.size() != 2) return;
    std::vector<std::vector<int>> sets;
    double total = 0.0;
    for (int t = 0; t < 2; ++t) {
        std::vector<int> ids;
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 4; ++k)
                if (same(vtx(s, t, c), P[k])) ids.push_back(k + 1);
        std::sort(ids.begin(), ids.end());
        sets.push_back(ids);
        double a = area_z(vtx(s, t, 0), vtx(s, t, 1), vtx(s, t, 2));
        CHECK_NEAR(a, 0.5, 1e-6);   // positive: winding preserved, no sliver
        total += a;
    }
    CHECK_NEAR(total, 1.0, 1e-6);
    std::sort(sets.begin(), sets.end());
    const std::vector<std::vector<int>> diag13 = {{1, 2, 3}, {1, 3, 4}};
    const std::vector<std::vector<int>> diag24 = {{1, 2, 4}, {2, 3, 4}};
    CHECK(sets == diag13 || sets == diag24);
}

TEST(obj_pentagon_is_triangulated_into_three_triangles_with_exact_total_area) {
    // Convex CCW pentagon; shoelace area computed independently. Three CCW triangles whose areas
    // sum to the polygon area cannot overlap.
    const double px[5] = {0, 2, 3, 1, -1}, py[5] = {0, 0, 1.5, 3, 1.5};
    double shoelace = 0.0;
    for (int i = 0; i < 5; ++i) shoelace += px[i] * py[(i + 1) % 5] - px[(i + 1) % 5] * py[i];
    shoelace *= 0.5;   // 7.5
    TmpFile f(join({"v 0 0 0", "v 2 0 0", "v 3 1.5 0", "v 1 3 0", "v -1 1.5 0", "f 1 2 3 4 5"}));
    Scene s = load_obj(f.path, 8, 8);
    CHECK(s.triangles.size() == 3u);
    double total = 0.0;
    for (size_t t = 0; t < s.triangles.size(); ++t) {
        double a = area_z(vtx(s, (int)t, 0), vtx(s, (int)t, 1), vtx(s, (int)t, 2));
        CHECK(a > 1e-6);
        total += a;
    }
    CHECK_NEAR(total, shoelace, 1e-5);
}

TEST(obj_ignores_comments_blank_lines_and_unused_statements) {
    TmpFile f(join({"# header comment", "", "v 0 0 0", "   ", "# between", "v 1 0 0", "mtllib x.mtl",
                    "usemtl red", "o thing", "g group", "s off", "v 0 1 0", "", "f 1 2 3", "# end"}));
    Scene s = load_obj(f.path, 8, 8);
    CHECK(s.triangles.size() == 1u);
    CHECK(s.vertices.size() == 3u);
}

TEST(obj_face_forms_with_texture_and_normal_indices_use_only_the_vertex_index) {
    // Three vt and three vn lines exist. f 1/2/3, f 1//3, f 1/2 all mean vertex index 1 etc.
    TmpFile f(join({"v 0 0 0", "v 1 0 0", "v 0 1 0", "v 1 1 0", "vt 0 0", "vt 1 0", "vt 0 1", "vn 0 0 1",
                    "vn 0 0 1", "vn 0 0 1", "f 1/2/3 2/3/1 3/1/2", "f 2//3 4//1 3//2", "f 1/2 2/3 4/1"}));
    Scene s = load_obj(f.path, 8, 8);
    CHECK(s.triangles.size() == 3u);
    if (s.triangles.size() != 3) return;
    CHECK(same(vtx(s, 0, 0), Vec3{0, 0, 0}) && same(vtx(s, 0, 1), Vec3{1, 0, 0}) && same(vtx(s, 0, 2), Vec3{0, 1, 0}));
    CHECK(same(vtx(s, 1, 0), Vec3{1, 0, 0}) && same(vtx(s, 1, 1), Vec3{1, 1, 0}) && same(vtx(s, 1, 2), Vec3{0, 1, 0}));
    CHECK(same(vtx(s, 2, 0), Vec3{0, 0, 0}) && same(vtx(s, 2, 1), Vec3{1, 0, 0}) && same(vtx(s, 2, 2), Vec3{1, 1, 0}));
    CHECK(validate_scene(s).empty());
}

TEST(obj_two_triangles_share_vertices_by_index) {
    TmpFile f(join({"v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0", "f 1 2 3", "f 1 3 4"}));
    Scene s = load_obj(f.path, 8, 8);
    CHECK(s.triangles.size() == 2u);
    if (s.triangles.size() == 2u) {
        CHECK(same(vtx(s, 1, 0), Vec3{0, 0, 0}));
        CHECK(same(vtx(s, 1, 1), Vec3{1, 1, 0}));
        CHECK(same(vtx(s, 1, 2), Vec3{0, 1, 0}));
    }
}

// ---------------------------------------------------------------- invalid assets

TEST(obj_missing_file_throws_runtime_error_naming_the_file) {
    std::string path = "/tmp/wfobj_missing_" + letters((unsigned long long)getpid()) + ".obj";
    std::remove(path.c_str());
    expect_load_error(path, 0, "missing file");
}

TEST(obj_non_numeric_vertex_coordinate_names_file_and_line) {
    expect_error_at_body_line({"v 0 0 0", "v 1 0 0", "v 0 abc 0", "v 0 1 0", "f 1 2 4"}, 2, "non-numeric coordinate");
}

TEST(obj_vertex_coordinate_with_trailing_garbage_is_malformed) {
    // "2abc" is not a number; a prefix-parsing reader would silently accept 2.
    expect_error_at_body_line({"v 0 0 0", "v 1 0 0", "v 0 2abc 0", "v 0 1 0", "f 1 2 4"}, 2, "trailing garbage");
}

TEST(obj_face_index_zero_is_rejected_with_file_and_line) {
    // OBJ indices are 1-based, so 0 is invalid (it must not wrap to -1 / "last vertex").
    expect_error_at_body_line({"v 0 0 0", "v 1 0 0", "v 0 1 0", "f 0 1 2"}, 3, "index 0");
}

TEST(obj_face_index_beyond_vertex_count_is_rejected_with_file_and_line) {
    expect_error_at_body_line({"v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 40"}, 3, "index 40 of 3");
}

TEST(obj_face_index_one_past_vertex_count_is_rejected) {
    // Off-by-one boundary: 3 vertices, index 4.
    expect_error_at_body_line({"v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2 4"}, 3, "index 4 of 3");
}

TEST(obj_face_with_only_two_indices_is_rejected_with_file_and_line) {
    expect_error_at_body_line({"v 0 0 0", "v 1 0 0", "v 0 1 0", "f 1 2"}, 3, "2-index face");
}

TEST(obj_file_with_vertices_but_no_faces_is_rejected) {
    TmpFile f(join({"v 0 0 0", "v 1 0 0", "v 0 1 0"}));
    expect_load_error(f.path, 0, "no faces");
}

TEST(obj_empty_file_is_rejected) {
    TmpFile f("");
    expect_load_error(f.path, 0, "empty file");
}

TEST(obj_comments_only_file_is_rejected) {
    TmpFile f(join({"# nothing here", "", "# still nothing"}));
    expect_load_error(f.path, 0, "comments only");
}
