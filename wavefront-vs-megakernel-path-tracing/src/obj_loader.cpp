#include "wf/scene.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace wf {

// Strict number parsing with error detection
static bool parse_float(const std::string& str, float& out) {
    if (str.empty()) return false;
    char* end = nullptr;
    float val = strtof(str.c_str(), &end);
    // Strict: all characters must be consumed
    if (*end != '\0') return false;
    out = val;
    return true;
}

static bool parse_int(const std::string& str, int& out) {
    if (str.empty()) return false;
    char* end = nullptr;
    long val = strtol(str.c_str(), &end, 10);
    // Strict: all characters must be consumed
    if (*end != '\0') return false;
    out = (int)val;
    return true;
}

// Split a line by whitespace, trimming leading/trailing whitespace
static std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> parts;
    size_t i = 0;
    while (i < line.length()) {
        // Skip whitespace
        while (i < line.length() && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i >= line.length()) break;
        // Read token
        size_t start = i;
        while (i < line.length() && line[i] != ' ' && line[i] != '\t') ++i;
        parts.push_back(line.substr(start, i - start));
    }
    return parts;
}

Scene load_obj(const std::string& path, int width, int height) {
    Scene scene;
    scene.name = path;
    scene.width = width;
    scene.height = height;

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::string line;
    int line_num = 0;
    std::vector<Vec3> vertices;
    std::vector<std::vector<int>> faces;  // face indices
    bool has_faces = false;

    while (std::getline(file, line)) {
        line_num++;

        // Remove comments
        size_t hash_pos = line.find('#');
        if (hash_pos != std::string::npos) {
            line = line.substr(0, hash_pos);
        }

        // Trim trailing whitespace
        while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }

        // Skip blank lines
        if (line.empty()) continue;

        auto parts = split_line(line);
        if (parts.empty()) continue;

        if (parts[0] == "v") {
            // Parse vertex
            if (parts.size() != 4) {
                throw std::runtime_error(path + " line " + std::to_string(line_num) + ": vertex needs exactly 3 coordinates");
            }
            float x, y, z;
            if (!parse_float(parts[1], x) || !parse_float(parts[2], y) || !parse_float(parts[3], z)) {
                throw std::runtime_error(path + " line " + std::to_string(line_num) + ": malformed vertex coordinate");
            }
            vertices.push_back(Vec3{x, y, z});
        } else if (parts[0] == "vn") {
            // Ignore vertex normals
        } else if (parts[0] == "usemtl") {
            // Ignore material
        } else if (parts[0] == "f") {
            // Parse face
            if (parts.size() < 4) {
                throw std::runtime_error(path + " line " + std::to_string(line_num) + ": face needs at least 3 indices");
            }
            has_faces = true;

            // Parse face indices (1-based in file, convert to 0-based)
            std::vector<int> face_indices;
            for (size_t i = 1; i < parts.size(); ++i) {
                // Handle "i", "i/vt", "i/vt/vn", "i//vn"
                size_t slash1 = parts[i].find('/');
                std::string idx_str = (slash1 != std::string::npos) ?
                                     parts[i].substr(0, slash1) : parts[i];
                int idx;
                if (!parse_int(idx_str, idx)) {
                    throw std::runtime_error(path + " line " + std::to_string(line_num) + ": malformed face index");
                }
                if (idx <= 0 || idx > (int)vertices.size()) {
                    throw std::runtime_error(path + " line " + std::to_string(line_num) + ": face index out of range");
                }
                face_indices.push_back(idx - 1);  // Convert to 0-based
            }
            faces.push_back(face_indices);
        }
    }

    // Convert to vertex array
    scene.vertices = vertices;

    // Convert faces to triangles (fan triangulation)
    if (!has_faces) {
        throw std::runtime_error(path + ": no faces found");
    }

    // Create a default Lambertian material (grey 0.7)
    Material mat;
    mat.type = kLambert;
    mat.albedo = Vec3{0.7f, 0.7f, 0.7f};
    mat.emission = Vec3{0, 0, 0};
    mat.ior = 1.5f;
    scene.materials.push_back(mat);

    for (const auto& face : faces) {
        if (face.size() < 3) continue;
        // Fan triangulation: triangles (0, i, i+1) for i >= 1
        for (size_t i = 1; i + 1 < face.size(); ++i) {
            scene.triangles.push_back(Triangle{face[0], (int)face[i], (int)face[i + 1], 0});
        }
    }

    // Set a default camera based on the mesh bounds
    // Position back along +z from the bounds, looking at the center
    if (!scene.vertices.empty()) {
        Vec3 lo = scene.vertices[0];
        Vec3 hi = scene.vertices[0];
        for (const auto& v : scene.vertices) {
            lo = vmin(lo, v);
            hi = vmax(hi, v);
        }
        
        Vec3 center = (lo + hi) * 0.5f;
        Vec3 extent = hi - lo;
        float diag = length(extent);
        
        // Position camera back along +z from bounds center, at a distance of 1.5 times the diagonal
        scene.camera.position = center + Vec3{0.0f, 0.0f, diag * 1.5f};
        scene.camera.look_at = center;
        scene.camera.up = Vec3{0.0f, 1.0f, 0.0f};
        scene.camera.vfov_deg = 45.0f;
    } else {
        // Fallback (should not happen if finalize_scene is called)
        scene.camera.position = Vec3{0.0f, 0.0f, 5.0f};
        scene.camera.look_at = Vec3{0.0f, 0.0f, 0.0f};
        scene.camera.up = Vec3{0.0f, 1.0f, 0.0f};
        scene.camera.vfov_deg = 45.0f;
    }

    // Finalize the scene (build BVH, compute lights)
    try {
        finalize_scene(scene);
    } catch (...) {
        throw;
    }

    return scene;
}

}  // namespace wf
