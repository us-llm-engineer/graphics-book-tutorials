// Shared presentation helpers: a 3x5 bitmap font for text overlays, the height colour palette and the
// recolouring of the triangle-heavy hills scene used by the viewer and the animation tool.
#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "wf/scene.hpp"

namespace wf::vis {

// ---- 3x5 bitmap font for the overlay (rows top to bottom, 3 bits each, left bit first) ----
struct Glyph { char c; unsigned short bits; };
inline constexpr unsigned short G(int a, int b, int c, int d, int e) { return (unsigned short)((a << 12) | (b << 9) | (c << 6) | (d << 3) | e); }
inline const Glyph kFont[] = {
    {'0', G(7,5,5,5,7)}, {'1', G(2,6,2,2,7)}, {'2', G(7,1,7,4,7)}, {'3', G(7,1,7,1,7)}, {'4', G(5,5,7,1,1)},
    {'5', G(7,4,7,1,7)}, {'6', G(7,4,7,5,7)}, {'7', G(7,1,1,1,1)}, {'8', G(7,5,7,5,7)}, {'9', G(7,5,7,1,7)},
    {'A', G(2,5,7,5,5)}, {'B', G(6,5,6,5,6)}, {'C', G(3,4,4,4,3)}, {'D', G(6,5,5,5,6)}, {'E', G(7,4,6,4,7)},
    {'F', G(7,4,6,4,4)}, {'G', G(3,4,5,5,3)}, {'H', G(5,5,7,5,5)}, {'I', G(7,2,2,2,7)}, {'J', G(1,1,1,5,2)},
    {'K', G(5,5,6,5,5)}, {'L', G(4,4,4,4,7)}, {'M', G(5,7,7,5,5)}, {'N', G(6,5,5,5,5)}, {'O', G(2,5,5,5,2)},
    {'P', G(6,5,6,4,4)}, {'Q', G(2,5,5,7,3)}, {'R', G(6,5,6,5,5)}, {'S', G(3,4,2,1,6)}, {'T', G(7,2,2,2,2)},
    {'U', G(5,5,5,5,7)}, {'V', G(5,5,5,5,2)}, {'W', G(5,5,7,7,5)}, {'X', G(5,5,2,5,5)}, {'Y', G(5,5,2,2,2)},
    {'Z', G(7,1,2,4,7)}, {'.', G(0,0,0,0,2)}, {':', G(0,2,0,2,0)}, {'-', G(0,0,7,0,0)}, {'/', G(1,1,2,4,4)},
    {'%', G(5,1,2,4,5)}, {'[', G(6,4,4,4,6)}, {']', G(3,1,1,1,3)}, {'=', G(0,7,0,7,0)}, {'+', G(0,2,7,2,0)},
};

inline unsigned short glyph_bits(char c) {
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    for (const Glyph& g : kFont) if (g.c == c) return g.bits;
    return 0;
}

// Draws text with a dark backing box into 8-bit RGB. scale is the pixel size of one font dot.
inline void draw_text(std::vector<uint8_t>& rgb, int w, int h, int x, int y, const std::string& s, int scale,
               uint8_t r, uint8_t g, uint8_t b) {
    const int cw = 4 * scale, chh = 5 * scale, pad = scale;
    const int box_w = (int)s.size() * cw + pad, box_h = chh + 2 * pad;
    for (int j = 0; j < box_h; ++j)
        for (int i = 0; i < box_w; ++i) {
            int px = x - pad + i, py = y - pad + j;
            if (px < 0 || py < 0 || px >= w || py >= h) continue;
            uint8_t* p = &rgb[3 * ((size_t)py * w + px)];
            p[0] = (uint8_t)(p[0] / 4); p[1] = (uint8_t)(p[1] / 4); p[2] = (uint8_t)(p[2] / 4);
        }
    for (size_t k = 0; k < s.size(); ++k) {
        unsigned short bits = glyph_bits(s[k]);
        for (int row = 0; row < 5; ++row)
            for (int col = 0; col < 3; ++col) {
                if (!(bits & (1 << ((4 - row) * 3 + (2 - col))))) continue;
                for (int dy = 0; dy < scale; ++dy)
                    for (int dx = 0; dx < scale; ++dx) {
                        int px = x + (int)k * cw + col * scale + dx, py = y + row * scale + dy;
                        if (px < 0 || py < 0 || px >= w || py >= h) continue;
                        uint8_t* p = &rgb[3 * ((size_t)py * w + px)];
                        p[0] = r; p[1] = g; p[2] = b;
                    }
            }
    }
}

// Height palette (low to high): indigo, blue, cyan, green, yellow, orange. Valleys are the darkest band.
inline constexpr int kBands = 96;
inline void height_color(float t, float rgb[3]) {
    static const float stops[6][3] = {{0.22f, 0.10f, 0.60f}, {0.12f, 0.40f, 0.95f}, {0.10f, 0.80f, 0.85f},
                                      {0.35f, 0.90f, 0.35f}, {1.00f, 0.92f, 0.20f}, {1.00f, 0.50f, 0.12f}};
    t = std::max(0.0f, std::min(1.0f, t)) * 5.0f;
    int i = std::min(4, (int)t); float f = t - (float)i;
    for (int c = 0; c < 3; ++c) rgb[c] = stops[i][c] * (1.0f - f) + stops[i + 1][c] * f;
}

inline void tint_by_height(Scene& scene) {
        if (scene.name != "trianglehv") return;
        float lo = 1e30f, hi = -1e30f;
        for (const Vec3& v : scene.vertices) { lo = std::min(lo, v.y); hi = std::max(hi, v.y); }
        const int base = (int)scene.materials.size();
        for (int b = 0; b < kBands; ++b) {
            Material m = scene.materials[0];
            float c[3]; height_color((b + 0.5f) / kBands, c);
            m.albedo = Vec3{c[0], c[1], c[2]};
            scene.materials.push_back(m);
        }
        for (Triangle& t : scene.triangles) {
            if (t.material != 0) continue;
            float y = (scene.vertices[t.i0].y + scene.vertices[t.i1].y + scene.vertices[t.i2].y) / 3.0f;
            t.material = base + std::min(kBands - 1, (int)((y - lo) / std::max(hi - lo, 1e-6f) * kBands));
        }
        // The two mirror spheres reflect the banded hills as a scalloped, speckled pattern; the viewer copy uses a
        // clean matte material for them instead.
        Material matte = scene.materials[0];
        matte.albedo = Vec3{0.92f, 0.92f, 0.95f};
        const int matte_id = (int)scene.materials.size();
        scene.materials.push_back(matte);
        for (Sphere& sp : scene.spheres) if (sp.material == 1) sp.material = matte_id;
        scene.env = Vec3{0.65f, 0.70f, 0.85f};
    }

}  // namespace wf::vis
