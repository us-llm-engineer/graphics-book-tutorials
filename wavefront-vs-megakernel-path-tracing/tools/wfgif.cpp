// wfgif: renders progressive-refinement frames of the three scenes and writes an animated GIF.
//   wfgif <out.gif> [--width N] [--height N] [--mode megakernel|wavefront] [--final-png-prefix PREFIX]
// Frames are real renders (1, 2, 4 ... 256 samples per pixel). The encoder quantises to a fixed
// 6x7x6 colour cube with Floyd-Steinberg error diffusion and compresses with GIF-style LZW.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "wf/film.hpp"
#include "wf/image_io.hpp"
#include "wf/renderer.hpp"
#include "wf/scene.hpp"
#include "wf/visual.hpp"

namespace {
constexpr int kLevelsR = 6, kLevelsG = 7, kLevelsB = 6;

// Maps an RGB image to palette indices (r*42 + g*6 + b) with error diffusion.
std::vector<uint8_t> quantize(const std::vector<uint8_t>& rgb, int w, int h) {
    std::vector<float> buf(rgb.begin(), rgb.end());
    std::vector<uint8_t> idx((size_t)w * h);
    const int levels[3] = {kLevelsR, kLevelsG, kLevelsB};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float* p = &buf[3 * ((size_t)y * w + x)];
            int q[3];
            for (int c = 0; c < 3; ++c) {
                float v = std::max(0.0f, std::min(255.0f, p[c]));
                q[c] = (int)std::lround(v / 255.0f * (levels[c] - 1));
                float approx = 255.0f * q[c] / (levels[c] - 1), err = p[c] - approx;
                if (x + 1 < w) buf[3 * ((size_t)y * w + x + 1) + c] += err * 7.0f / 16.0f;
                if (y + 1 < h) {
                    if (x > 0) buf[3 * ((size_t)(y + 1) * w + x - 1) + c] += err * 3.0f / 16.0f;
                    buf[3 * ((size_t)(y + 1) * w + x) + c] += err * 5.0f / 16.0f;
                    if (x + 1 < w) buf[3 * ((size_t)(y + 1) * w + x + 1) + c] += err * 1.0f / 16.0f;
                }
            }
            idx[(size_t)y * w + x] = (uint8_t)(q[0] * (kLevelsG * kLevelsB) + q[1] * kLevelsB + q[2]);
        }
    return idx;
}

struct BitWriter {
    std::vector<uint8_t> out; uint32_t acc = 0; int nbits = 0;
    void put(int code, int size) { acc |= (uint32_t)code << nbits; nbits += size; while (nbits >= 8) { out.push_back((uint8_t)(acc & 255)); acc >>= 8; nbits -= 8; } }
    void flush() { if (nbits > 0) out.push_back((uint8_t)(acc & 255)); acc = 0; nbits = 0; }
};

// GIF LZW with minimum code size 8: clear code 256, end code 257, codes grow from 9 to 12 bits.
std::vector<uint8_t> lzw(const std::vector<uint8_t>& px) {
    BitWriter bw; std::unordered_map<uint32_t, int> dict; int code_size = 9, max_code = 257;
    bw.put(256, code_size);
    int cur = px[0];
    for (size_t i = 1; i < px.size(); ++i) {
        uint32_t key = ((uint32_t)cur << 8) | px[i];
        auto it = dict.find(key);
        if (it != dict.end()) { cur = it->second; continue; }
        bw.put(cur, code_size);
        dict[key] = ++max_code;
        if (max_code >= (1 << code_size)) ++code_size;
        if (max_code == 4095) { bw.put(256, code_size); dict.clear(); code_size = 9; max_code = 257; }
        cur = px[i];
    }
    bw.put(cur, code_size); bw.put(256, code_size); bw.put(257, 9);
    bw.flush();
    return bw.out;
}

void put16(std::ostream& o, int v) { o.put((char)(v & 255)); o.put((char)((v >> 8) & 255)); }

void write_header(std::ostream& o, int w, int h) {
    o.write("GIF89a", 6); put16(o, w); put16(o, h); o.put((char)0xF7); o.put(0); o.put(0);
    for (int r = 0; r < kLevelsR; ++r) for (int g = 0; g < kLevelsG; ++g) for (int b = 0; b < kLevelsB; ++b) {
        o.put((char)std::lround(255.0 * r / (kLevelsR - 1))); o.put((char)std::lround(255.0 * g / (kLevelsG - 1))); o.put((char)std::lround(255.0 * b / (kLevelsB - 1)));
    }
    for (int i = kLevelsR * kLevelsG * kLevelsB; i < 256; ++i) { o.put(0); o.put(0); o.put(0); }
    o.write("\x21\xFF\x0BNETSCAPE2.0\x03\x01\x00\x00\x00", 19);   // loop forever
}

void write_frame(std::ostream& o, const std::vector<uint8_t>& idx, int w, int h, int delay_cs) {
    o.write("\x21\xF9\x04\x04", 4); put16(o, delay_cs); o.put(0); o.put(0);   // graphic control: keep, delay
    o.put(0x2C); put16(o, 0); put16(o, 0); put16(o, w); put16(o, h); o.put(0);
    o.put(8);
    std::vector<uint8_t> data = lzw(idx);
    for (size_t pos = 0; pos < data.size(); pos += 255) { size_t n = std::min<size_t>(255, data.size() - pos); o.put((char)n); o.write((const char*)&data[pos], (std::streamsize)n); }
    o.put(0);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) { std::cerr << "Usage: wfgif <out.gif> [--width N] [--height N] [--mode megakernel|wavefront] [--final-png-prefix P]\n"; return 2; }
        std::string out = argv[1], png_prefix; int W = 448, H = 252; wf::GpuMode mode = wf::GpuMode::Megakernel;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i]; auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a); return argv[++i]; };
            if (a == "--width") W = std::stoi(next()); else if (a == "--height") H = std::stoi(next());
            else if (a == "--mode") { std::string m = next(); mode = m == "wavefront" ? wf::GpuMode::Wavefront : wf::GpuMode::Megakernel; }
            else if (a == "--final-png-prefix") png_prefix = next(); else throw std::invalid_argument("unknown argument " + a);
        }
        std::ofstream f(out, std::ios::binary); if (!f) throw std::runtime_error("cannot write " + out);
        write_header(f, W, H);
        const char* scenes[3] = {"cornell", "divergence", "trianglehv"}; const float exposure[3] = {2.2f, 3.0f, 1.0f};
        const int steps[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
        int frames = 0;
        for (int si = 0; si < 3; ++si) {
            wf::Scene scene = wf::make_scene(scenes[si], W, H); wf::vis::tint_by_height(scene);
            wf::GpuRenderer renderer(scene, mode); renderer.reset();
            std::vector<float> mean; int done = 0;
            for (int target : steps) {
                wf::RenderParams p; wf::RenderStats st;
                renderer.run_passes(done, target - done, p, &st); done = target;
                renderer.download_mean(done, &mean);
                std::vector<uint8_t> rgb8 = wf::tonemap_to_srgb8(mean, exposure[si]);
                if (target == steps[8] && !png_prefix.empty()) wf::write_png(png_prefix + scenes[si] + ".png", W, H, rgb8);
                wf::vis::draw_text(rgb8, W, H, 6, 6, std::string(scenes[si]) + "  " + std::to_string(done) + " SPP", 2, 255, 255, 255);
                write_frame(f, quantize(rgb8, W, H), W, H, target == steps[8] ? 150 : 25);
                ++frames;
            }
        }
        f.put(0x3B); f.close();
        std::cout << "wrote " << out << " (" << frames << " frames, " << W << "x" << H << ")\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 2; }
}
