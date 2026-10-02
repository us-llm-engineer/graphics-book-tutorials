// Tests for PNG/PFM writers, PFM reader and FNV-1a hashing (include/wf/image_io.hpp).
// Oracles: the PNG spec (signature, chunk layout, CRC-32, zlib/Adler-32, stored deflate blocks), which the test
// re-implements independently (crc32, adler32, stored-block decoder); the PFM format text; published FNV-1a vectors.
// The PNG tests decode the file bytes themselves and compare against the pixels that were written, so a
// writer that emits a self-consistent but wrong stream (bad CRC, wrong Adler, wrong block length) is caught.
// Assumption not stated in image_io.hpp: scanlines use filter type 0 (None), the only filter a stored-deflate
// writer has any reason to use.
#include "test_framework.hpp"
#include "wf/image_io.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------- independent reference code
uint32_t ref_crc32(const uint8_t* d, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c ^= d[i];
        for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0xEDB88320u : (c >> 1);
    }
    return ~c;
}
uint32_t ref_adler32(const uint8_t* d, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; ++i) { a = (a + d[i]) % 65521u; b = (b + a) % 65521u; }
    return (b << 16) | a;
}
uint64_t ref_fnv1a64(const uint8_t* d, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) { h ^= d[i]; h *= 0x100000001b3ull; }
    return h;
}

std::vector<uint8_t> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void write_bytes(const std::string& path, const std::vector<uint8_t>& b) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}
uint32_t be32(const uint8_t* p) { return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]; }

std::string tmp_dir() {
    static std::string d = [] {
        fs::path p = fs::temp_directory_path() / ("wf_test_image_io_" + std::to_string(std::random_device{}()));
        fs::create_directories(p);
        return p.string();
    }();
    return d;
}
std::string tmp(const std::string& name) { return tmp_dir() + "/" + name; }

std::vector<uint8_t> make_rgb(int w, int h, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<uint8_t> v(static_cast<size_t>(w) * h * 3);
    for (auto& b : v) b = static_cast<uint8_t>(g() >> 13);
    return v;
}

// Parses and validates a PNG file; fills raw scanline pixels. Returns false (after recording failures) if the
// structure is too broken to continue.
struct PngInfo { uint32_t adler = 0; int stored_blocks = 0; uint32_t iend_crc = 0; };
bool decode_png(const std::string& path, int w, int h, std::vector<uint8_t>* pixels, PngInfo* info) {
    std::vector<uint8_t> f = read_bytes(path);
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (f.size() < 8 + 25 + 12 || std::memcmp(f.data(), sig, 8) != 0) {
        std::printf("  FAIL %s: missing PNG signature or too short (%zu bytes)\n", path.c_str(), f.size());
        ++wftest::failures();
        return false;
    }
    size_t pos = 8;
    std::vector<uint8_t> z;
    int chunk_index = 0;
    bool saw_iend = false;
    std::string last_type;
    while (pos + 12 <= f.size()) {
        uint32_t len = be32(&f[pos]);
        if (pos + 12 + len > f.size()) { std::printf("  FAIL chunk length %u overruns file\n", len); ++wftest::failures(); return false; }
        std::string type(reinterpret_cast<const char*>(&f[pos + 4]), 4);
        const uint8_t* data = &f[pos + 8];
        uint32_t stored_crc = be32(&f[pos + 8 + len]);
        uint32_t want_crc = ref_crc32(&f[pos + 4], 4 + len);  // CRC covers type + data
        if (stored_crc != want_crc) {
            std::printf("  FAIL chunk %s CRC 0x%08x, expected 0x%08x\n", type.c_str(), stored_crc, want_crc);
            ++wftest::failures();
        }
        if (chunk_index == 0) {
            CHECK(type == "IHDR");
            CHECK(len == 13);
            if (type == "IHDR" && len == 13) {
                CHECK(be32(data) == static_cast<uint32_t>(w));
                CHECK(be32(data + 4) == static_cast<uint32_t>(h));
                CHECK(data[8] == 8);   // bit depth
                CHECK(data[9] == 2);   // colour type: truecolour RGB
                CHECK(data[10] == 0);  // compression: deflate
                CHECK(data[11] == 0);  // filter method
                CHECK(data[12] == 0);  // no interlace
            }
        } else if (type == "IDAT") {
            CHECK(last_type == "IHDR" || last_type == "IDAT");  // IDATs are consecutive, right after IHDR
            z.insert(z.end(), data, data + len);
        } else if (type == "IEND") {
            CHECK(len == 0);
            if (info) info->iend_crc = stored_crc;
            saw_iend = true;
        }
        last_type = type;
        pos += 12 + len;
        ++chunk_index;
        if (saw_iend) break;
    }
    CHECK(saw_iend);
    CHECK(pos == f.size());  // nothing after IEND
    if (z.size() < 2 + 5 + 4) { std::printf("  FAIL IDAT payload too short\n"); ++wftest::failures(); return false; }

    // zlib header: CM = 8, CINFO <= 7, (CMF*256 + FLG) % 31 == 0, no preset dictionary.
    CHECK((z[0] & 0x0F) == 8);
    CHECK((z[0] >> 4) <= 7);
    CHECK(((z[0] << 8) | z[1]) % 31 == 0);
    CHECK((z[1] & 0x20) == 0);

    // stored deflate blocks
    std::vector<uint8_t> raw;
    size_t p = 2;
    bool final_seen = false;
    int blocks = 0;
    while (!final_seen) {
        if (p + 5 > z.size() - 4) { std::printf("  FAIL stored block header overruns zlib stream\n"); ++wftest::failures(); return false; }
        uint8_t hdr = z[p++];
        final_seen = (hdr & 1) != 0;
        CHECK(((hdr >> 1) & 3) == 0);  // BTYPE 00 = stored
        uint32_t len = z[p] | (uint32_t(z[p + 1]) << 8);
        uint32_t nlen = z[p + 2] | (uint32_t(z[p + 3]) << 8);
        p += 4;
        CHECK(nlen == (~len & 0xFFFFu));
        if (p + len > z.size() - 4) { std::printf("  FAIL stored block of %u bytes overruns stream\n", len); ++wftest::failures(); return false; }
        raw.insert(raw.end(), z.begin() + p, z.begin() + p + len);
        p += len;
        ++blocks;
    }
    CHECK(p + 4 == z.size());  // exactly the Adler-32 trailer remains
    uint32_t adler = be32(&z[z.size() - 4]);
    CHECK(adler == ref_adler32(raw.data(), raw.size()));
    if (info) { info->adler = adler; info->stored_blocks = blocks; }

    const size_t row = 1 + 3 * static_cast<size_t>(w);
    if (raw.size() != row * h) {
        std::printf("  FAIL decoded %zu bytes, expected %zu\n", raw.size(), row * h);
        ++wftest::failures();
        return false;
    }
    pixels->assign(static_cast<size_t>(w) * h * 3, 0);
    for (int y = 0; y < h; ++y) {
        CHECK(raw[row * y] == 0);  // filter type None
        std::memcpy(pixels->data() + static_cast<size_t>(y) * w * 3, &raw[row * y + 1], 3 * static_cast<size_t>(w));
    }
    return true;
}

void png_round_trip(int w, int h, uint32_t seed, int min_blocks) {
    std::vector<uint8_t> rgb = make_rgb(w, h, seed);
    std::string path = tmp("rt_" + std::to_string(w) + "x" + std::to_string(h) + ".png");
    wf::write_png(path, w, h, rgb);
    std::vector<uint8_t> got;
    PngInfo info;
    if (!decode_png(path, w, h, &got, &info)) return;
    CHECK(got == rgb);
    CHECK(info.stored_blocks >= min_blocks);
    // Every byte of each scanline-flattened pixel must land in order; report the first mismatch for debugging.
    if (got != rgb) {
        for (size_t i = 0; i < rgb.size() && i < got.size(); ++i)
            if (got[i] != rgb[i]) { std::printf("  first differing byte at %zu: got %u want %u\n", i, got[i], rgb[i]); break; }
    }
}

struct PfmHeader { std::string magic; int w = 0, h = 0; double scale = 0; size_t data_offset = 0; bool ok = false; };
PfmHeader parse_pfm_header(const std::vector<uint8_t>& f) {
    PfmHeader H;
    size_t p = 0;
    auto token = [&](std::string& out) {
        while (p < f.size() && std::isspace(f[p])) ++p;
        out.clear();
        while (p < f.size() && !std::isspace(f[p])) out += static_cast<char>(f[p++]);
        return !out.empty();
    };
    std::string a, b, c;
    if (!token(H.magic) || !token(a) || !token(b) || !token(c)) return H;
    H.w = std::stoi(a); H.h = std::stoi(b); H.scale = std::stod(c);
    H.data_offset = p + 1;  // exactly one whitespace character separates the scale from the data
    H.ok = H.data_offset <= f.size();
    return H;
}

uint32_t bits_of(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
float float_of(uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }

std::vector<float> random_floats(size_t n, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<float> v(n);
    for (auto& x : v) {
        uint32_t u = g();
        if (((u >> 23) & 0xFF) == 0xFF) u &= ~0x7F800000u;  // no inf/nan: becomes a denormal or zero
        x = float_of(u);
    }
    const float special[] = {0.0f, -0.0f, 1.4e-45f, -1.4e-45f, -1e-40f, 3.4028235e38f, -3.4028235e38f, 1.0f, -1.0f, 0.5f};
    for (size_t i = 0; i < n && i < sizeof(special) / sizeof(float); ++i) v[i] = special[i];
    return v;
}

}  // namespace

// ================================================================ reference self-checks

TEST(test_reference_checksums_match_published_vectors) {
    // Anchors the independent implementations used as oracles below.
    const char* s = "123456789";
    CHECK(ref_crc32(reinterpret_cast<const uint8_t*>(s), 9) == 0xCBF43926u);
    const char* w = "Wikipedia";
    CHECK(ref_adler32(reinterpret_cast<const uint8_t*>(w), 9) == 0x11E60398u);
    const uint8_t iend[4] = {'I', 'E', 'N', 'D'};
    CHECK(ref_crc32(iend, 4) == 0xAE426082u);
}

// ================================================================ PNG

TEST(png_1x1_red_pixel_has_hand_computed_adler_and_iend_crc) {
    // raw stream = 00 (filter) FF 00 00. Adler-32: s1 = 1+0+255+0+0 = 256 = 0x0100;
    // s2 = (1)+(256)+(256)+(256) = 769 = 0x0301 -> 0x03010100. IEND CRC is the well-known 0xAE426082.
    std::vector<uint8_t> rgb{255, 0, 0};
    std::string path = tmp("red1x1.png");
    wf::write_png(path, 1, 1, rgb);
    std::vector<uint8_t> got;
    PngInfo info;
    if (!decode_png(path, 1, 1, &got, &info)) return;
    CHECK(got == rgb);
    CHECK(info.adler == 0x03010100u);
    CHECK(info.iend_crc == 0xAE426082u);
}

TEST(png_3x5_decodes_to_the_input_pixels) { png_round_trip(3, 5, 11, 1); }

TEST(png_257x3_decodes_to_the_input_pixels) { png_round_trip(257, 3, 12, 1); }

TEST(png_300x300_spans_several_stored_blocks_and_decodes_to_the_input_pixels) {
    // 300 * (1 + 900) = 270300 raw bytes; a stored block holds at most 65535, so ceil(270300/65535) = 5 blocks.
    png_round_trip(300, 300, 13, 5);
}

TEST(png_stream_of_exactly_65535_bytes_decodes_correctly) {
    // 28 * (1 + 84) ... rows: 771 * (1 + 3*28) = 771 * 85 = 65535: the largest single stored block.
    png_round_trip(28, 771, 14, 1);
}

TEST(png_stream_of_exactly_65536_bytes_needs_a_second_block_and_decodes_correctly) {
    // 1024 * (1 + 3*21) = 1024 * 64 = 65536: one byte over a single block (off-by-one at the block limit).
    png_round_trip(21, 1024, 15, 2);
}

TEST(png_rejects_pixel_buffer_of_wrong_size_and_writes_no_file) {
    std::string p1 = tmp("short.png"), p2 = tmp("long.png"), p3 = tmp("empty.png");
    std::vector<uint8_t> ok = make_rgb(4, 3, 1);
    std::vector<uint8_t> shorter(ok.begin(), ok.end() - 1);
    std::vector<uint8_t> longer = ok; longer.push_back(0);
    bool t1 = false, t2 = false, t3 = false;
    try { wf::write_png(p1, 4, 3, shorter); } catch (const std::runtime_error&) { t1 = true; }
    try { wf::write_png(p2, 4, 3, longer); } catch (const std::runtime_error&) { t2 = true; }
    try { wf::write_png(p3, 4, 3, std::vector<uint8_t>{}); } catch (const std::runtime_error&) { t3 = true; }
    CHECK(t1);
    CHECK(t2);
    CHECK(t3);
    // state unchanged: a rejected call leaves no (half-written) file behind
    CHECK(!fs::exists(p1));
    CHECK(!fs::exists(p2));
    CHECK(!fs::exists(p3));
    // Positive counterpart: the same call with the correct size succeeds, so size was the only cause.
    wf::write_png(p1, 4, 3, ok);
    CHECK(fs::exists(p1));
}

TEST(png_to_unwritable_path_throws) {
    std::string bad = tmp("no_such_dir/deeper/x.png");
    bool threw = false;
    try { wf::write_png(bad, 1, 1, std::vector<uint8_t>{1, 2, 3}); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    CHECK(!fs::exists(bad));
    // Same call into an existing directory works, so the missing directory was the only cause.
    wf::write_png(tmp("fine.png"), 1, 1, std::vector<uint8_t>{1, 2, 3});
    CHECK(fs::exists(tmp("fine.png")));
}

// ================================================================ PFM

TEST(pfm_round_trip_is_bit_exact_including_negative_zero_and_denormals) {
    const int w = 7, h = 5;
    std::vector<float> img = random_floats(static_cast<size_t>(w) * h * 3, 99);
    std::string path = tmp("rt.pfm");
    wf::write_pfm(path, w, h, img);
    int rw = -1, rh = -1;
    std::vector<float> back = wf::read_pfm(path, &rw, &rh);
    CHECK(rw == w);
    CHECK(rh == h);
    CHECK(back.size() == img.size());
    size_t bad = 0;
    for (size_t i = 0; i < img.size() && i < back.size(); ++i)
        if (bits_of(img[i]) != bits_of(back[i])) ++bad;  // compare bit patterns: -0.0 and denormals included
    CHECK(bad == 0);
}

TEST(pfm_file_header_is_PF_with_negative_scale_and_correct_size) {
    const int w = 3, h = 2;
    std::vector<float> img(static_cast<size_t>(w) * h * 3, 0.25f);
    std::string path = tmp("hdr.pfm");
    wf::write_pfm(path, w, h, img);
    std::vector<uint8_t> f = read_bytes(path);
    CHECK(f.size() >= 2);
    CHECK(f.size() >= 2 && f[0] == 'P' && f[1] == 'F');
    PfmHeader H = parse_pfm_header(f);
    CHECK(H.ok);
    CHECK(H.magic == "PF");
    CHECK(H.w == w);
    CHECK(H.h == h);
    CHECK_NEAR(H.scale, -1.0, 0.0);  // negative scale = little endian
    CHECK(f.size() == H.data_offset + 12u * w * h);
}

TEST(pfm_stores_rows_bottom_to_top_so_first_float_is_bottom_left_pixel) {
    // 2x3 image, all 18 floats distinct: value = index. Memory order is top row first.
    const int w = 2, h = 3;
    std::vector<float> img(18);
    for (int i = 0; i < 18; ++i) img[i] = static_cast<float>(i);
    std::string path = tmp("order.pfm");
    wf::write_pfm(path, w, h, img);
    std::vector<uint8_t> f = read_bytes(path);
    PfmHeader H = parse_pfm_header(f);
    if (!H.ok || f.size() < H.data_offset + 72) { CHECK(false); return; }
    auto file_float = [&](int idx) { float x; std::memcpy(&x, &f[H.data_offset + 4 * idx], 4); return x; };
    // Bottom-left pixel = row 2, col 0 = img[(2*2+0)*3 ..] = 12, 13, 14.
    CHECK_NEAR(file_float(0), 12.0, 0.0);
    CHECK_NEAR(file_float(1), 13.0, 0.0);
    CHECK_NEAR(file_float(2), 14.0, 0.0);
    // Last row in the file is the top row: img[0..5] = 0..5.
    CHECK_NEAR(file_float(12), 0.0, 0.0);
    CHECK_NEAR(file_float(17), 5.0, 0.0);
    // And the reader undoes the flip.
    int rw = 0, rh = 0;
    std::vector<float> back = wf::read_pfm(path, &rw, &rh);
    CHECK(back == img);
}

TEST(pfm_reader_accepts_big_endian_positive_scale_files) {
    // Hand-written 1x2 file: "PF\n1 2\n1.0\n" then rows bottom-to-top, big-endian floats.
    // Bottom pixel (1,2,3), top pixel (4,5,6) -> read_pfm returns top row first: 4 5 6 1 2 3.
    std::string header = "PF\n1 2\n1.0\n";
    std::vector<uint8_t> f(header.begin(), header.end());
    const float vals[6] = {1, 2, 3, 4, 5, 6};
    for (float v : vals) {
        uint32_t u = bits_of(v);
        f.push_back(u >> 24); f.push_back((u >> 16) & 0xFF); f.push_back((u >> 8) & 0xFF); f.push_back(u & 0xFF);
    }
    std::string path = tmp("be.pfm");
    write_bytes(path, f);
    int w = 0, h = 0;
    std::vector<float> got = wf::read_pfm(path, &w, &h);
    CHECK(w == 1);
    CHECK(h == 2);
    const float want[6] = {4, 5, 6, 1, 2, 3};
    CHECK(got.size() == 6);
    for (size_t i = 0; i < got.size() && i < 6; ++i) CHECK_NEAR(got[i], want[i], 0.0);
}

TEST(read_pfm_missing_file_throws) {
    int w = 0, h = 0;
    CHECK_THROWS(wf::read_pfm(tmp("does_not_exist.pfm"), &w, &h));
}

TEST(read_pfm_truncated_file_throws_but_the_full_file_reads) {
    std::vector<float> img = random_floats(4 * 3 * 3, 5);
    std::string good = tmp("full.pfm");
    wf::write_pfm(good, 4, 3, img);
    std::vector<uint8_t> f = read_bytes(good);
    int w = 0, h = 0;
    CHECK(wf::read_pfm(good, &w, &h).size() == img.size());  // positive counterpart: only truncation differs
    std::string cut = tmp("cut.pfm");
    write_bytes(cut, std::vector<uint8_t>(f.begin(), f.end() - 1));
    CHECK_THROWS(wf::read_pfm(cut, &w, &h));
    std::string cut_hdr = tmp("cut_hdr.pfm");
    write_bytes(cut_hdr, std::vector<uint8_t>(f.begin(), f.begin() + 4));
    CHECK_THROWS(wf::read_pfm(cut_hdr, &w, &h));
    std::string empty = tmp("empty.pfm");
    write_bytes(empty, {});
    CHECK_THROWS(wf::read_pfm(empty, &w, &h));
}

TEST(read_pfm_wrong_magic_throws) {
    // Same payload as a valid file, only the magic differs.
    std::string header_ok = "PF\n1 1\n-1.0\n";
    std::string header_p6 = "P6\n1 1\n-1.0\n";
    std::string header_xx = "XX\n1 1\n-1.0\n";
    const float px[3] = {1.0f, 2.0f, 3.0f};
    auto make = [&](const std::string& hd) {
        std::vector<uint8_t> f(hd.begin(), hd.end());
        const uint8_t* b = reinterpret_cast<const uint8_t*>(px);
        f.insert(f.end(), b, b + 12);
        return f;
    };
    write_bytes(tmp("magic_ok.pfm"), make(header_ok));
    write_bytes(tmp("magic_p6.pfm"), make(header_p6));
    write_bytes(tmp("magic_xx.pfm"), make(header_xx));
    int w = 0, h = 0;
    CHECK(wf::read_pfm(tmp("magic_ok.pfm"), &w, &h).size() == 3);  // little-endian host assumed, as in write_pfm
    CHECK_THROWS(wf::read_pfm(tmp("magic_p6.pfm"), &w, &h));
    CHECK_THROWS(wf::read_pfm(tmp("magic_xx.pfm"), &w, &h));
}

// ================================================================ hashing

TEST(fnv1a64_matches_published_vectors) {
    // Offset basis for empty input; "a" and "foobar" from the FNV reference test suite.
    CHECK(wf::fnv1a64("", 0) == 0xcbf29ce484222325ull);
    CHECK(wf::fnv1a64("a", 1) == 0xaf63dc4c8601ec8cull);
    CHECK(wf::fnv1a64("foobar", 6) == 0x85944171f73967e8ull);
}

TEST(fnv1a64_is_order_sensitive_and_handles_high_bytes) {
    CHECK(wf::fnv1a64("ab", 2) != wf::fnv1a64("ba", 2));
    // All 256 byte values: guards against signed-char sign extension. Oracle = independent loop above.
    std::vector<uint8_t> all(256);
    for (int i = 0; i < 256; ++i) all[i] = static_cast<uint8_t>(i);
    CHECK(wf::fnv1a64(all.data(), all.size()) == ref_fnv1a64(all.data(), all.size()));
    // Only the length argument selects the bytes hashed (not NUL termination).
    CHECK(wf::fnv1a64("foobarXYZ", 6) == 0x85944171f73967e8ull);
}

TEST(hex64_is_sixteen_lowercase_digits_zero_padded) {
    CHECK(wf::hex64(0) == "0000000000000000");
    CHECK(wf::hex64(0xABCDEFull) == "0000000000abcdef");
    CHECK(wf::hex64(0xcbf29ce484222325ull) == "cbf29ce484222325");
    CHECK(wf::hex64(~0ull) == "ffffffffffffffff");
    CHECK(wf::hex64(1) == "0000000000000001");
    CHECK(wf::hex64(0xaf63dc4c8601ec8cull).size() == 16);
}

TEST(cleanup_temporary_files) {
    std::error_code ec;
    fs::remove_all(tmp_dir(), ec);
    CHECK(!ec);
}
