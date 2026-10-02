// failure mapping and artifacts: command-line parsing, report.json schema, percentile,
// PNG / PFM writers (decoded here by independent code), FNV-1a, and the real `wf` binary.
// The binary is taken from env WF_BIN (default build/wf); the working directory is ROOT.
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "wf/config.hpp"
#include "wf/film.hpp"
#include "wf/image_io.hpp"
#include "wf/json.hpp"
#include "wf/renderer.hpp"
#include "wf/report.hpp"
#include "wf/scene.hpp"

using namespace wf;
namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------- shared small helpers
fs::path fresh_dir(const std::string& tag) {
    static int counter = 0;
    fs::path d = fs::temp_directory_path() /
                 ("wf_r1_2_s2_" + std::to_string((long)::getpid()) + "_" + tag + "_" + std::to_string(counter++));
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
std::vector<uint8_t> read_bytes(const fs::path& p) {
    std::string s = read_text(p);
    return std::vector<uint8_t>(s.begin(), s.end());
}

// Result of parse_args: the message of the thrown ConfigError, or "<no throw>" / "<other exception>".
// argv is given WITHOUT the program name; "wf" is prepended so {"render", ...} reads as `wf render ...`.
std::string parse_error(std::vector<const char*> argv) {
    argv.insert(argv.begin(), "wf");
    try {
        (void)parse_args((int)argv.size(), argv.data());
    } catch (const ConfigError& e) {
        return e.what();
    } catch (...) {
        return "<other exception>";
    }
    return "<no throw>";
}

// ---- independent checksum implementations (bitwise CRC32, plain Adler32) ----
uint32_t ref_crc32(const uint8_t* d, size_t n, uint32_t crc = 0) {
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) {
        crc ^= d[i];
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
uint32_t ref_adler32(const uint8_t* d, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; ++i) {
        a = (a + d[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}
uint64_t ref_fnv1a64(const uint8_t* d, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= d[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
uint32_t be32(const uint8_t* p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

struct DecodedPng {
    std::string error;  // empty when the structure is valid
    uint32_t w = 0, h = 0;
    int stored_blocks = 0;
    std::vector<uint8_t> rgb;  // 3*w*h recovered through the stored blocks
};

// Structural decode: signature, chunk CRCs, IHDR fields, IDAT concatenation, zlib header, stored
// deflate blocks (BFINAL only on the last), Adler32, scanline filter bytes.
DecodedPng decode_png(const std::vector<uint8_t>& f) {
    DecodedPng out;
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    if (f.size() < 8 || std::memcmp(f.data(), sig, 8) != 0) { out.error = "bad signature"; return out; }
    size_t pos = 8;
    std::vector<uint8_t> idat;
    bool have_ihdr = false, have_iend = false;
    int chunk_index = 0;
    while (pos < f.size()) {
        if (pos + 12 > f.size()) { out.error = "truncated chunk header"; return out; }
        uint32_t len = be32(&f[pos]);
        if (pos + 12 + (size_t)len > f.size()) { out.error = "chunk overruns file"; return out; }
        std::string type((const char*)&f[pos + 4], 4);
        uint32_t crc = be32(&f[pos + 8 + len]);
        if (ref_crc32(&f[pos + 4], 4 + (size_t)len) != crc) { out.error = "CRC32 mismatch in " + type; return out; }
        const uint8_t* data = &f[pos + 8];
        if (chunk_index == 0) {
            if (type != "IHDR" || len != 13) { out.error = "first chunk is not a 13 byte IHDR"; return out; }
            out.w = be32(data);
            out.h = be32(data + 4);
            if (data[8] != 8 || data[9] != 2 || data[10] != 0 || data[11] != 0 || data[12] != 0) {
                out.error = "IHDR is not 8-bit RGB, deflate, filter 0, no interlace";
                return out;
            }
            have_ihdr = true;
        } else if (type == "IDAT") {
            idat.insert(idat.end(), data, data + len);
        } else if (type == "IEND") {
            if (len != 0) { out.error = "IEND has data"; return out; }
            have_iend = true;
            pos += 12;
            break;
        }
        pos += 12 + len;
        ++chunk_index;
    }
    if (!have_ihdr) { out.error = "no IHDR"; return out; }
    if (!have_iend) { out.error = "no IEND"; return out; }
    if (pos != f.size()) { out.error = "bytes after IEND"; return out; }
    if (idat.size() < 6) { out.error = "IDAT too short"; return out; }
    if ((idat[0] & 0x0F) != 8 || ((idat[0] * 256u + idat[1]) % 31u) != 0) { out.error = "bad zlib header"; return out; }
    if (idat[1] & 0x20) { out.error = "preset dictionary"; return out; }
    size_t q = 2;
    std::vector<uint8_t> raw;
    bool final_seen = false;
    while (!final_seen) {
        if (q + 5 > idat.size()) { out.error = "truncated stored block header"; return out; }
        uint8_t hdr = idat[q++];
        final_seen = hdr & 1;
        if (((hdr >> 1) & 3) != 0) { out.error = "deflate block is not stored"; return out; }
        uint16_t len = (uint16_t)(idat[q] | idat[q + 1] << 8);
        uint16_t nlen = (uint16_t)(idat[q + 2] | idat[q + 3] << 8);
        q += 4;
        if ((uint16_t)~len != nlen) { out.error = "LEN/NLEN mismatch"; return out; }
        if (q + len > idat.size()) { out.error = "stored block overruns IDAT"; return out; }
        raw.insert(raw.end(), idat.begin() + q, idat.begin() + q + len);
        q += len;
        ++out.stored_blocks;
    }
    if (q + 4 != idat.size()) { out.error = "zlib trailer is not exactly the 4 byte Adler32"; return out; }
    if (be32(&idat[q]) != ref_adler32(raw.data(), raw.size())) { out.error = "Adler32 mismatch"; return out; }
    size_t stride = 1 + 3 * (size_t)out.w;
    if (raw.size() != stride * out.h) { out.error = "raw size != h*(1+3w)"; return out; }
    for (uint32_t y = 0; y < out.h; ++y) {
        if (raw[y * stride] != 0) { out.error = "scanline filter byte is not 0"; return out; }
        out.rgb.insert(out.rgb.end(), raw.begin() + y * stride + 1, raw.begin() + (y + 1) * stride);
    }
    return out;
}

std::vector<uint8_t> pattern_rgb8(int w, int h) {
    std::vector<uint8_t> v((size_t)3 * w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) v[((size_t)y * w + x) * 3 + c] = (uint8_t)((x * 37 + y * 11 + c * 101 + (x * y) % 7) % 256);
    v[0] = 0;
    v[v.size() - 1] = 255;
    return v;
}

// ---- JSON text surgery for corrupting a valid report without relying on its whitespace ----
// Replaces the value of the first `"key"` (scalar token, or a flat [...] array) with `repl`.
std::string replace_value(const std::string& text, const std::string& key, const std::string& repl) {
    size_t k = text.find("\"" + key + "\"");
    REQUIRE(k != std::string::npos);
    size_t c = text.find(':', k);
    REQUIRE(c != std::string::npos);
    size_t b = c + 1;
    while (b < text.size() && std::isspace((unsigned char)text[b])) ++b;
    size_t e;
    if (text[b] == '[') e = text.find(']', b) + 1;
    else if (text[b] == '"') e = text.find('"', b + 1) + 1;
    else { e = b; while (e < text.size() && text[e] != ',' && text[e] != '}' && text[e] != ']' && !std::isspace((unsigned char)text[e])) ++e; }
    return text.substr(0, b) + repl + text.substr(e);
}

std::string rename_key(std::string text, const std::string& from, const std::string& to) {
    size_t k = text.find("\"" + from + "\"");
    REQUIRE(k != std::string::npos);
    return text.replace(k + 1, from.size(), to);
}

Report gpu_report() {
    Report r;
    r.git_revision = "0123abc";
    r.scene_name = "cornell";
    r.scene_hash = 0x0123456789abcdefULL;
    r.width = 1280; r.height = 720; r.spp = 64; r.max_depth = 8; r.seed = 1234;
    r.mode = "wavefront";
    r.device.name = "NVIDIA GeForce GTX 1650 Ti";
    r.device.driver_version = "555.42";
    r.device.cuda_runtime_version = "12.4";
    r.device.cuda_driver_version = "12.5";
    r.device.compute_capability = "7.5";
    r.device.total_memory_bytes = 4294967296ULL;
    r.run_ms = {10.5, 11.0, 10.8, 10.9, 11.2};
    r.samples_per_second = 5.4e9;
    r.rays_per_second = 1.5e8;
    r.peak_vram_bytes = 1234567890ULL;
    long long qh[5] = {100, 200, 300, 400, 500};
    for (int i = 0; i < 5; ++i) r.queue_high_water[i] = qh[i];
    r.image_png = "out/cornell.png";
    r.image_pfm = "out/cornell.pfm";
    r.pfm_hash = 0xfedcba9876543210ULL;
    return r;
}

Report cpu_report() {
    Report r;  // git revision empty (null), device defaults to the cpu placeholder, no ray count
    r.scene_name = "furnace";
    r.scene_hash = 0x1ULL;
    r.width = 16; r.height = 9; r.spp = 4; r.max_depth = 8; r.seed = 1234;
    r.mode = "cpu";
    r.run_ms = {123.0};
    r.samples_per_second = 4.7e3;
    r.pfm_hash = 0xcbf29ce484222325ULL;
    return r;
}

int exit_code_of(int system_status) {
    if (system_status == -1 || !WIFEXITED(system_status)) return -1000;  // killed by a signal or not started
    return WEXITSTATUS(system_status);
}

std::string wf_bin() {
    const char* e = std::getenv("WF_BIN");
    return e && *e ? e : "build/wf";
}

struct CliRun {
    int code;
    std::string err, out;
};
CliRun run_cli(const fs::path& dir, const std::string& args) {
    fs::path e = dir / "stderr.txt", o = dir / "stdout.txt";
    std::string cmd = "\"" + wf_bin() + "\" " + args + " 2>\"" + e.string() + "\" >\"" + o.string() + "\"";
    int st = std::system(cmd.c_str());
    return {exit_code_of(st), read_text(e), read_text(o)};
}

}  // namespace

// =================================================================================================
TEST_CASE("parse_args rejects each documented invalid input with a message naming the flag",
          "[cli_artifacts]") {
    // Given exactly one invalid element per invocation (all else defaulted/valid)
    // When parse_args runs  Then it throws ConfigError whose message names the offending flag and
    // does not name any other flag (so a usage dump cannot satisfy the check vacuously).
    struct Bad {
        std::vector<const char*> argv;
        const char* flag;
    };
    const std::vector<Bad> bad = {
        {{"render", "--bogus", "1"}, "--bogus"},
        {{"render", "--width"}, "--width"},                       // missing value at end
        {{"render", "--spp"}, "--spp"},
        {{"render", "--width", "abc"}, "--width"},                // non-numeric
        {{"render", "--height", "9x"}, "--height"},
        {{"render", "--seed", "seed"}, "--seed"},
        {{"render", "--width", "0"}, "--width"},
        {{"render", "--width", "8193"}, "--width"},
        {{"render", "--height", "0"}, "--height"},
        {{"render", "--height", "8193"}, "--height"},
        {{"render", "--spp", "0"}, "--spp"},
        {{"render", "--max-depth", "0"}, "--max-depth"},
        {{"render", "--max-depth", "65"}, "--max-depth"},
        {{"render", "--reps", "0"}, "--reps"},
        {{"render", "--warmup", "-1"}, "--warmup"},
        {{"render", "--mode", "gpu"}, "--mode"},
    };
    const char* all_flags[] = {"--scene", "--width", "--height", "--spp", "--seed", "--max-depth",
                               "--mode", "--report", "--out", "--threads", "--reps", "--warmup"};
    for (const Bad& b : bad) {
        std::string msg = parse_error(b.argv);
        INFO("invocation with flag " << b.flag << " -> message: " << msg);
        REQUIRE(msg != "<no throw>");
        REQUIRE(msg != "<other exception>");
        CHECK(msg.find(b.flag) != std::string::npos);
        for (const char* other : all_flags)
            if (std::strcmp(other, b.flag) != 0 && std::strncmp(b.flag, "--bogus", 7) != 0)
                CHECK(msg.find(other) == std::string::npos);
    }
    // argv[1] must be "render".
    for (const std::vector<const char*>& v : {std::vector<const char*>{"cornell"}, std::vector<const char*>{}}) {
        std::string msg = parse_error(v);
        CHECK(msg != "<no throw>");
        CHECK(msg != "<other exception>");
        CHECK(!msg.empty());
    }
}

TEST_CASE("parse_args defaults and boundary-legal values are accepted exactly", "[cli_artifacts]") {
    // Defaults are the values documented in config.hpp; every flag is stored in its own field;
    // the legal side of each boundary (1 and 8192, 1 and 64, warmup 0) is accepted.
    std::vector<const char*> none = {"wf", "render"};
    Config d = parse_args((int)none.size(), none.data());
    CHECK(d.scene == "cornell");
    CHECK(d.width == 1280);
    CHECK(d.height == 720);
    CHECK(d.spp == 64);
    CHECK(d.seed == 1234u);
    CHECK(d.max_depth == 8);
    CHECK(d.mode == Mode::Wavefront);
    CHECK(d.report_path.empty());
    CHECK(d.out_prefix.empty());
    CHECK(d.threads == 0);
    CHECK(d.reps == 1);
    CHECK(d.warmup == 0);

    std::vector<const char*> all = {"wf", "render", "--scene", "furnace", "--width", "16", "--height", "9",
                                    "--spp", "4", "--seed", "7", "--max-depth", "3", "--mode", "cpu",
                                    "--report", "r.json", "--out", "pre", "--threads", "2",
                                    "--reps", "3", "--warmup", "1"};
    Config c = parse_args((int)all.size(), all.data());
    CHECK(c.scene == "furnace");
    CHECK(c.width == 16);
    CHECK(c.height == 9);
    CHECK(c.spp == 4);
    CHECK(c.seed == 7u);
    CHECK(c.max_depth == 3);
    CHECK(c.mode == Mode::Cpu);
    CHECK(c.report_path == "r.json");
    CHECK(c.out_prefix == "pre");
    CHECK(c.threads == 2);
    CHECK(c.reps == 3);
    CHECK(c.warmup == 1);

    std::vector<const char*> lo = {"wf", "render", "--width", "1", "--height", "1", "--spp", "1",
                                   "--max-depth", "1", "--reps", "1", "--warmup", "0"};
    Config l = parse_args((int)lo.size(), lo.data());
    CHECK((l.width == 1 && l.height == 1 && l.spp == 1 && l.max_depth == 1 && l.reps == 1 && l.warmup == 0));
    std::vector<const char*> hi = {"wf", "render", "--width", "8192", "--height", "8192", "--max-depth", "64"};
    Config hc = parse_args((int)hi.size(), hi.data());
    CHECK((hc.width == 8192 && hc.height == 8192 && hc.max_depth == 64));

    // mode_name is the inverse of the --mode spelling.
    for (Mode m : {Mode::Cpu, Mode::Megakernel, Mode::Wavefront}) {
        std::string name = mode_name(m);
        std::vector<const char*> a = {"wf", "render", "--mode", name.c_str()};
        CHECK(parse_args((int)a.size(), a.data()).mode == m);
    }
    CHECK(mode_name(Mode::Cpu) == "cpu");
    CHECK(mode_name(Mode::Megakernel) == "megakernel");
    CHECK(mode_name(Mode::Wavefront) == "wavefront");
}

TEST_CASE("report json round trips for gpu and cpu reports and eight corruptions are each rejected",
          "[cli_artifacts]") {
    // Given a valid gpu-like report, a cpu report; When serialised and validated Then "" (valid) and
    // the parsed values equal the inputs (hand-computed percentiles of {10.5,10.8,10.9,11.0,11.2}).
    // Then each of eight single-defect corruptions of the valid text is rejected with a message.
    std::string gj = report_to_json(gpu_report());
    CHECK(validate_report_json(gj) == "");
    Json g = json_parse(gj);
    CHECK(g.at("schema_version").num == 1.0);
    CHECK(g.at("git_revision").str == "0123abc");
    CHECK(g.at("scene").at("name").str == "cornell");
    CHECK(g.at("scene").at("hash").str == "0123456789abcdef");
    CHECK(g.at("config").at("width").num == 1280.0);
    CHECK(g.at("config").at("mode").str == "wavefront");
    CHECK(g.at("device").at("compute_capability").str == "7.5");
    const Json& t = g.at("timings_ms");
    CHECK(t.at("runs").arr.size() == 5);
    CHECK(std::fabs(t.at("min").num - 10.5) < 1e-9);
    CHECK(std::fabs(t.at("median").num - 10.9) < 1e-9);
    CHECK(std::fabs(t.at("p10").num - 10.62) < 1e-9);   // pos 0.4: 10.5 + 0.4*0.3
    CHECK(std::fabs(t.at("p90").num - 11.12) < 1e-9);   // pos 3.6: 11.0 + 0.6*0.2
    CHECK(std::fabs(t.at("max").num - 11.2) < 1e-9);
    CHECK(g.at("outputs").at("pfm_hash").str == "fedcba9876543210");
    CHECK(g.at("queue_high_water").at("shadow").num == 400.0);
    CHECK(g.at("validity").at("nonfinite_pixels").num == 0.0);

    std::string cj = report_to_json(cpu_report());
    CHECK(validate_report_json(cj) == "");
    Json c = json_parse(cj);
    CHECK(c.at("git_revision").type == Json::Null);
    CHECK(c.at("throughput").at("rays_per_second").type == Json::Null);
    CHECK(c.at("device").at("driver_version").type == Json::String);
    CHECK(c.at("device").at("driver_version").str.empty());
    CHECK(c.at("config").at("mode").str == "cpu");
    CHECK(c.at("timings_ms").at("median").num == 123.0);

    struct Corrupt {
        const char* what;
        std::string text;
    };
    const std::vector<Corrupt> corrupt = {
        {"schema_version 2", replace_value(gj, "schema_version", "2")},
        {"required key renamed away", rename_key(gj, "peak_vram_bytes", "peak_vram_bytez")},
        {"width is a string", replace_value(gj, "width", "\"1280\"")},
        {"runs empty", replace_value(gj, "runs", "[]")},
        {"min above median", replace_value(gj, "min", "10.95")},
        {"max below median", replace_value(gj, "max", "10.7")},
        {"pfm_hash not 16 hex digits", replace_value(gj, "pfm_hash", "\"12345\"")},
        {"truncated document", gj.substr(0, gj.size() / 2)},
    };
    REQUIRE(corrupt.size() == 8);
    for (const Corrupt& k : corrupt) {
        INFO("corruption: " << k.what);
        REQUIRE(k.text != gj);
        CHECK(!validate_report_json(k.text).empty());
    }
}

TEST_CASE("percentile uses linear interpolation between order statistics", "[cli_artifacts]") {
    // Oracle: position p*(n-1) between sorted neighbours, computed by hand.
    CHECK(std::fabs(percentile({1, 2, 3, 4, 5}, 0.5) - 3.0) < 1e-12);
    CHECK(std::fabs(percentile({1, 2, 3, 4, 5}, 0.1) - 1.4) < 1e-12);   // pos 0.4
    CHECK(std::fabs(percentile({1, 2, 3, 4, 5}, 0.9) - 4.6) < 1e-12);   // pos 3.6
    CHECK(std::fabs(percentile({5, 1, 4, 2, 3}, 0.1) - 1.4) < 1e-12);   // input order irrelevant
    CHECK(std::fabs(percentile({10, 20}, 0.5) - 15.0) < 1e-12);
    CHECK(std::fabs(percentile({1, 2, 3, 4}, 0.5) - 2.5) < 1e-12);
    CHECK(std::fabs(percentile({1, 2, 3, 4}, 0.25) - 1.75) < 1e-12);    // pos 0.75
    CHECK(percentile({9, 3, 7}, 0.0) == 3.0);                            // minimum
    CHECK(percentile({9, 3, 7}, 1.0) == 9.0);                            // maximum
    CHECK(percentile({7}, 0.37) == 7.0);
    CHECK_THROWS(percentile({}, 0.5));
}

TEST_CASE("write_png output decodes structurally and recovers every pixel (1x1, 3x5, 300x300)",
          "[cli_artifacts]") {
    // Independent oracle: this file's CRC32 / Adler32 / stored-block decoder. First its own
    // self-check on published vectors, so the decoder cannot be wrong in the same way as the writer.
    const uint8_t nine[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    REQUIRE(ref_crc32(nine, 9) == 0xCBF43926u);
    const uint8_t wiki[] = {'W', 'i', 'k', 'i', 'p', 'e', 'd', 'i', 'a'};
    REQUIRE(ref_adler32(wiki, 9) == 0x11E60398u);

    fs::path dir = fresh_dir("png");
    const int sizes[3][2] = {{1, 1}, {3, 5}, {300, 300}};
    for (auto& s : sizes) {
        int w = s[0], h = s[1];
        std::vector<uint8_t> px = pattern_rgb8(w, h);
        fs::path p = dir / ("img_" + std::to_string(w) + "x" + std::to_string(h) + ".png");
        write_png(p.string(), w, h, px);
        DecodedPng d = decode_png(read_bytes(p));
        INFO("size " << w << "x" << h << " decode error: " << d.error);
        REQUIRE(d.error.empty());
        CHECK(d.w == (uint32_t)w);
        CHECK(d.h == (uint32_t)h);
        CHECK(d.rgb == px);
        size_t raw = (size_t)h * (1 + 3 * (size_t)w);
        CHECK((size_t)d.stored_blocks >= (raw + 65534) / 65535);  // stored blocks hold at most 65535 bytes
        if (w == 300) CHECK(d.stored_blocks >= 5);                // 270300 raw bytes: multi-block path
    }
    // Negative: a pixel buffer of the wrong size throws and leaves no file behind.
    fs::path bad = dir / "bad.png";
    std::vector<uint8_t> wrong(10, 7);
    bool threw = false;
    try { write_png(bad.string(), 3, 5, wrong); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    CHECK(!fs::exists(bad));
    fs::remove_all(dir);
}

TEST_CASE("PFM round trip is bit exact and rows are stored bottom to top", "[cli_artifacts]") {
    fs::path dir = fresh_dir("pfm");
    // Awkward floats: negative, -0.0, denormals, FLT_MAX, FLT_MIN, infinity.
    const float specials[] = {-1.5f, -0.0f, 0.0f, 1e-40f, -1e-42f, 3.4028235e38f, 1.17549435e-38f,
                              std::numeric_limits<float>::infinity(), 0.1f, 123456.789f};
    const int dims[3][2] = {{1, 1}, {2, 3}, {7, 5}};
    for (auto& d : dims) {
        int w = d[0], h = d[1];
        std::vector<float> img((size_t)3 * w * h);
        for (size_t i = 0; i < img.size(); ++i) img[i] = specials[i % 10] * (float)(1 + i / 10 % 3);
        fs::path p = dir / "t.pfm";
        write_pfm(p.string(), w, h, img);
        int rw = 0, rh = 0;
        std::vector<float> back = read_pfm(p.string(), &rw, &rh);
        INFO("size " << w << "x" << h);
        CHECK(rw == w);
        CHECK(rh == h);
        REQUIRE(back.size() == img.size());
        CHECK(std::memcmp(back.data(), img.data(), img.size() * sizeof(float)) == 0);
    }
    // Raw layout, parsed here: "PF", w h, negative scale (little endian), then rows bottom-to-top.
    {
        int w = 2, h = 3;
        std::vector<float> img(18);
        for (int i = 0; i < 18; ++i) img[i] = (float)i;  // row y holds floats 6y .. 6y+5
        fs::path p = dir / "layout.pfm";
        write_pfm(p.string(), w, h, img);
        std::vector<uint8_t> bytes = read_bytes(p);
        std::istringstream hs(std::string(bytes.begin(), bytes.end()));
        std::string magic; int hw, hh; double scale;
        hs >> magic >> hw >> hh >> scale;
        CHECK(magic == "PF");
        CHECK(hw == 2);
        CHECK(hh == 3);
        CHECK(scale < 0.0);
        size_t data_off = bytes.size() - 18 * sizeof(float);
        CHECK(data_off > 0);
        float first_row[6];
        std::memcpy(first_row, bytes.data() + data_off, 6 * sizeof(float));
        for (int i = 0; i < 6; ++i) CHECK(first_row[i] == (float)(12 + i));  // image row h-1 first
        float last_row[6];
        std::memcpy(last_row, bytes.data() + bytes.size() - 6 * sizeof(float), 6 * sizeof(float));
        for (int i = 0; i < 6; ++i) CHECK(last_row[i] == (float)i);           // image row 0 last
    }
    // Big-endian generic PF (positive scale) is read; a hand-built 2x1 image.
    {
        fs::path p = dir / "be.pfm";
        std::string hdr = "PF\n2 1\n1.0\n";
        std::vector<uint8_t> bytes(hdr.begin(), hdr.end());
        const uint8_t be_vals[6][4] = {{0x3f, 0x80, 0, 0}, {0x40, 0, 0, 0}, {0x40, 0x40, 0, 0},
                                       {0xbf, 0x80, 0, 0}, {0, 0, 0, 0}, {0x3f, 0, 0, 0}};  // 1,2,3,-1,0,0.5
        for (auto& v : be_vals) bytes.insert(bytes.end(), v, v + 4);
        std::ofstream(p, std::ios::binary).write((const char*)bytes.data(), (std::streamsize)bytes.size());
        int rw = 0, rh = 0;
        std::vector<float> back = read_pfm(p.string(), &rw, &rh);
        REQUIRE(back.size() == 6);
        const float expect[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 0.5f};
        for (int i = 0; i < 6; ++i) CHECK(back[i] == expect[i]);
        CHECK((rw == 2 && rh == 1));
    }
    // Negative: a missing file and a truncated pixel block both throw.
    {
        int rw = 0, rh = 0;
        bool threw_missing = false;
        try { read_pfm((dir / "nope.pfm").string(), &rw, &rh); } catch (const std::exception&) { threw_missing = true; }
        CHECK(threw_missing);
        fs::path p = dir / "trunc.pfm";
        std::string content = "PF\n2 2\n-1.0\n";
        content.append(7, '\0');  // needs 48 bytes
        std::ofstream(p, std::ios::binary) << content;
        bool threw_trunc = false;
        try { read_pfm(p.string(), &rw, &rh); } catch (const std::exception&) { threw_trunc = true; }
        CHECK(threw_trunc);
    }
    fs::remove_all(dir);
}

TEST_CASE("fnv1a64 matches published vectors and an independent implementation; hex64 formats 16 digits",
          "[cli_artifacts]") {
    // Published FNV-1a 64 test vectors (Noll): "" , "a", "foobar".
    CHECK(fnv1a64("", 0) == 0xcbf29ce484222325ULL);
    CHECK(fnv1a64("a", 1) == 0xaf63dc4c8601ec8cULL);
    CHECK(fnv1a64("foobar", 6) == 0x85944171f73967e8ULL);
    // Independent bytewise implementation on varied buffers, including bytes >= 0x80.
    std::vector<uint8_t> buf(1000);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t)(i * 131 + 7);
    for (size_t n : {size_t(1), size_t(2), size_t(7), size_t(8), size_t(9), size_t(1000)})
        CHECK(fnv1a64(buf.data(), n) == ref_fnv1a64(buf.data(), n));
    // Order sensitive, and a float buffer hashes its raw bytes.
    const uint8_t ab[2] = {1, 2}, ba[2] = {2, 1};
    CHECK(fnv1a64(ab, 2) != fnv1a64(ba, 2));
    const float fl[3] = {1.0f, -2.5f, 1e-40f};
    CHECK(fnv1a64(fl, sizeof fl) == ref_fnv1a64((const uint8_t*)fl, sizeof fl));
    // hex64: exactly 16 lowercase hex digits, zero padded.
    CHECK(hex64(0xabcULL) == "0000000000000abc");
    CHECK(hex64(0xcbf29ce484222325ULL) == "cbf29ce484222325");
    CHECK(hex64(0ULL) == "0000000000000000");
}

TEST_CASE("wf render CLI writes a valid report and artifacts; invalid invocations fail loudly and write nothing",
          "[cli_artifacts]") {
    // Real binary from $WF_BIN (default build/wf). Cpu mode so no GPU is needed.
    fs::path dir = fresh_dir("cli");
    fs::path rep = dir / "r.json";
    std::string out_prefix = (dir / "img").string();
    CliRun ok = run_cli(dir, "render --scene furnace --width 16 --height 9 --spp 4 --mode cpu --report \"" +
                                 rep.string() + "\" --out \"" + out_prefix + "\"");
    INFO("stderr: " << ok.err);
    REQUIRE(ok.code == 0);
    REQUIRE(fs::exists(rep));
    REQUIRE(fs::exists(out_prefix + ".png"));
    REQUIRE(fs::exists(out_prefix + ".pfm"));

    std::string text = read_text(rep);
    CHECK(validate_report_json(text) == "");
    Json j = json_parse(text);
    CHECK(j.at("config").at("width").num == 16.0);
    CHECK(j.at("config").at("height").num == 9.0);
    CHECK(j.at("config").at("spp").num == 4.0);
    CHECK(j.at("config").at("mode").str == "cpu");
    CHECK(j.at("scene").at("name").str == "furnace");
    CHECK(j.at("scene").at("hash").str == hex64(scene_hash(make_scene("furnace", 16, 9))));
    CHECK(j.at("validity").at("nonfinite_pixels").num == 0.0);
    CHECK(j.at("validity").at("queue_overflows").num == 0.0);

    // pfm_hash is fnv1a64 of the float buffer actually written to disk.
    int pw = 0, ph = 0;
    std::vector<float> pfm = read_pfm(out_prefix + ".pfm", &pw, &ph);
    CHECK((pw == 16 && ph == 9));
    REQUIRE(pfm.size() == (size_t)(3 * 16 * 9));
    CHECK(j.at("outputs").at("pfm_hash").str == hex64(fnv1a64(pfm.data(), pfm.size() * sizeof(float))));

    // Wiring: defaults (seed 1234, max-depth 8) reach the renderer: the PFM equals an in-process
    // render_cpu of the same scene and parameters (bitwise: counter-based RNG).
    RenderParams rp;
    rp.spp = 4; rp.seed = 1234; rp.max_depth = 8; rp.threads = 0;
    RenderStats stats;
    RenderResult ref = render_cpu(make_scene("furnace", 16, 9), rp, &stats);
    REQUIRE(ref.rgb.size() == pfm.size());
    CHECK(std::memcmp(ref.rgb.data(), pfm.data(), pfm.size() * sizeof(float)) == 0);
    // PNG: structurally valid and equal to the tone-mapped buffer (exposure 1).
    DecodedPng png = decode_png(read_bytes(out_prefix + ".png"));
    INFO("png decode error: " << png.error);
    REQUIRE(png.error.empty());
    CHECK((png.w == 16u && png.h == 9u));
    CHECK(png.rgb == tonemap_to_srgb8(pfm, 1.0f));

    // Invalid invocations: non-zero exit (not a crash), message naming the problem on stderr, and
    // no artifacts written.
    struct Inv {
        const char* what;
        std::string args;
        std::string must_name;
    };
    const std::string tail = " --report \"" + (dir / "bad.json").string() + "\" --out \"" + (dir / "bad").string() + "\"";
    const std::vector<Inv> invs = {
        {"unknown flag", "render --scene furnace --width 8 --height 8 --spp 1 --mode cpu --bogus 1" + tail, "--bogus"},
        {"missing obj", "render --scene /nonexistent_dir/missing_scene.obj --width 8 --height 8 --spp 1 --mode cpu" + tail, "missing_scene.obj"},
        {"width zero", "render --scene furnace --width 0 --height 8 --spp 1 --mode cpu" + tail, "--width"},
    };
    for (const Inv& v : invs) {
        CliRun r = run_cli(dir, v.args);
        INFO(v.what << " -> exit " << r.code << ", stderr: " << r.err);
        CHECK(r.code > 0);
        CHECK(r.err.find(v.must_name) != std::string::npos);
        CHECK(!fs::exists(dir / "bad.json"));
        CHECK(!fs::exists(dir / "bad.png"));
        CHECK(!fs::exists(dir / "bad.pfm"));
    }
    fs::remove_all(dir);
}
