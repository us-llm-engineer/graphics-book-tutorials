// benchmark protocol execution, quality criteria, failure mapping, wfbench and imgdiff CLIs
//. Needs the GPU. Binaries come from env WFBENCH_BIN (default build/wfbench)
// and IMGDIFF_BIN (default build/imgdiff); the working directory is ROOT.
// Oracles: independent median/throughput recomputation from run_ms, the verdict rule restated
// independently, the documented thresholds, and hand-built PFM pairs with known relative error.
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "wf/bench.hpp"
#include "wf/image_io.hpp"
#include "wf/json.hpp"
#include "wf/renderer.hpp"
#include "wf/scene.hpp"

using namespace wf;
using Catch::Matchers::WithinRel;
namespace fs = std::filesystem;

namespace {

fs::path fresh_dir(const std::string& tag) {
    static int counter = 0;
    fs::path d = fs::temp_directory_path() /
                 ("wf_r2_2_s2_" + std::to_string((long)::getpid()) + "_" + tag + "_" + std::to_string(counter++));
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}
std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
int exit_code_of(int st) {
    if (st == -1 || !WIFEXITED(st)) return -1000;
    return WEXITSTATUS(st);
}
std::string bin_from_env(const char* env, const char* def) {
    const char* e = std::getenv(env);
    return e && *e ? e : def;
}
struct Cli { int code; std::string err, out; };
Cli run_tool(const std::string& bin, const fs::path& dir, const std::string& args) {
    fs::path e = dir / "stderr.txt", o = dir / "stdout.txt";
    std::string cmd = "\"" + bin + "\" " + args + " 2>\"" + e.string() + "\" >\"" + o.string() + "\"";
    int st = std::system(cmd.c_str());
    return {exit_code_of(st), read_text(e), read_text(o)};
}
Cli run_wfbench(const fs::path& d, const std::string& a) { return run_tool(bin_from_env("WFBENCH_BIN", "build/wfbench"), d, a); }
Cli run_imgdiff(const fs::path& d, const std::string& a) { return run_tool(bin_from_env("IMGDIFF_BIN", "build/imgdiff"), d, a); }

BenchConfig make_cfg(int spp, int reps, int warmup, int cpu_spp) {
    BenchConfig c;
    c.reps = reps;
    c.warmup = warmup;
    c.cpu_spp = cpu_spp;
    c.params.spp = spp;
    c.params.seed = 4242;
    c.params.max_depth = 5;
    return c;
}

const ConditionStats* cond(const BenchResult& r, const std::string& mode) {
    for (const auto& c : r.conditions) if (c.mode == mode) return &c;
    return nullptr;
}
const QualityCheck* check(const BenchResult& r, const std::string& name) {
    for (const auto& q : r.quality) if (q.name == name) return &q;
    return nullptr;
}
double stage_sum(const StageTimes& s) {
    return s.extend + s.classify + s.shade_lambert + s.shade_metal + s.shade_dielectric + s.shadow +
           s.compact + s.other;
}
double indep_median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

// =================================================================================================
TEST_CASE("run_benchmark on cornell returns ordered conditions, three quality checks and the config",
          "[bench_run]") {
    // Given cornell 64x36, spp 4, seed 4242, depth 5, reps 3, warmup 1, cpu_spp 4
    Scene scene = make_scene("cornell", 64, 36);
    BenchConfig cfg = make_cfg(4, 3, 1, 4);
    BenchResult r = run_benchmark(scene, cfg);
    // Then conditions are megakernel then wavefront, each with reps timed runs
    REQUIRE(r.conditions.size() == 2);
    CHECK(r.conditions[0].mode == "megakernel");
    CHECK(r.conditions[1].mode == "wavefront");
    for (const auto& c : r.conditions) {
        INFO(c.mode);
        REQUIRE(c.run_ms.size() == 3);
        for (double ms : c.run_ms) CHECK(ms > 0.0);
        // independent recomputation from run_ms (3 elements: the middle sorted value)
        CHECK(c.median == indep_median(c.run_ms));
        CHECK(c.median > 0.0);
        CHECK(c.min <= c.median);
        CHECK(c.median <= c.max);
        CHECK(c.p10 <= c.median);
        CHECK(c.median <= c.p90);
        CHECK_THAT(c.samples_per_second, WithinRel(64.0 * 36.0 * 4.0 / (c.median / 1000.0), 1e-9));
        CHECK(c.peak_vram_bytes > 0);
    }
    // Then the three quality checks exist with the documented thresholds
    REQUIRE(r.quality.size() == 3);
    const QualityCheck* bw = check(r, "megakernel_vs_wavefront_bitwise");
    const QualityCheck* cm = check(r, "cpu_vs_megakernel");
    const QualityCheck* cw = check(r, "cpu_vs_wavefront");
    REQUIRE(bw); REQUIRE(cm); REQUIRE(cw);
    CHECK(bw->threshold == kBitwiseRelTolerance);
    CHECK(cm->threshold == kSePixelFraction);
    CHECK(cw->threshold == kSePixelFraction);
    // pass flags agree with their own recorded value/threshold (bitwise: smaller is better; CPU: a
    // passing check cannot have fewer than 99.5% of pixels within 3 SE)
    CHECK(bw->pass == (bw->value <= bw->threshold));
    for (const QualityCheck* q : {cm, cw}) {
        INFO(q->name);
        CHECK(std::isfinite(q->value));
        if (q->pass) CHECK(q->value >= kSePixelFraction);
    }
    // Then the recorded configuration equals the requested one
    CHECK(r.scene_name == "cornell");
    CHECK(r.scene_hash == scene_hash(scene));
    CHECK(r.width == 64);
    CHECK(r.height == 36);
    CHECK(r.spp == 4);
    CHECK(r.seed == 4242u);
    CHECK(r.max_depth == 5);
    CHECK(r.reps == 3);
    CHECK(r.warmup == 1);
    CHECK_FALSE(r.device_name.empty());
    CHECK_FALSE(r.cuda_runtime_version.empty());
}

TEST_CASE("stage timings exist only for the wavefront and ray counting is switchable",
          "[bench_run]") {
    // Given the default flags (count_rays and stage_timing on)
    Scene scene = make_scene("cornell", 64, 36);
    BenchConfig on = make_cfg(4, 3, 1, 0);
    BenchResult a = run_benchmark(scene, on);
    const ConditionStats* mk = cond(a, "megakernel");
    const ConditionStats* wf = cond(a, "wavefront");
    REQUIRE(mk); REQUIRE(wf);
    // Then the megakernel has no stage timing (nothing to time), the wavefront has, rays counted
    CHECK(stage_sum(mk->stages) == 0.0);
    CHECK(stage_sum(wf->stages) > 0.0);
    CHECK(wf->stages.extend > 0.0);
    CHECK(mk->rays_per_second > 0.0);
    CHECK(wf->rays_per_second > 0.0);
    // Given both switches off, Then rays_per_second is 0 (not counted) and no stage time is kept
    BenchConfig off = on;
    off.count_rays = false;
    off.stage_timing = false;
    BenchResult b = run_benchmark(scene, off);
    REQUIRE(b.conditions.size() == 2);
    for (const auto& c : b.conditions) {
        INFO(c.mode);
        CHECK(c.rays_per_second == 0.0);
        CHECK(stage_sum(c.stages) == 0.0);
        CHECK(c.run_ms.size() == 3);   // timed runs unaffected by the switches
    }
}

TEST_CASE("cpu_spp 0 skips the CPU quality checks and keeps only the bitwise check",
          "[bench_run]") {
    Scene scene = make_scene("cornell", 32, 18);
    BenchResult skip = run_benchmark(scene, make_cfg(2, 2, 0, 0));
    REQUIRE(skip.quality.size() == 1);
    CHECK(skip.quality[0].name == "megakernel_vs_wavefront_bitwise");
    CHECK(check(skip, "cpu_vs_megakernel") == nullptr);
    CHECK(check(skip, "cpu_vs_wavefront") == nullptr);
    CHECK(skip.warmup == 0);       // zero warm-up is legal and recorded
    // metamorphic control: with cpu_spp 2 the same scene gains exactly the two CPU checks
    BenchResult with = run_benchmark(scene, make_cfg(2, 2, 0, 2));
    CHECK(with.quality.size() == 3);
}

TEST_CASE("invalid configs throw std::invalid_argument and an unfinalized scene std::runtime_error",
          "[bench_run]") {
    Scene scene = make_scene("cornell", 16, 16);
    REQUIRE_NOTHROW(run_benchmark(scene, make_cfg(1, 1, 0, 0)));   // control: the valid base config
    {
        BenchConfig c = make_cfg(1, 0, 0, 0);   // reps 0 only
        REQUIRE_THROWS_AS(run_benchmark(scene, c), std::invalid_argument);
    }
    {
        BenchConfig c = make_cfg(1, 1, -1, 0);  // warmup -1 only
        REQUIRE_THROWS_AS(run_benchmark(scene, c), std::invalid_argument);
    }
    {
        BenchConfig c = make_cfg(0, 1, 0, 0);   // spp 0 only
        REQUIRE_THROWS_AS(run_benchmark(scene, c), std::invalid_argument);
    }
    {   // unfinalized: BVH removed (the single defect); runtime_error and not invalid_argument
        Scene bad = scene;
        bad.nodes.clear();
        bool runtime = false, invalid = false;
        try {
            (void)run_benchmark(bad, make_cfg(1, 1, 0, 0));
        } catch (const std::invalid_argument&) {
            invalid = true;
        } catch (const std::runtime_error&) {
            runtime = true;
        }
        CHECK(runtime);
        CHECK_FALSE(invalid);
    }
    // the device stays usable after the failures
    BenchResult ok = run_benchmark(scene, make_cfg(1, 1, 0, 0));
    CHECK(ok.conditions.size() == 2);
}

TEST_CASE("the stored verdict equals the fixed rule applied to the stored values",
          "[bench_run]") {
    // Given a real run (outcome may be supported or weakened; both are legitimate)
    Scene scene = make_scene("divergence", 64, 36);
    BenchResult r = run_benchmark(scene, make_cfg(4, 3, 1, 0));
    REQUIRE(r.conditions.size() == 2);
    // When apply_verdict is applied again to a copy Then nothing changes (idempotent, no hidden state)
    BenchResult again = r;
    apply_verdict(&again);
    CHECK(again.verdict == r.verdict);
    CHECK(again.reasons == r.reasons);
    CHECK(again.speedup == r.speedup);
    CHECK(again.speedup_meets_target == r.speedup_meets_target);
    CHECK(again.vram_within_budget == r.vram_within_budget);
    // And it equals an independent restatement of the rule of bench.hpp
    const double speedup = r.conditions[0].median / r.conditions[1].median;
    CHECK(r.speedup == speedup);
    bool vram_ok = true, valid = true, quality_ok = true;
    for (const auto& c : r.conditions) {
        vram_ok = vram_ok && (double)c.peak_vram_bytes <= 3.5e9;
        valid = valid && c.queue_overflows == 0 && c.invalid_hits == 0 && c.nonfinite_pixels == 0;
    }
    for (const auto& q : r.quality) quality_ok = quality_ok && q.pass;
    CHECK(r.vram_within_budget == vram_ok);
    const bool expect_supported = speedup >= 1.30 && quality_ok && vram_ok && valid;
    CHECK(r.verdict == (expect_supported ? "hypothesis_supported" : "hypothesis_weakened"));
    CHECK(r.reasons.empty() == expect_supported);
    // And the JSON of the real result is accepted by the validator
    CHECK(validate_bench_json(bench_to_json(r)).empty());
}

TEST_CASE("wfbench CLI end to end writes JSON that validates and exits 0",
          "[bench_run]") {
    // Given a small valid invocation Then exit 0 (whatever the verdict), a schema-valid file whose
    // recorded reps and image size equal the arguments.
    fs::path dir = fresh_dir("wfbench_ok");
    fs::path out = dir / "bench.json";
    Cli c = run_wfbench(dir, "--scene cornell --width 32 --height 18 --spp 2 --seed 7 --max-depth 4 "
                             "--reps 3 --warmup 1 --cpu-spp 2 --out \"" + out.string() + "\"");
    INFO("stderr: " << c.err);
    REQUIRE(c.code == 0);
    REQUIRE(fs::exists(out));
    const std::string text = read_text(out);
    CHECK(validate_bench_json(text).empty());
    Json j = json_parse(text);
    REQUIRE(j.at("conditions").arr.size() == 2);
    CHECK(j.at("conditions").arr[0].at("run_ms").arr.size() == 3);
    CHECK(j.at("conditions").arr[1].at("run_ms").arr.size() == 3);
    CHECK(j.at("reps").num == 3.0);
    CHECK(j.at("warmup").num == 1.0);
    CHECK(j.at("spp").num == 2.0);
    CHECK(j.at("seed").num == 7.0);
    CHECK(j.at("width").num == 32.0);
    CHECK(j.at("height").num == 18.0);
}

TEST_CASE("wfbench rejects invalid invocations with exit 2, a message naming the flag, and no file",
          "[bench_run]") {
    // Given one invalid element per invocation (the rest valid) Then exit 2, stderr names exactly the
    // offending flag (and not an unrelated valid flag, so a usage dump cannot satisfy it), no output.
    struct Bad { const char* label; std::string args; const char* flag; };
    fs::path dir = fresh_dir("wfbench_bad");
    fs::path out = dir / "never.json";
    const std::string o = " --out \"" + out.string() + "\"";
    const std::string base = "--width 16 --height 9 --spp 1 --max-depth 4 --warmup 0 ";
    std::vector<Bad> bad = {
        {"missing value for --reps", "--scene cornell " + base + o + " --reps", "--reps"},
        {"reps 0", "--scene cornell " + base + "--reps 0" + o, "--reps"},
        {"unknown scene", "--scene nosuchscene " + base + "--reps 1" + o, "--scene"},
        {"spp 0", "--scene cornell --width 16 --height 9 --max-depth 4 --warmup 0 --spp 0 --reps 1" + o, "--spp"},
        {"warmup -1", "--scene cornell --width 16 --height 9 --spp 1 --max-depth 4 --warmup -1 --reps 1" + o, "--warmup"},
    };
    for (const auto& b : bad) {
        INFO(b.label);
        Cli c = run_wfbench(dir, b.args);
        CHECK(c.code == 2);
        CHECK(c.err.find(b.flag) != std::string::npos);
        if (std::string(b.flag) != "--max-depth") CHECK(c.err.find("--max-depth") == std::string::npos);
        CHECK_FALSE(fs::exists(out));
    }
}

TEST_CASE("imgdiff exits 0 / 1 / 2 by tolerance and input validity and prints max_abs_rel and mean_rel",
          "[bench_run]") {
    fs::path dir = fresh_dir("imgdiff");
    const int W = 4, H = 3;
    std::vector<float> a(3 * W * H, 0.5f);
    std::vector<float> c = a;
    a[10] = 1.0f;
    c[10] = 1.01f;   // 1% in one channel of one pixel; every other channel identical
    std::vector<float> tiny = a;
    tiny[10] = 1.000001f;   // relative 9.5e-7
    const std::string pa = (dir / "a.pfm").string(), pb = (dir / "b.pfm").string(),
                      pc = (dir / "c.pfm").string(), pt = (dir / "t.pfm").string();
    write_pfm(pa, W, H, a);
    write_pfm(pb, W, H, a);
    write_pfm(pc, W, H, c);
    write_pfm(pt, W, H, tiny);
    auto parse = [](const std::string& s, double* mx, double* mean) {
        return std::sscanf(s.c_str(), "max_abs_rel=%lf mean_rel=%lf", mx, mean) == 2;
    };
    double mx = -1, mean = -1;
    {   // identical pair: exit 0 and zero difference
        Cli r = run_imgdiff(dir, "\"" + pa + "\" \"" + pb + "\"");
        REQUIRE(r.code == 0);
        REQUIRE(parse(r.out, &mx, &mean));
        CHECK(mx == 0.0);
        CHECK(mean == 0.0);
    }
    {   // 1% in one channel with --tol 1e-5: exit 1, max_abs_rel ~ 0.01 (0.01/1.01 or 0.01/1.0)
        Cli r = run_imgdiff(dir, "\"" + pa + "\" \"" + pc + "\" --tol 1e-5");
        CHECK(r.code == 1);
        REQUIRE(parse(r.out, &mx, &mean));
        CHECK(mx > 0.0098);
        CHECK(mx < 0.0102);
        // mean(a) = (35*0.5+1)/36 = 0.5139; |mean a - mean b| = 0.01/36 -> rel 5.4e-4 (denominator ~mean a)
        CHECK(mean > 5.2e-4);
        CHECK(mean < 5.6e-4);
        // the default tolerance is 1e-5: same verdict without --tol
        CHECK(run_imgdiff(dir, "\"" + pa + "\" \"" + pc + "\"").code == 1);
        // metamorphic: loosening the tolerance above the difference flips the exit code to 0
        CHECK(run_imgdiff(dir, "\"" + pa + "\" \"" + pc + "\" --tol 0.02").code == 0);
    }
    {   // 9.5e-7 passes the default 1e-5 but fails --tol 1e-7 (boundary on the tolerance)
        CHECK(run_imgdiff(dir, "\"" + pa + "\" \"" + pt + "\"").code == 0);
        CHECK(run_imgdiff(dir, "\"" + pa + "\" \"" + pt + "\" --tol 1e-7").code == 1);
    }
    {   // missing file
        Cli r = run_imgdiff(dir, "\"" + pa + "\" \"" + (dir / "absent.pfm").string() + "\"");
        CHECK(r.code == 2);
        CHECK_FALSE(r.err.empty());
    }
    {   // different dimensions (3x4 vs 4x3 hold the same number of floats)
        const std::string pd = (dir / "d.pfm").string();
        write_pfm(pd, H, W, a);
        Cli r = run_imgdiff(dir, "\"" + pa + "\" \"" + pd + "\"");
        CHECK(r.code == 2);
        CHECK_FALSE(r.err.empty());
    }
    {   // malformed file
        const std::string pm = (dir / "m.pfm").string();
        std::ofstream(pm, std::ios::binary) << "this is not a pfm";
        Cli r = run_imgdiff(dir, "\"" + pa + "\" \"" + pm + "\"");
        CHECK(r.code == 2);
        CHECK_FALSE(r.err.empty());
    }
}
