// benchmark statistics, verdict rule and JSON schema. Host only.
// Oracles: hand-computed linear-interpolation percentiles, the verdict rule of include/wf/bench.hpp
// restated as a truth table, and mutation of a parsed valid document (every rejection test first
// proves the unmutated, re-serialised document is accepted, so the mutation is the only cause).
// Assumption (header says "keys mirror the structs"): top-level keys schema_version, conditions,
// quality, speedup, verdict; condition keys mode, run_ms, min, median, max; quality key pass.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "wf/bench.hpp"
#include "wf/image_io.hpp"
#include "wf/json.hpp"

using namespace wf;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// A condition whose stats are set by hand (finalize_condition is NOT used, so verdict cases are
// independent of the statistics case). Image geometry 64x36 at spp 4.
ConditionStats make_cond(const std::string& mode, double median_ms, uint64_t vram = 1000000000ull) {
    ConditionStats c;
    c.mode = mode;
    c.run_ms = {median_ms, median_ms, median_ms};
    c.min = c.median = c.p10 = c.p90 = c.max = median_ms;
    c.spread_rel = 0.0;
    c.samples_per_second = 64.0 * 36.0 * 4.0 / (median_ms / 1000.0);
    c.rays_per_second = 1.0e7;
    c.peak_vram_bytes = vram;
    c.queue_high_water[0] = 5;
    return c;
}

// A result that satisfies every criterion when mega/wave >= 1.30; verdict already applied.
BenchResult make_result(double mega_ms = 13.0, double wave_ms = 10.0) {
    BenchResult r;
    r.scene_name = "divergence";
    r.scene_hash = 0x0123456789abcdefull;
    r.width = 64; r.height = 36; r.spp = 4; r.max_depth = 8; r.reps = 3; r.warmup = 1;
    r.seed = 1234;
    r.device_name = "Test GPU";
    r.driver_version = "555.1";
    r.cuda_runtime_version = "12.4";
    r.conditions = {make_cond("megakernel", mega_ms), make_cond("wavefront", wave_ms)};
    r.quality = {{"megakernel_vs_wavefront_bitwise", true, 0.0, kBitwiseRelTolerance, "ok"},
                 {"cpu_vs_megakernel", true, 0.999, kSePixelFraction, "ok"},
                 {"cpu_vs_wavefront", true, 0.999, kSePixelFraction, "ok"}};
    apply_verdict(&r);
    return r;
}

bool supported(const BenchResult& r) { return r.verdict == "hypothesis_supported"; }
bool weakened(const BenchResult& r) { return r.verdict == "hypothesis_weakened"; }

// ---- JSON tree helpers (the library only parses; this writes the tree back) ----
void dump(const Json& j, std::string& o) {
    switch (j.type) {
        case Json::Null: o += "null"; break;
        case Json::Bool: o += j.b ? "true" : "false"; break;
        case Json::Number: { char b[64]; std::snprintf(b, sizeof b, "%.17g", j.num); o += b; break; }
        case Json::String: o += "\"" + json_escape(j.str) + "\""; break;
        case Json::Array: {
            o += "[";
            for (size_t i = 0; i < j.arr.size(); ++i) { if (i) o += ","; dump(j.arr[i], o); }
            o += "]";
            break;
        }
        case Json::Object: {
            o += "{";
            bool first = true;
            for (const auto& kv : j.obj) {
                if (!first) o += ",";
                first = false;
                o += "\"" + json_escape(kv.first) + "\":";
                dump(kv.second, o);
            }
            o += "}";
            break;
        }
    }
}
std::string dumps(const Json& j) { std::string s; dump(j, s); return s; }

bool contains_string(const Json& j, const std::string& s) {
    if (j.type == Json::String && j.str == s) return true;
    for (const auto& e : j.arr) if (contains_string(e, s)) return true;
    for (const auto& kv : j.obj) if (contains_string(kv.second, s)) return true;
    return false;
}
bool contains_number(const Json& j, double v) {
    if (j.type == Json::Number && j.num == v) return true;
    for (const auto& e : j.arr) if (contains_number(e, v)) return true;
    for (const auto& kv : j.obj) if (contains_number(kv.second, v)) return true;
    return false;
}
bool contains_huge_number(const Json& j, double lo) {  // a 64-bit hash written as a double
    if (j.type == Json::Number && std::fabs(j.num) >= lo) return true;
    for (const auto& e : j.arr) if (contains_huge_number(e, lo)) return true;
    for (const auto& kv : j.obj) if (contains_huge_number(kv.second, lo)) return true;
    return false;
}
Json* find_string(Json& j, const std::string& s) {
    if (j.type == Json::String && j.str == s) return &j;
    for (auto& e : j.arr) if (Json* p = find_string(e, s)) return p;
    for (auto& kv : j.obj) if (Json* p = find_string(kv.second, s)) return p;
    return nullptr;
}

Json num(double v) { Json j; j.type = Json::Number; j.num = v; return j; }
Json str(const std::string& s) { Json j; j.type = Json::String; j.str = s; return j; }
Json boolean(bool b) { Json j; j.type = Json::Bool; j.b = b; return j; }

}  // namespace

// =================================================================================================
TEST_CASE("finalize_condition computes linear-interpolation statistics and rejects an empty set",
          "[bench_stats]") {
    // Given hand-computed run_ms sets (odd, even, single element, unsorted input)
    // When finalize_condition runs Then min/median/p10/p90/max/spread/throughput match the
    // hand values (index (n-1)*p, linear between order statistics) and run_ms stays in run order.
    {   // sorted 1 2 3 4 5: p10 at index 0.4 -> 1.4; p90 at 3.6 -> 4.6; median 3
        ConditionStats c;
        c.run_ms = {5, 1, 3, 2, 4};
        finalize_condition(&c, 8, 4, 2);
        CHECK(c.min == 1.0);
        CHECK(c.max == 5.0);
        CHECK_THAT(c.median, WithinAbs(3.0, 1e-12));
        CHECK_THAT(c.p10, WithinAbs(1.4, 1e-12));
        CHECK_THAT(c.p90, WithinAbs(4.6, 1e-12));
        CHECK_THAT(c.spread_rel, WithinAbs((4.6 - 1.4) / 3.0, 1e-12));
        // 8*4*2 = 64 samples over 3 ms
        CHECK_THAT(c.samples_per_second, WithinRel(64.0 / 0.003, 1e-12));
        CHECK(c.run_ms == std::vector<double>{5, 1, 3, 2, 4});
    }
    {   // sorted 10 20 30 40: median at index 1.5 -> 25 (not 20 and not 30); p10 0.3 -> 13; p90 2.7 -> 37
        ConditionStats c;
        c.run_ms = {40, 10, 30, 20};
        finalize_condition(&c, 10, 10, 1);
        CHECK_THAT(c.median, WithinAbs(25.0, 1e-12));
        CHECK_THAT(c.p10, WithinAbs(13.0, 1e-12));
        CHECK_THAT(c.p90, WithinAbs(37.0, 1e-12));
        CHECK_THAT(c.spread_rel, WithinAbs(24.0 / 25.0, 1e-12));
        CHECK_THAT(c.samples_per_second, WithinRel(100.0 / 0.025, 1e-12));  // 4000
    }
    {   // one element: every statistic equals it, spread 0
        ConditionStats c;
        c.run_ms = {7.0};
        finalize_condition(&c, 1, 1, 1);
        CHECK(c.min == 7.0); CHECK(c.median == 7.0); CHECK(c.p10 == 7.0);
        CHECK(c.p90 == 7.0); CHECK(c.max == 7.0);
        CHECK(c.spread_rel == 0.0);
        CHECK_THAT(c.samples_per_second, WithinRel(1.0 / 0.007, 1e-12));
    }
    {   // empty set: rejected and nothing written
        ConditionStats e;
        REQUIRE_THROWS_AS(finalize_condition(&e, 8, 4, 2), std::invalid_argument);
        CHECK(e.run_ms.empty());
        CHECK(e.median == 0.0);
        CHECK(e.samples_per_second == 0.0);
    }
}

TEST_CASE("apply_verdict speedup rule: 1.30 inclusive, 1.29 weakened, missing mode gives zero",
          "[bench_stats]") {
    // Given a result where only the speedup varies (quality, vram, validity counters all pass)
    // Then speedup = median(megakernel)/median(wavefront), the target is inclusive, and a missing
    // condition yields speedup 0 and a weakened verdict. Stale verdict state is overwritten.
    {   // 13/10 = 1.30 exactly (both correctly rounded divisions of the same real)
        BenchResult r = make_result(13.0, 10.0);
        CHECK_THAT(r.speedup, WithinAbs(1.30, 1e-12));
        CHECK(r.speedup_meets_target);
        CHECK(r.vram_within_budget);
        CHECK(supported(r));
        CHECK(r.reasons.empty());
    }
    {   // 12.9/10 = 1.29
        BenchResult r = make_result(12.9, 10.0);
        CHECK_THAT(r.speedup, WithinAbs(1.29, 1e-12));
        CHECK_FALSE(r.speedup_meets_target);
        CHECK(r.vram_within_budget);  // isolation: only the speedup criterion is violated
        CHECK(weakened(r));
        CHECK(r.reasons.size() == 1);
    }
    {   // direction: wavefront SLOWER (10/13 = 0.769) is weakened; wavefront twice as fast gives 2
        BenchResult slow = make_result(10.0, 13.0);
        CHECK_THAT(slow.speedup, WithinAbs(10.0 / 13.0, 1e-12));
        CHECK(weakened(slow));
        BenchResult fast = make_result(20.0, 10.0);
        CHECK_THAT(fast.speedup, WithinAbs(2.0, 1e-12));
        CHECK(supported(fast));
    }
    {   // missing wavefront / missing megakernel / no conditions: speedup exactly 0, weakened
        for (int keep = 0; keep < 3; ++keep) {
            BenchResult r = make_result(13.0, 10.0);
            if (keep == 0) r.conditions.erase(r.conditions.begin() + 1);   // megakernel only
            if (keep == 1) r.conditions.erase(r.conditions.begin());       // wavefront only
            if (keep == 2) r.conditions.clear();
            apply_verdict(&r);
            INFO("variant " << keep);
            CHECK(r.speedup == 0.0);
            CHECK_FALSE(r.speedup_meets_target);
            CHECK(weakened(r));
            CHECK_FALSE(r.reasons.empty());
        }
    }
    {   // stale state: a weakened result whose cause is fixed becomes supported with no leftover reason
        BenchResult r = make_result(12.0, 10.0);
        REQUIRE(weakened(r));
        r.conditions[0].median = 14.0;
        apply_verdict(&r);
        CHECK(supported(r));
        CHECK(r.reasons.empty());
        CHECK(r.speedup_meets_target);
    }
}

TEST_CASE("apply_verdict quality and vram rules: one failed check, 3.5e9 inclusive, every condition",
          "[bench_stats]") {
    // Given a speedup-passing result, flipping exactly one quality check to failed (each of three)
    // weakens with one reason; vram 3.5e9 passes, 3.5e9+1 fails, in either condition alone.
    for (size_t k = 0; k < 3; ++k) {
        BenchResult r = make_result();
        REQUIRE(supported(r));
        r.quality[k].pass = false;
        apply_verdict(&r);
        INFO("failed check " << r.quality[k].name);
        CHECK(weakened(r));
        CHECK(r.reasons.size() == 1);
        CHECK(r.speedup_meets_target);  // isolation
        CHECK(r.vram_within_budget);
    }
    for (size_t cond = 0; cond < 2; ++cond) {
        INFO("condition index " << cond);
        BenchResult at = make_result();
        at.conditions[cond].peak_vram_bytes = 3500000000ull;  // exactly the budget: passes
        apply_verdict(&at);
        CHECK(at.vram_within_budget);
        CHECK(supported(at));

        BenchResult over = make_result();
        over.conditions[cond].peak_vram_bytes = 3500000001ull;  // one byte over in ONE condition
        apply_verdict(&over);
        CHECK_FALSE(over.vram_within_budget);
        CHECK(weakened(over));
        CHECK(over.reasons.size() == 1);
        CHECK(over.speedup_meets_target);
    }
    {   // both at the budget: still within (the rule is per condition, not a sum)
        BenchResult both = make_result();
        both.conditions[0].peak_vram_bytes = 3500000000ull;
        both.conditions[1].peak_vram_bytes = 3500000000ull;
        apply_verdict(&both);
        CHECK(both.vram_within_budget);
        CHECK(supported(both));
    }
}

TEST_CASE("apply_verdict validity counters: overflow, invalid hit or non-finite pixel in either condition",
          "[bench_stats]") {
    // Given a passing result, any positive counter in any condition weakens with exactly one reason
    // while speedup, vram and quality flags stay true (so only the counter can be the cause).
    using Setter = std::function<void(ConditionStats&, long long)>;
    std::vector<std::pair<std::string, Setter>> counters = {
        {"queue_overflows", [](ConditionStats& c, long long v) { c.queue_overflows = v; }},
        {"invalid_hits", [](ConditionStats& c, long long v) { c.invalid_hits = v; }},
        {"nonfinite_pixels", [](ConditionStats& c, long long v) { c.nonfinite_pixels = v; }},
    };
    for (const auto& [name, set] : counters) {
        for (size_t cond = 0; cond < 2; ++cond) {
            for (long long v : {1ll, 1000000ll}) {
                BenchResult r = make_result();
                REQUIRE(supported(r));
                set(r.conditions[cond], v);
                apply_verdict(&r);
                INFO(name << " = " << v << " in condition " << cond);
                CHECK(weakened(r));
                CHECK(r.reasons.size() == 1);
                CHECK(r.speedup_meets_target);
                CHECK(r.vram_within_budget);
            }
        }
    }
    BenchResult zero = make_result();  // control: all counters 0
    CHECK(supported(zero));
}

TEST_CASE("apply_verdict lists one reason per violated criterion and none when supported",
          "[bench_stats]") {
    // Six independent violations: speedup, one quality check, vram, overflow, invalid hit,
    // non-finite pixel. Given the full set Then six distinct non-empty reasons; metamorphic:
    // removing any one violation removes exactly one reason.
    auto build = [](unsigned mask) {
        BenchResult r = make_result(mask & 1 ? 12.0 : 13.0, 10.0);
        if (mask & 2) r.quality[1].pass = false;
        if (mask & 4) r.conditions[0].peak_vram_bytes = 4000000000ull;
        if (mask & 8) r.conditions[1].queue_overflows = 3;
        if (mask & 16) r.conditions[0].invalid_hits = 2;
        if (mask & 32) r.conditions[1].nonfinite_pixels = 1;
        apply_verdict(&r);
        return r;
    };
    BenchResult all = build(63);
    CHECK(weakened(all));
    REQUIRE(all.reasons.size() == 6);
    std::set<std::string> distinct(all.reasons.begin(), all.reasons.end());
    CHECK(distinct.size() == 6);
    for (const auto& s : all.reasons) CHECK_FALSE(s.empty());
    for (unsigned bit = 0; bit < 6; ++bit) {
        BenchResult r = build(63u & ~(1u << bit));
        INFO("violation " << bit << " removed");
        CHECK(r.reasons.size() == 5);
        CHECK(weakened(r));
    }
    BenchResult none = build(0);
    CHECK(supported(none));
    CHECK(none.reasons.empty());
}

TEST_CASE("pre-registered constants equal the documented values and the speedup test is >=",
          "[bench_stats]") {
    // Given DESIGN s8/s9: 1.30, 3.5e9, 0.995, 0.005, 1e-5, schema 1.
    CHECK(kSpeedupTarget == 1.30);
    CHECK(kVramBudgetBytes == 3.5e9);
    CHECK(kSePixelFraction == 0.995);
    CHECK(kMeanRelTolerance == 0.005);
    CHECK(kBitwiseRelTolerance == 1e-5);
    CHECK(kBenchSchemaVersion == 1);
    // Boundary at the ulp: wavefront 1 ms, megakernel = target exactly / one ulp below / one ulp above.
    BenchResult at = make_result(kSpeedupTarget, 1.0);
    CHECK(at.speedup == kSpeedupTarget);
    CHECK(supported(at));
    BenchResult below = make_result(std::nextafter(kSpeedupTarget, 0.0), 1.0);
    CHECK(weakened(below));
    BenchResult above = make_result(std::nextafter(kSpeedupTarget, 2.0), 1.0);
    CHECK(supported(above));
}

TEST_CASE("bench_to_json round trips through validate_bench_json and keeps 64-bit values exact",
          "[bench_stats]") {
    // Given a populated result with a scene hash above 2^53 (not representable as a double),
    // leading-zero hash digits, the maximum 32-bit seed, and a device name needing escapes
    // When serialised, validated and parsed Then the document is valid, the hash is a 16-digit hex
    // string identical to hex64(hash), no number near 2^64 exists, and the seed is exact.
    BenchResult r = make_result();
    r.scene_hash = 0xFEDCBA9876543211ull;
    r.seed = 4294967295u;
    r.device_name = "GTX \"1650\"\nTi";
    const std::string text = bench_to_json(r);
    CHECK(validate_bench_json(text).empty());
    Json j = json_parse(text);
    CHECK(j.at("schema_version").num == 1.0);
    CHECK(j.at("verdict").str == r.verdict);
    CHECK_THAT(j.at("speedup").num, WithinAbs(1.30, 1e-12));
    CHECK(text.find("fedcba9876543211") != std::string::npos);
    CHECK(contains_string(j, "fedcba9876543211"));
    CHECK_FALSE(contains_huge_number(j, 1e17));
    CHECK(contains_number(j, 4294967295.0));
    CHECK(contains_string(j, r.device_name));
    REQUIRE(j.at("conditions").arr.size() == 2);
    CHECK(j.at("conditions").arr[0].at("mode").str == "megakernel");
    CHECK(j.at("conditions").arr[1].at("mode").str == "wavefront");
    REQUIRE(j.at("conditions").arr[0].at("run_ms").arr.size() == 3);
    CHECK(j.at("conditions").arr[0].at("run_ms").arr[0].num == 13.0);

    BenchResult z = make_result();  // leading zeros survive: 16 digits, not 13
    z.scene_hash = 0xabcull;
    CHECK(validate_bench_json(bench_to_json(z)).empty());
    CHECK(contains_string(json_parse(bench_to_json(z)), "0000000000000abc"));
    CHECK(hex64(0xabcull) == "0000000000000abc");  // independent: the helper the library should use

    BenchResult weak = make_result(12.0, 10.0);  // a weakened result with reasons also round trips
    REQUIRE(weakened(weak));
    CHECK(validate_bench_json(bench_to_json(weak)).empty());
}

TEST_CASE("validate_bench_json rejects each structural and semantic corruption",
          "[bench_stats]") {
    // Given a valid document re-serialised from its parsed tree (control must be accepted)
    // When exactly one thing is corrupted Then a non-empty problem is returned.
    const std::string valid_text = bench_to_json(make_result());
    const Json base = json_parse(valid_text);
    REQUIRE(validate_bench_json(valid_text).empty());
    REQUIRE(validate_bench_json(dumps(base)).empty());  // control: the serialiser is not the cause
    const std::string weak_text = bench_to_json(make_result(12.0, 10.0));
    const Json weak_base = json_parse(weak_text);
    REQUIRE(validate_bench_json(dumps(weak_base)).empty());  // a consistent weakened doc is valid

    auto rejects = [](const std::string& label, const std::string& text) {
        INFO(label);
        std::string problem = validate_bench_json(text);
        CHECK_FALSE(problem.empty());
    };
    auto mutated = [&](const std::string& label, const Json& from, const std::function<void(Json&)>& f) {
        Json j = from;
        f(j);
        rejects(label, dumps(j));
    };

    // invalid JSON
    rejects("empty text", "");
    rejects("truncated", "{\"schema_version\":1");
    rejects("trailing garbage", valid_text + " x");
    rejects("not an object", "[]");

    // missing key at top level, in a condition and in a quality check (every key present)
    for (const auto& kv : base.obj)
        mutated("remove top-level key " + kv.first, base, [&](Json& j) { j.obj.erase(kv.first); });
    for (const auto& kv : base.at("conditions").arr[0].obj)
        mutated("remove condition key " + kv.first, base,
                [&](Json& j) { j.obj["conditions"].arr[0].obj.erase(kv.first); });
    for (const auto& kv : base.at("quality").arr[0].obj)
        mutated("remove quality key " + kv.first, base,
                [&](Json& j) { j.obj["quality"].arr[0].obj.erase(kv.first); });

    // wrong type
    mutated("speedup as string", base, [](Json& j) { j.obj["speedup"] = str("1.3"); });
    mutated("verdict as number", base, [](Json& j) { j.obj["verdict"] = num(1); });
    mutated("conditions as object", base, [](Json& j) { j.obj["conditions"] = Json{}; j.obj["conditions"].type = Json::Object; });
    mutated("run_ms element as string", base, [](Json& j) { j.obj["conditions"].arr[0].obj["run_ms"].arr[0] = str("13"); });
    mutated("quality pass as number", base, [](Json& j) { j.obj["quality"].arr[0].obj["pass"] = num(1); });

    // schema version
    mutated("schema_version 2", base, [](Json& j) { j.obj["schema_version"] = num(2); });
    mutated("schema_version 0", base, [](Json& j) { j.obj["schema_version"] = num(0); });

    // run_ms empty, condition order, mode name
    mutated("empty run_ms", base, [](Json& j) { j.obj["conditions"].arr[0].obj["run_ms"].arr.clear(); });
    mutated("wavefront before megakernel", base, [](Json& j) { std::swap(j.obj["conditions"].arr[0], j.obj["conditions"].arr[1]); });
    mutated("two megakernel conditions", base, [](Json& j) { j.obj["conditions"].arr[1].obj["mode"] = str("megakernel"); });

    // verdict inconsistent with the parsed values
    mutated("supported verdict with speedup 1.2", weak_base, [](Json& j) { j.obj["verdict"] = str("hypothesis_supported"); });
    mutated("weakened verdict though every criterion passes", base, [](Json& j) { j.obj["verdict"] = str("hypothesis_weakened"); });
    mutated("supported verdict with a failed quality check", base, [](Json& j) { j.obj["quality"].arr[1].obj["pass"] = boolean(false); });
    mutated("supported verdict with an overflow", base, [](Json& j) {
        j.obj["conditions"].arr[1].obj["queue_overflows"] = num(4);
    });
    mutated("unknown verdict string", base, [](Json& j) { j.obj["verdict"] = str("maybe"); });

    // median outside min..max
    mutated("min above median", base, [](Json& j) { j.obj["conditions"].arr[0].obj["min"] = num(14); });
    mutated("median above max", base, [](Json& j) { j.obj["conditions"].arr[0].obj["median"] = num(20); });

    // hashes are exactly 16 hex digits
    const std::string hash = hex64(make_result().scene_hash);
    mutated("hash 15 digits", base, [&](Json& j) { Json* p = find_string(j, hash); REQUIRE(p); p->str = hash.substr(1); });
    mutated("hash 17 digits", base, [&](Json& j) { Json* p = find_string(j, hash); REQUIRE(p); p->str = "0" + hash; });
    mutated("hash non-hex digit", base, [&](Json& j) { Json* p = find_string(j, hash); REQUIRE(p); p->str[5] = 'g'; });
    mutated("hash as number", base, [&](Json& j) {
        Json* p = find_string(j, hash);
        REQUIRE(p);
        *p = num(8.1985529216486896e16);
    });
}
