// Tests for wf::json_parse / json_escape (json.hpp) and percentile / report_to_json /
// validate_report_json (report.hpp).
// Oracles: the JSON grammar (RFC 8259), hand-computed order statistics, the schema text in report.hpp.
// Mutation tests build a valid document, prove it validates, then change exactly one thing, so the
// rejection can only be caused by that change.
#include "test_framework.hpp"
#include "wf/json.hpp"
#include "wf/report.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using wf::Json;

namespace {

bool parse_throws_runtime(const std::string& text) {
    try {
        (void)wf::json_parse(text);
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// Independent minimal serialiser for a Json tree (used to build mutated documents).
std::string dump_str(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else o += c;
    }
    return o + "\"";
}
std::string dump(const Json& j) {
    switch (j.type) {
        case Json::Null: return "null";
        case Json::Bool: return j.b ? "true" : "false";
        case Json::Number: { char buf[64]; std::snprintf(buf, sizeof buf, "%.17g", j.num); return buf; }
        case Json::String: return dump_str(j.str);
        case Json::Array: {
            std::string o = "[";
            for (size_t i = 0; i < j.arr.size(); ++i) o += (i ? "," : "") + dump(j.arr[i]);
            return o + "]";
        }
        case Json::Object: {
            std::string o = "{";
            bool first = true;
            for (auto& kv : j.obj) { o += (first ? "" : ",") + dump_str(kv.first) + ":" + dump(kv.second); first = false; }
            return o + "}";
        }
    }
    return "";
}

wf::Report full_report() {
    wf::Report r;
    r.git_revision = "abc123def";
    r.scene_name = "divergence";
    r.scene_hash = 0x0123456789abcdefull;  // leading zero: must be padded to 16 digits
    r.width = 320; r.height = 200; r.spp = 16; r.max_depth = 6; r.seed = 4242;
    r.mode = "wavefront";
    r.device.name = "NVIDIA GeForce GTX 1650 Ti";
    r.device.driver_version = "555.01";
    r.device.cuda_runtime_version = "12.4";
    r.device.cuda_driver_version = "12.5";
    r.device.compute_capability = "7.5";
    r.device.total_memory_bytes = 4ull << 30;
    r.run_ms = {12.0, 30.0, 10.0, 13.0, 11.0};  // sorted: 10 11 12 13 30
    r.samples_per_second = 1.5e6;
    r.rays_per_second = 2.5e7;
    r.peak_vram_bytes = 1500000000ull;
    r.queue_high_water[0] = 11; r.queue_high_water[1] = 22; r.queue_high_water[2] = 33;
    r.queue_high_water[3] = 44; r.queue_high_water[4] = 55;
    r.image_png = "out/img.png";
    r.image_pfm = "out/img.pfm";
    r.pfm_hash = 0xffull;  // must become 00000000000000ff
    r.nonfinite_pixels = 0; r.queue_overflows = 0; r.invalid_hits = 0;
    return r;
}

wf::Report cpu_report() {
    wf::Report r;  // defaults: device "cpu", rays 0, peak vram 0, git_revision empty
    r.scene_name = "cornell";
    r.scene_hash = 0xabcdefull;
    r.width = 64; r.height = 48; r.spp = 4; r.max_depth = 8; r.seed = 1;
    r.mode = "cpu";
    r.run_ms = {5.0, 7.0};
    r.samples_per_second = 1000.0;
    r.rays_per_second = 0.0;
    return r;
}

// Navigate to a node by key path; null if absent.
Json* node(Json& root, std::initializer_list<const char*> path) {
    Json* cur = &root;
    for (const char* k : path) {
        if (cur->type != Json::Object) return nullptr;
        auto it = cur->obj.find(k);
        if (it == cur->obj.end()) return nullptr;
        cur = &it->second;
    }
    return cur;
}

Json parsed_full() { return wf::json_parse(wf::report_to_json(full_report())); }

bool is_hex16_lower(const std::string& s) {
    if (s.size() != 16) return false;
    for (char c : s) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

struct Leaf { std::initializer_list<const char*> path; };

}  // namespace

// ================================================================ json_parse

TEST(json_parse_reads_objects_arrays_and_nesting) {
    Json j = wf::json_parse(" { \"a\" : [1, 2 ,3], \"b\": {\"c\": true}, \"d\": [] , \"e\": {} } ");
    CHECK(j.type == Json::Object);
    CHECK(j.has("a") && j.has("b") && j.has("d") && j.has("e"));
    CHECK(!j.has("zzz"));
    CHECK(j.at("a").type == Json::Array);
    CHECK(j.at("a").arr.size() == 3);
    CHECK_NEAR(j.at("a").arr[2].num, 3.0, 0.0);
    CHECK(j.at("b").at("c").type == Json::Bool && j.at("b").at("c").b == true);
    CHECK(j.at("d").type == Json::Array && j.at("d").arr.empty());
    CHECK(j.at("e").type == Json::Object && j.at("e").obj.empty());
}

TEST(json_parse_numbers_integer_negative_fraction_exponent) {
    // Oracle: decimal value of each literal.
    CHECK_NEAR(wf::json_parse("0").num, 0.0, 0.0);
    CHECK_NEAR(wf::json_parse("42").num, 42.0, 0.0);
    CHECK_NEAR(wf::json_parse("-17").num, -17.0, 0.0);
    CHECK_NEAR(wf::json_parse("0.5").num, 0.5, 0.0);
    CHECK_NEAR(wf::json_parse("-0.25").num, -0.25, 0.0);
    CHECK_NEAR(wf::json_parse("1e3").num, 1000.0, 0.0);
    CHECK_NEAR(wf::json_parse("1E+2").num, 100.0, 0.0);
    CHECK_NEAR(wf::json_parse("-2.5e-2").num, -0.025, 1e-15);
    CHECK_NEAR(wf::json_parse("12345678901").num, 12345678901.0, 0.0);
    CHECK(wf::json_parse("7").type == Json::Number);
}

TEST(json_parse_strings_with_escapes_and_unicode_escapes) {
    CHECK(wf::json_parse("\"hello\"").str == "hello");
    CHECK(wf::json_parse("\"\"").str.empty());
    CHECK(wf::json_parse("\"a\\nb\"").str == "a\nb");
    CHECK(wf::json_parse("\"t\\tq\\\"b\\\\s\\/\"").str == "t\tq\"b\\s/");
    CHECK(wf::json_parse("\"\\r\\b\\f\"").str == "\r\b\f");
    CHECK(wf::json_parse("\"\\u0041\\u0062\"").str == "Ab");
    CHECK(wf::json_parse("\"\\u0001\"").str == std::string(1, '\x01'));
    // U+00E9 as UTF-8 is C3 A9 (json.hpp does not state the encoding; UTF-8 is the only sensible one).
    CHECK(wf::json_parse("\"\\u00e9\"").str == "\xC3\xA9");
    CHECK(wf::json_parse("\"\\u00E9\"").str == "\xC3\xA9");  // upper-case hex digits
}

TEST(json_parse_literals_true_false_null) {
    Json t = wf::json_parse("true");
    Json f = wf::json_parse("false");
    Json n = wf::json_parse("null");
    CHECK(t.type == Json::Bool && t.b == true);
    CHECK(f.type == Json::Bool && f.b == false);
    CHECK(n.type == Json::Null);
}

TEST(json_parse_rejects_malformed_input_with_runtime_error) {
    CHECK(parse_throws_runtime("[1,2,]"));            // trailing comma, array
    CHECK(parse_throws_runtime("{\"a\":1,}"));        // trailing comma, object
    CHECK(parse_throws_runtime("[1,2"));              // unbalanced bracket
    CHECK(parse_throws_runtime("{\"a\":1"));          // unbalanced brace
    CHECK(parse_throws_runtime("[1]]"));              // extra closer = trailing garbage
    CHECK(parse_throws_runtime("{} {}"));             // trailing garbage
    CHECK(parse_throws_runtime("1 2"));               // trailing garbage after a scalar
    CHECK(parse_throws_runtime("hello"));             // bare word
    CHECK(parse_throws_runtime("tru"));               // truncated literal
    CHECK(parse_throws_runtime("\"abc"));             // unterminated string
    CHECK(parse_throws_runtime("{\"a\" 1}"));         // missing colon
    CHECK(parse_throws_runtime("[1 2]"));             // missing comma
    CHECK(parse_throws_runtime("{a:1}"));             // unquoted key
    CHECK(parse_throws_runtime(""));                  // empty document
    CHECK(parse_throws_runtime("   "));               // whitespace only
}

TEST(json_parse_accepts_valid_text_that_looks_like_the_rejected_cases) {
    // Positive counterparts: the same shapes without the defect parse fine, so the rejections above
    // are caused by the defect and not by, e.g., any array being refused.
    CHECK(!parse_throws_runtime("[1,2]"));
    CHECK(!parse_throws_runtime("{\"a\":1}"));
    CHECK(!parse_throws_runtime("[1]"));
    CHECK(!parse_throws_runtime("\"abc\""));
    CHECK(!parse_throws_runtime("true"));
    CHECK(!parse_throws_runtime("\n\t [ ] \r\n"));
}

// ================================================================ json_escape

TEST(json_escape_quote_backslash_newline_and_control_char) {
    // Oracle: RFC 8259 section 7 short escapes and \u00XX for other control characters.
    // The header does not say whether the result carries surrounding quotes; the contract tested is
    // "escape the content" (no quotes added).
    CHECK(wf::json_escape("a\"b") == "a\\\"b");
    CHECK(wf::json_escape("a\\b") == "a\\\\b");
    CHECK(wf::json_escape("a\nb") == "a\\nb");
    CHECK(wf::json_escape(std::string("a\x01" "b")) == "a\\u0001b");
    CHECK(wf::json_escape("plain text 123") == "plain text 123");
    CHECK(wf::json_escape("").empty());
}

TEST(json_escape_output_parses_back_to_the_original_string) {
    // Inverse property against json_parse for every control character and the special characters.
    for (int c = 1; c < 32; ++c) {
        std::string s = std::string("x") + static_cast<char>(c) + "y";
        Json j = wf::json_parse("\"" + wf::json_escape(s) + "\"");
        if (j.str != s) {
            std::printf("  FAIL escape round trip for control char %d\n", c);
            ++wftest::failures();
        }
    }
    std::string nasty = "q\"uo\\te\n\r\t/ end";
    CHECK(wf::json_parse("\"" + wf::json_escape(nasty) + "\"").str == nasty);
}

// ================================================================ percentile

TEST(percentile_of_five_values_hits_exact_order_statistics) {
    // Linear interpolation on rank p*(n-1): n=5 -> ranks 0, 1, 2, 3, 4 at p = 0, .25, .5, .75, 1.
    std::vector<double> v{1, 2, 3, 4, 5};
    CHECK_NEAR(wf::percentile(v, 0.0), 1.0, 1e-12);
    CHECK_NEAR(wf::percentile(v, 0.25), 2.0, 1e-12);
    CHECK_NEAR(wf::percentile(v, 0.5), 3.0, 1e-12);
    CHECK_NEAR(wf::percentile(v, 0.75), 4.0, 1e-12);
    CHECK_NEAR(wf::percentile(v, 1.0), 5.0, 1e-12);
}

TEST(percentile_interpolates_linearly_between_two_values) {
    // rank = p*(2-1) = p; value = 10 + 10*p.
    std::vector<double> v{10, 20};
    CHECK_NEAR(wf::percentile(v, 0.5), 15.0, 1e-12);
    CHECK_NEAR(wf::percentile(v, 0.25), 12.5, 1e-12);
    CHECK_NEAR(wf::percentile(v, 0.1), 11.0, 1e-12);
    CHECK_NEAR(wf::percentile(v, 0.9), 19.0, 1e-12);
}

TEST(percentile_of_unsorted_input_equals_percentile_of_sorted_input) {
    // Metamorphic: permutation invariance. Sorted {10,11,12,13,30}: p10 rank .4 -> 10.4, p90 rank 3.6 -> 13+.6*17 = 23.2.
    std::vector<double> unsorted{12, 30, 10, 13, 11};
    CHECK_NEAR(wf::percentile(unsorted, 0.5), 12.0, 1e-12);
    CHECK_NEAR(wf::percentile(unsorted, 0.1), 10.4, 1e-12);
    CHECK_NEAR(wf::percentile(unsorted, 0.9), 23.2, 1e-12);
    CHECK_NEAR(wf::percentile(unsorted, 0.0), 10.0, 1e-12);
    CHECK_NEAR(wf::percentile(unsorted, 1.0), 30.0, 1e-12);
}

TEST(percentile_of_single_value_is_that_value) {
    CHECK_NEAR(wf::percentile({7.5}, 0.0), 7.5, 0.0);
    CHECK_NEAR(wf::percentile({7.5}, 0.5), 7.5, 0.0);
    CHECK_NEAR(wf::percentile({7.5}, 1.0), 7.5, 0.0);
}

TEST(percentile_of_empty_input_throws) {
    CHECK_THROWS(wf::percentile(std::vector<double>{}, 0.5));
}

// ================================================================ report_to_json

TEST(full_report_serialises_to_json_that_validates) {
    std::string text = wf::report_to_json(full_report());
    CHECK(wf::validate_report_json(text).empty());
}

TEST(cpu_report_with_zero_rays_and_empty_git_revision_validates_and_writes_nulls) {
    // Header: empty git_revision -> null; rays_per_second 0 -> null.
    std::string text = wf::report_to_json(cpu_report());
    std::string problem = wf::validate_report_json(text);
    CHECK(problem.empty());
    if (!problem.empty()) std::printf("  validator said: %s\n", problem.c_str());
    Json j = wf::json_parse(text);
    CHECK(j.at("git_revision").type == Json::Null);
    CHECK(j.at("throughput").at("rays_per_second").type == Json::Null);
    CHECK_NEAR(j.at("throughput").at("samples_per_second").num, 1000.0, 1e-9);
    // Positive counterpart: the populated report writes real values in the same places.
    Json f = parsed_full();
    CHECK(f.at("git_revision").type == Json::String && f.at("git_revision").str == "abc123def");
    CHECK(f.at("throughput").at("rays_per_second").type == Json::Number);
}

TEST(report_json_contains_every_schema_key_with_the_right_type) {
    Json j = parsed_full();
    CHECK(j.type == Json::Object);
    CHECK(node(j, {"schema_version"}) && node(j, {"schema_version"})->type == Json::Number);
    CHECK(node(j, {"git_revision"}) && node(j, {"git_revision"})->type == Json::String);
    CHECK(node(j, {"scene", "name"}) && node(j, {"scene", "name"})->type == Json::String);
    CHECK(node(j, {"scene", "hash"}) && node(j, {"scene", "hash"})->type == Json::String);
    const char* cfg_num[] = {"width", "height", "spp", "seed", "max_depth"};
    for (const char* k : cfg_num) {
        Json* n = node(j, {"config", k});
        if (!n || n->type != Json::Number) { std::printf("  FAIL config.%s missing or not a number\n", k); ++wftest::failures(); }
    }
    CHECK(node(j, {"config", "mode"}) && node(j, {"config", "mode"})->type == Json::String);
    const char* dev_str[] = {"name", "driver_version", "cuda_runtime_version", "cuda_driver_version", "compute_capability"};
    for (const char* k : dev_str) {
        Json* n = node(j, {"device", k});
        if (!n || n->type != Json::String) { std::printf("  FAIL device.%s missing or not a string\n", k); ++wftest::failures(); }
    }
    CHECK(node(j, {"device", "total_memory_bytes"}) && node(j, {"device", "total_memory_bytes"})->type == Json::Number);
    CHECK(node(j, {"timings_ms", "runs"}) && node(j, {"timings_ms", "runs"})->type == Json::Array);
    const char* tim[] = {"min", "median", "p10", "p90", "max"};
    for (const char* k : tim) {
        Json* n = node(j, {"timings_ms", k});
        if (!n || n->type != Json::Number) { std::printf("  FAIL timings_ms.%s missing or not a number\n", k); ++wftest::failures(); }
    }
    CHECK(node(j, {"throughput", "samples_per_second"}) && node(j, {"throughput", "samples_per_second"})->type == Json::Number);
    CHECK(node(j, {"throughput", "rays_per_second"}) && node(j, {"throughput", "rays_per_second"})->type == Json::Number);
    CHECK(node(j, {"peak_vram_bytes"}) && node(j, {"peak_vram_bytes"})->type == Json::Number);
    const char* q[] = {"lambert", "metal", "dielectric", "shadow", "active"};
    for (const char* k : q) {
        Json* n = node(j, {"queue_high_water", k});
        if (!n || n->type != Json::Number) { std::printf("  FAIL queue_high_water.%s missing or not a number\n", k); ++wftest::failures(); }
    }
    CHECK(node(j, {"outputs", "png"}) && node(j, {"outputs", "png"})->type == Json::String);
    CHECK(node(j, {"outputs", "pfm"}) && node(j, {"outputs", "pfm"})->type == Json::String);
    CHECK(node(j, {"outputs", "pfm_hash"}) && node(j, {"outputs", "pfm_hash"})->type == Json::String);
    const char* val[] = {"nonfinite_pixels", "queue_overflows", "invalid_hits"};
    for (const char* k : val) {
        Json* n = node(j, {"validity", k});
        if (!n || n->type != Json::Number) { std::printf("  FAIL validity.%s missing or not a number\n", k); ++wftest::failures(); }
    }
}

TEST(report_json_values_equal_the_report_fields_they_came_from) {
    Json j = parsed_full();
    CHECK_NEAR(j.at("schema_version").num, 1.0, 0.0);
    CHECK(wf::kReportSchemaVersion == 1);
    CHECK(j.at("scene").at("name").str == "divergence");
    CHECK_NEAR(j.at("config").at("width").num, 320, 0);
    CHECK_NEAR(j.at("config").at("height").num, 200, 0);
    CHECK_NEAR(j.at("config").at("spp").num, 16, 0);
    CHECK_NEAR(j.at("config").at("seed").num, 4242, 0);
    CHECK_NEAR(j.at("config").at("max_depth").num, 6, 0);
    CHECK(j.at("config").at("mode").str == "wavefront");
    CHECK(j.at("device").at("name").str == "NVIDIA GeForce GTX 1650 Ti");
    CHECK(j.at("device").at("compute_capability").str == "7.5");
    CHECK(j.at("device").at("driver_version").str == "555.01");
    CHECK(j.at("device").at("cuda_runtime_version").str == "12.4");
    CHECK(j.at("device").at("cuda_driver_version").str == "12.5");
    CHECK_NEAR(j.at("device").at("total_memory_bytes").num, 4294967296.0, 0);
    CHECK_NEAR(j.at("throughput").at("samples_per_second").num, 1.5e6, 1e-6);
    CHECK_NEAR(j.at("throughput").at("rays_per_second").num, 2.5e7, 1e-6);
    CHECK_NEAR(j.at("peak_vram_bytes").num, 1500000000.0, 0);
    // distinct values 11..55 catch permuted queue slots
    CHECK_NEAR(j.at("queue_high_water").at("lambert").num, 11, 0);
    CHECK_NEAR(j.at("queue_high_water").at("metal").num, 22, 0);
    CHECK_NEAR(j.at("queue_high_water").at("dielectric").num, 33, 0);
    CHECK_NEAR(j.at("queue_high_water").at("shadow").num, 44, 0);
    CHECK_NEAR(j.at("queue_high_water").at("active").num, 55, 0);
    CHECK(j.at("outputs").at("png").str == "out/img.png");
    CHECK(j.at("outputs").at("pfm").str == "out/img.pfm");
}

TEST(report_hashes_are_sixteen_lowercase_hex_digits_padded_with_zeros) {
    Json j = parsed_full();
    CHECK(is_hex16_lower(j.at("scene").at("hash").str));
    CHECK(is_hex16_lower(j.at("outputs").at("pfm_hash").str));
    CHECK(j.at("scene").at("hash").str == "0123456789abcdef");
    CHECK(j.at("outputs").at("pfm_hash").str == "00000000000000ff");
    wf::Report r = full_report();
    r.scene_hash = 0xCBF29CE484222325ull;  // upper-case letters in the source value
    Json k = wf::json_parse(wf::report_to_json(r));
    CHECK(k.at("scene").at("hash").str == "cbf29ce484222325");
}

TEST(report_timings_are_order_statistics_of_the_runs_with_min_le_median_le_max) {
    // run_ms = {12,30,10,13,11}: sorted 10 11 12 13 30 -> min 10, median 12, p10 10.4, p90 23.2, max 30.
    Json j = parsed_full();
    const Json& t = j.at("timings_ms");
    CHECK(t.at("runs").arr.size() == 5);
    const double in[5] = {12, 30, 10, 13, 11};
    for (size_t i = 0; i < t.at("runs").arr.size() && i < 5; ++i) CHECK_NEAR(t.at("runs").arr[i].num, in[i], 1e-12);
    CHECK_NEAR(t.at("min").num, 10.0, 1e-9);
    CHECK_NEAR(t.at("median").num, 12.0, 1e-9);
    CHECK_NEAR(t.at("p10").num, 10.4, 1e-9);
    CHECK_NEAR(t.at("p90").num, 23.2, 1e-9);
    CHECK_NEAR(t.at("max").num, 30.0, 1e-9);
    CHECK(t.at("min").num <= t.at("median").num);
    CHECK(t.at("median").num <= t.at("max").num);
    // Two runs: median is the mean of both.
    Json c = wf::json_parse(wf::report_to_json(cpu_report()));
    CHECK_NEAR(c.at("timings_ms").at("median").num, 6.0, 1e-9);
    CHECK_NEAR(c.at("timings_ms").at("min").num, 5.0, 1e-9);
    CHECK_NEAR(c.at("timings_ms").at("max").num, 7.0, 1e-9);
}

// ================================================================ validate_report_json

TEST(validator_accepts_the_reserialised_baseline_used_by_the_mutation_tests) {
    // Guards every mutation test below: the untouched tree validates, so any rejection is caused by the mutation.
    Json j = parsed_full();
    CHECK(wf::validate_report_json(dump(j)).empty());
}

TEST(validator_rejects_each_missing_required_key) {
    struct P { std::vector<const char*> path; };
    const std::vector<std::vector<const char*>> leaves = {
        {"schema_version"}, {"git_revision"}, {"scene"}, {"scene", "name"}, {"scene", "hash"}, {"config"},
        {"config", "width"}, {"config", "height"}, {"config", "spp"}, {"config", "seed"},
        {"config", "max_depth"}, {"config", "mode"}, {"device"}, {"device", "name"},
        {"device", "driver_version"}, {"device", "cuda_runtime_version"}, {"device", "cuda_driver_version"},
        {"device", "compute_capability"}, {"device", "total_memory_bytes"}, {"timings_ms"},
        {"timings_ms", "runs"}, {"timings_ms", "min"}, {"timings_ms", "median"}, {"timings_ms", "p10"},
        {"timings_ms", "p90"}, {"timings_ms", "max"}, {"throughput"}, {"throughput", "samples_per_second"},
        {"throughput", "rays_per_second"}, {"peak_vram_bytes"}, {"queue_high_water"},
        {"queue_high_water", "lambert"}, {"queue_high_water", "metal"}, {"queue_high_water", "dielectric"},
        {"queue_high_water", "shadow"}, {"queue_high_water", "active"}, {"outputs"}, {"outputs", "png"},
        {"outputs", "pfm"}, {"outputs", "pfm_hash"}, {"validity"}, {"validity", "nonfinite_pixels"},
        {"validity", "queue_overflows"}, {"validity", "invalid_hits"}};
    for (const auto& path : leaves) {
        Json j = parsed_full();
        Json* parent = &j;
        for (size_t i = 0; i + 1 < path.size(); ++i) parent = &parent->obj.at(path[i]);
        size_t erased = parent->obj.erase(path.back());
        std::string msg = wf::validate_report_json(dump(j));
        if (erased != 1 || msg.empty()) {
            std::printf("  FAIL removing key '%s' (depth %zu) was not rejected (erased=%zu)\n", path.back(), path.size(), erased);
            ++wftest::failures();
        }
    }
}

TEST(validator_rejects_each_value_of_the_wrong_type) {
    // Replace each node by a boolean, which is the wrong type for every schema position.
    const std::vector<std::vector<const char*>> leaves = {
        {"schema_version"}, {"git_revision"}, {"scene"}, {"scene", "name"}, {"scene", "hash"},
        {"config", "width"}, {"config", "height"}, {"config", "spp"}, {"config", "seed"},
        {"config", "max_depth"}, {"config", "mode"}, {"device", "name"}, {"device", "total_memory_bytes"},
        {"timings_ms"}, {"timings_ms", "runs"}, {"timings_ms", "min"}, {"timings_ms", "median"},
        {"timings_ms", "p10"}, {"timings_ms", "p90"}, {"timings_ms", "max"}, {"throughput"},
        {"throughput", "samples_per_second"}, {"throughput", "rays_per_second"}, {"peak_vram_bytes"},
        {"queue_high_water", "lambert"}, {"queue_high_water", "active"}, {"outputs", "png"},
        {"outputs", "pfm_hash"}, {"validity", "nonfinite_pixels"}, {"validity", "queue_overflows"},
        {"validity", "invalid_hits"}};
    for (const auto& path : leaves) {
        Json j = parsed_full();
        Json* n = &j;
        for (const char* k : path) n = &n->obj.at(k);
        Json wrong; wrong.type = Json::Bool; wrong.b = true;
        *n = wrong;
        if (wf::validate_report_json(dump(j)).empty()) {
            std::printf("  FAIL boolean in place of '%s' (depth %zu) was accepted\n", path.back(), path.size());
            ++wftest::failures();
        }
    }
}

TEST(validator_rejects_number_where_string_expected_and_string_where_number_expected) {
    {
        Json j = parsed_full();
        j.obj.at("schema_version").type = Json::String; j.obj.at("schema_version").str = "1";
        CHECK(!wf::validate_report_json(dump(j)).empty());
    }
    {
        Json j = parsed_full();
        Json& w = j.obj.at("config").obj.at("width");
        w.type = Json::String; w.str = "320";
        CHECK(!wf::validate_report_json(dump(j)).empty());
    }
    {
        Json j = parsed_full();
        Json& n = j.obj.at("scene").obj.at("name");
        n.type = Json::Number; n.num = 5;
        CHECK(!wf::validate_report_json(dump(j)).empty());
    }
    {
        Json j = parsed_full();
        Json& r = j.obj.at("timings_ms").obj.at("runs").arr[0];  // an element of runs that is not a number
        r.type = Json::String; r.str = "fast";
        CHECK(!wf::validate_report_json(dump(j)).empty());
    }
}

TEST(validator_rejects_schema_version_other_than_one) {
    for (double v : {2.0, 0.0}) {
        Json j = parsed_full();
        j.obj.at("schema_version").num = v;
        CHECK(!wf::validate_report_json(dump(j)).empty());
    }
}

TEST(validator_rejects_empty_runs_array) {
    Json j = parsed_full();
    j.obj.at("timings_ms").obj.at("runs").arr.clear();
    CHECK(!wf::validate_report_json(dump(j)).empty());
}

TEST(validator_rejects_min_greater_than_median_and_median_greater_than_max) {
    {
        // baseline min 10, median 12, max 30; min = 13 breaks only min <= median.
        Json j = parsed_full();
        j.obj.at("timings_ms").obj.at("min").num = 13.0;
        CHECK(!wf::validate_report_json(dump(j)).empty());
    }
    {
        // median = 31 breaks only median <= max (min 10 <= 31).
        Json j = parsed_full();
        j.obj.at("timings_ms").obj.at("median").num = 31.0;
        CHECK(!wf::validate_report_json(dump(j)).empty());
    }
    {
        // Boundary: min == median == max is legal (a single-run report).
        Json j = parsed_full();
        Json& t = j.obj.at("timings_ms");
        t.obj.at("min").num = 12; t.obj.at("median").num = 12; t.obj.at("max").num = 12;
        CHECK(wf::validate_report_json(dump(j)).empty());
    }
}

TEST(validator_rejects_hashes_that_are_not_sixteen_hex_digits) {
    const char* bad[] = {"abc", "", "0123456789abcde", "0123456789abcdef0", "g123456789abcdef", "0123456789abcdeZ"};
    for (const char* h : bad) {
        Json a = parsed_full();
        a.obj.at("scene").obj.at("hash").str = h;
        if (wf::validate_report_json(dump(a)).empty()) { std::printf("  FAIL scene.hash '%s' accepted\n", h); ++wftest::failures(); }
        Json b = parsed_full();
        b.obj.at("outputs").obj.at("pfm_hash").str = h;
        if (wf::validate_report_json(dump(b)).empty()) { std::printf("  FAIL outputs.pfm_hash '%s' accepted\n", h); ++wftest::failures(); }
    }
}

TEST(validator_rejects_text_that_is_not_json_or_not_an_object) {
    CHECK(!wf::validate_report_json("this is not json").empty());
    CHECK(!wf::validate_report_json("").empty());
    CHECK(!wf::validate_report_json("{").empty());
    CHECK(!wf::validate_report_json("[1,2,3]").empty());
    CHECK(!wf::validate_report_json("{}").empty());
    // trailing garbage after an otherwise valid document
    CHECK(!wf::validate_report_json(wf::report_to_json(full_report()) + " xyz").empty());
}

TEST(validator_accepts_null_git_revision_and_null_rays_but_rejects_null_elsewhere) {
    {
        Json j = parsed_full();
        j.obj.at("git_revision") = Json{};
        j.obj.at("throughput").obj.at("rays_per_second") = Json{};
        CHECK(wf::validate_report_json(dump(j)).empty());
    }
    {
        Json j = parsed_full();
        j.obj.at("scene").obj.at("name") = Json{};  // null where a string is required
        CHECK(!wf::validate_report_json(dump(j)).empty());
    }
}
