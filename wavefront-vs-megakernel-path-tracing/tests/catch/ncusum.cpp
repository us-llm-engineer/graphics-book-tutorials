// ncusum parsing and failure mapping. Hand-written Nsight Compute CSV fixtures are written
// to a temp dir; the real binary (env NCUSUM_BIN, default build/ncusum) is run and its JSON parsed.
// Working directory is ROOT. Oracles are hand-computed means noted at each case.
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "wf/json.hpp"

using namespace wf;
namespace fs = std::filesystem;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::ContainsSubstring;

namespace {

const char* kOcc = "sm__warps_active.avg.pct_of_peak_sustained_active";
const char* kWarp = "smsp__thread_inst_executed_per_inst_executed.ratio";
const char* kReg = "launch__registers_per_thread";
const char* kDram = "dram__throughput.avg.pct_of_peak_sustained_elapsed";
const char* kSm = "sm__throughput.avg.pct_of_peak_sustained_elapsed";
const char* kHeader = "\"ID\",\"Kernel Name\",\"Metric Name\",\"Metric Unit\",\"Metric Value\"";

fs::path fresh_dir() {
    static int n = 0;
    fs::path d = fs::temp_directory_path() / ("ncs_" + std::to_string((long)::getpid()) + "_" + std::to_string(n++));
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}
std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
void write_text(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::binary);
    f << s;
}
std::string bin() {
    const char* e = std::getenv("NCUSUM_BIN");
    return e ? e : "build/ncusum";
}
struct Run { int code; std::string out, err; };
// Runs ncusum with extra args; stdout and stderr captured separately.
Run run_ncusum(const fs::path& dir, const std::string& args) {
    fs::path o = dir / "stdout.txt", e = dir / "stderr.txt";
    std::string cmd = "'" + bin() + "' " + args + " >'" + o.string() + "' 2>'" + e.string() + "'";
    int rc = std::system(cmd.c_str());
    Run r;
    r.code = WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
    r.out = read_text(o);
    r.err = read_text(e);
    return r;
}
std::string row(int id, const std::string& kernel, const std::string& metric, const std::string& unit,
                const std::string& value, const char* eol = "\n") {
    return "\"" + std::to_string(id) + "\",\"" + kernel + "\",\"" + metric + "\",\"" + unit + "\",\"" + value + "\"" + eol;
}
const Json* kernel_named(const Json& j, const std::string& name) {
    for (const Json& k : j.at("kernels").arr)
        if (k.at("name").str == name) return &k;
    return nullptr;
}
const Json* metric_named(const Json& k, const std::string& name) {
    for (const Json& m : k.at("metrics").arr)
        if (m.at("name").str == name) return &m;
    return nullptr;
}
bool same(const Json& a, const Json& b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case Json::Null: return true;
        case Json::Bool: return a.b == b.b;
        case Json::Number: return a.num == b.num;
        case Json::String: return a.str == b.str;
        case Json::Array:
            if (a.arr.size() != b.arr.size()) return false;
            for (size_t i = 0; i < a.arr.size(); ++i) if (!same(a.arr[i], b.arr[i])) return false;
            return true;
        case Json::Object:
            if (a.obj.size() != b.obj.size()) return false;
            for (auto& kv : a.obj) { if (!b.has(kv.first) || !same(kv.second, b.at(kv.first))) return false; }
            return true;
    }
    return false;
}
// Runs on `csv` text, requires exit 0, returns parsed stdout.
Json summarize(const std::string& csv) {
    fs::path d = fresh_dir();
    write_text(d / "in.csv", csv);
    Run r = run_ncusum(d, "'" + (d / "in.csv").string() + "'");
    INFO("stderr: " << r.err);
    REQUIRE(r.code == 0);
    return json_parse(r.out);
}
}  // namespace

TEST_CASE("single kernel reports the five summary metrics exactly", "[ncusum]") {
    // Given one kernel with one invocation of each summary metric (hand-written values)
    std::string csv = std::string(kHeader) + "\n" +
        row(0, "k", kOcc, "%", "45.5") + row(0, "k", kWarp, "", "20.25") + row(0, "k", kReg, "register/thread", "64") +
        row(0, "k", kDram, "%", "12.5") + row(0, "k", kSm, "%", "30");
    Json j = summarize(csv);
    // Then schema_version 1 and every number equals the input (oracle: the fixture itself)
    CHECK(j.at("schema_version").num == 1);
    REQUIRE(j.at("kernels").arr.size() == 1);
    const Json& k = j.at("kernels").arr[0];
    CHECK(k.at("name").str == "k");
    CHECK(k.at("invocations").num == 1);
    CHECK(k.at("metrics").arr.size() == 5);
    const Json* m = metric_named(k, kOcc);
    REQUIRE(m);
    CHECK(m->at("unit").str == "%");
    CHECK(m->at("value").num == 45.5);
    CHECK(m->at("samples").num == 1);
    const Json& s = j.at("summary");
    CHECK(s.at("achieved_occupancy_pct").num == 45.5);
    CHECK(s.at("warp_execution_efficiency").num == 20.25);
    CHECK(s.at("registers_per_thread").num == 64);
    CHECK(s.at("dram_throughput_pct").num == 12.5);
    CHECK(s.at("sm_throughput_pct").num == 30);
}

TEST_CASE("several invocations of one kernel are averaged, not last-wins", "[ncusum]") {
    // Given three invocations of kernel k with occupancy 10, 20, 60: mean (10+20+60)/3 = 30, last = 60
    std::string csv = std::string(kHeader) + "\n" + row(0, "k", kOcc, "%", "10") + row(1, "k", kOcc, "%", "20") +
                      row(2, "k", kOcc, "%", "60");
    Json j = summarize(csv);
    REQUIRE(j.at("kernels").arr.size() == 1);  // one entry per kernel name, not per invocation
    const Json& k = j.at("kernels").arr[0];
    CHECK(k.at("invocations").num == 3);
    const Json* m = metric_named(k, kOcc);
    REQUIRE(m);
    CHECK_THAT(m->at("value").num, WithinAbs(30.0, 1e-9));
    CHECK(m->at("samples").num == 3);
    CHECK_THAT(j.at("summary").at("achieved_occupancy_pct").num, WithinAbs(30.0, 1e-9));
}

TEST_CASE("quoted commas and CRLF line endings are parsed without leftovers", "[ncusum]") {
    // Given CRLF everywhere, a kernel name with commas and a value "1,234.5" with a thousands comma
    const std::string name = "void foo<int, float>(int, float *)";
    std::string csv = std::string(kHeader) + "\r\n" + row(0, name, kDram, "%", "1,234.5", "\r\n") +
                      row(1, name, kDram, "%", "2,000.5", "\r\n");
    Json j = summarize(csv);
    // Then the name is intact (a comma split would shift columns), values are 1234.5 and 2000.5 -> mean 1617.5
    REQUIRE(j.at("kernels").arr.size() == 1);
    const Json* k = kernel_named(j, name);
    REQUIRE(k);
    const Json* m = metric_named(*k, kDram);
    REQUIRE(m);
    CHECK(m->at("unit").str == "%");  // no trailing CR in any field
    CHECK(m->at("value").type == Json::Number);
    CHECK_THAT(m->at("value").num, WithinAbs(1617.5, 1e-9));
    CHECK(m->at("samples").num == 2);
}

TEST_CASE("n/a and empty values become null and are excluded from means", "[ncusum]") {
    // Given occupancy 10, "n/a", 30 (mean of numeric = 20; n/a read as 0 would give 13.33),
    // sm throughput "N/A" and "" only (all null) and warp efficiency "" , 8
    std::string csv = std::string(kHeader) + "\n" + row(0, "k", kOcc, "%", "10") + row(1, "k", kOcc, "%", "n/a") +
                      row(2, "k", kOcc, "%", "30") + row(0, "k", kSm, "%", "N/A") + row(1, "k", kSm, "%", "") +
                      row(0, "k", kWarp, "", "") + row(1, "k", kWarp, "", "8");
    Json j = summarize(csv);
    const Json& k = j.at("kernels").arr.at(0);
    const Json* occ = metric_named(k, kOcc);
    REQUIRE(occ);
    CHECK_THAT(occ->at("value").num, WithinAbs(20.0, 1e-9));
    CHECK(occ->at("samples").num == 2);
    const Json* sm = metric_named(k, kSm);
    REQUIRE(sm);
    CHECK(sm->at("value").type == Json::Null);
    CHECK(sm->at("samples").num == 0);
    const Json* w = metric_named(k, kWarp);
    REQUIRE(w);
    CHECK(w->at("value").num == 8);  // the empty value is excluded, not averaged as 0 (would give 4)
    CHECK(j.at("summary").at("sm_throughput_pct").type == Json::Null);
    CHECK(j.at("summary").at("warp_execution_efficiency").num == 8);
}

TEST_CASE("two kernels are reported separately and the summary averages over kernels", "[ncusum]") {
    // Given kernel a (occupancy 40, 2 invocations: 30,50 -> 40; registers 32) and b (occupancy 60; registers 96)
    // summary occupancy = (40+60)/2 = 50; registers = (32+96)/2 = 64 (invocation-weighted would give 46.67 / 53.3)
    std::string csv = std::string(kHeader) + "\n" + row(0, "a", kOcc, "%", "30") + row(0, "a", kReg, "r", "32") +
                      row(1, "a", kOcc, "%", "50") + row(2, "b", kOcc, "%", "60") + row(2, "b", kReg, "r", "96");
    Json j = summarize(csv);
    REQUIRE(j.at("kernels").arr.size() == 2);
    const Json* a = kernel_named(j, "a");
    const Json* b = kernel_named(j, "b");
    REQUIRE(a);
    REQUIRE(b);
    CHECK(a->at("invocations").num == 2);
    CHECK(b->at("invocations").num == 1);
    CHECK_THAT(metric_named(*a, kOcc)->at("value").num, WithinAbs(40.0, 1e-9));
    CHECK_THAT(metric_named(*b, kOcc)->at("value").num, WithinAbs(60.0, 1e-9));
    CHECK_THAT(j.at("summary").at("achieved_occupancy_pct").num, WithinAbs(50.0, 1e-9));
    CHECK_THAT(j.at("summary").at("registers_per_thread").num, WithinAbs(64.0, 1e-9));
}

TEST_CASE("summary metrics that are absent from the export are null", "[ncusum]") {
    // Given only registers_per_thread (and an unrelated metric that must not leak into the summary)
    std::string csv = std::string(kHeader) + "\n" + row(0, "k", kReg, "r", "48") + row(0, "k", "gpu__time_duration.sum", "ns", "999");
    Json j = summarize(csv);
    const Json& s = j.at("summary");
    CHECK(s.at("registers_per_thread").num == 48);
    CHECK(s.at("achieved_occupancy_pct").type == Json::Null);
    CHECK(s.at("warp_execution_efficiency").type == Json::Null);
    CHECK(s.at("dram_throughput_pct").type == Json::Null);
    CHECK(s.at("sm_throughput_pct").type == Json::Null);
    CHECK(metric_named(j.at("kernels").arr.at(0), "gpu__time_duration.sum") != nullptr);  // still listed per kernel
}

TEST_CASE("unusable inputs exit 2 with a message naming the problem and write no output", "[ncusum]") {
    struct Bad { const char* label; std::string csv; bool create; std::string must_name; };
    fs::path d = fresh_dir();
    std::vector<Bad> bad = {
        {"missing file", "", false, "nofile.csv"},
        {"empty file", "", true, "in.csv"},
        // header lacks "Metric Value"
        {"header without required columns", "\"ID\",\"Kernel Name\",\"Metric Name\",\"Metric Unit\"\n\"0\",\"k\",\"m\",\"%\"\n", true, "Metric Value"},
        // second data row has 3 fields instead of 5
        {"row with fewer fields", std::string(kHeader) + "\n" + row(0, "k", kOcc, "%", "1") + "\"1\",\"k\",\"x\"\n", true, "in.csv"},
    };
    for (const Bad& b : bad) {
        INFO(b.label);
        fs::path in = d / (b.create ? "in.csv" : "nofile.csv");
        fs::path outj = d / "out.json";
        fs::remove(outj);
        if (b.create) write_text(in, b.csv);
        Run r = run_ncusum(d, "'" + in.string() + "' --out '" + outj.string() + "'");
        CHECK(r.code == 2);
        CHECK_THAT(r.err, ContainsSubstring(b.must_name));
        CHECK_FALSE(fs::exists(outj));  // state unchanged: no partial JSON left behind
        CHECK(r.out.find("schema_version") == std::string::npos);
    }
}

TEST_CASE("--out writes the same JSON that is printed to stdout without --out", "[ncusum]") {
    fs::path d = fresh_dir();
    std::string csv = std::string(kHeader) + "\n" + row(0, "k", kOcc, "%", "10") + row(1, "k", kOcc, "%", "30") +
                      row(0, "q", kReg, "r", "n/a");
    write_text(d / "in.csv", csv);
    Run a = run_ncusum(d, "'" + (d / "in.csv").string() + "'");
    REQUIRE(a.code == 0);
    Json stdout_json = json_parse(a.out);
    fs::path outj = d / "sub_out.json";
    Run b = run_ncusum(d, "'" + (d / "in.csv").string() + "' --out '" + outj.string() + "'");
    REQUIRE(b.code == 0);
    REQUIRE(fs::exists(outj));
    Json file_json = json_parse(read_text(outj));
    CHECK(same(stdout_json, file_json));
    CHECK(file_json.at("schema_version").num == 1);
    CHECK_THAT(metric_named(file_json.at("kernels").arr.at(0), kOcc)->at("value").num, WithinAbs(20.0, 1e-9));
}
