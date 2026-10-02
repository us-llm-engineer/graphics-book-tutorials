// behaviour and artifact schemas of scripts/run_benchmarks.sh, run_sanitizers.sh and
// profile_ncu.sh, run with --quick through std::system. Working directory is ROOT; results go to
// ROOT/results. A case SKIPs (with a message) when the binary it needs is missing, never passes silently.
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "wf/json.hpp"

using namespace wf;
namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;

namespace {
const char* kScripts[3] = {"scripts/run_benchmarks.sh", "scripts/run_sanitizers.sh", "scripts/profile_ncu.sh"};

std::string build_dir() {
    const char* e = std::getenv("WF_BUILD_DIR");
    return e ? e : "build";
}
std::string read_text(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
struct Run { int code; std::string output; };
// Runs `cmd` (stdout+stderr merged) and returns exit code and output.
Run run(const std::string& cmd) {
    static int n = 0;
    fs::path o = fs::temp_directory_path() / ("r23s2_" + std::to_string((long)::getpid()) + "_" + std::to_string(n++) + ".txt");
    int rc = std::system((cmd + " >'" + o.string() + "' 2>&1").c_str());
    Run r{WIFEXITED(rc) ? WEXITSTATUS(rc) : -1, read_text(o)};
    fs::remove(o);
    return r;
}
std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
bool executable(const std::string& p) { return ::access(p.c_str(), X_OK) == 0; }
bool need_bin(const std::string& rel) { return fs::exists(fs::path(build_dir()) / rel); }
std::vector<fs::path> result_files() {
    std::vector<fs::path> v;
    if (!fs::exists("results")) return v;
    for (auto& e : fs::directory_iterator("results")) {
        std::string n = e.path().filename().string();
        if (e.is_regular_file() && (n.rfind("bench-", 0) == 0 || n == "sanitizers.json")) v.push_back(e.path());
    }
    if (fs::exists("results/profiles"))
        for (auto& e : fs::directory_iterator("results/profiles")) if (e.is_regular_file()) v.push_back(e.path());
    return v;
}
std::set<std::string> keys_of(const Json& j) {
    std::set<std::string> s;
    for (auto& kv : j.obj) s.insert(kv.first);
    return s;
}
void check_bench_file(const std::string& scene) {
    fs::path p = "results/bench-" + scene + ".json";
    REQUIRE(fs::exists(p));
    Json j = json_parse(read_text(p));  // throws (fails the test) if not JSON
    CHECK(j.at("schema_version").num == 1);
    CHECK(j.at("scene_name").str == scene);
    CHECK(j.at("width").num == 64);
    CHECK(j.at("height").num == 36);
    CHECK(j.at("spp").num == 4);
    CHECK(j.at("reps").num == 3);
    CHECK(j.at("warmup").num == 1);
    const std::string v = j.at("verdict").str;
    CHECK((v == "hypothesis_supported" || v == "hypothesis_weakened"));
    REQUIRE(j.at("conditions").arr.size() == 2);
    for (auto& c : j.at("conditions").arr) CHECK(c.at("run_ms").arr.size() == 3);
}
}  // namespace

TEST_CASE("every script is executable, strict, and rejects an unknown flag with exit 2 and usage", "[scripts]") {
    for (const char* s : kScripts) {
        INFO(s);
        REQUIRE(fs::exists(s));
        CHECK(executable(s));
        std::string text = read_text(s);
        // strict mode must be in the first lines, before any command
        std::string head = text.substr(0, std::min<size_t>(text.size(), 400));
        CHECK_THAT(head, ContainsSubstring("set -euo pipefail"));
        CHECK(run(std::string("bash -n ") + s).code == 0);  // parses
        Run r = run(std::string("bash ") + s + " --definitely-not-a-flag");
        CHECK(r.code == 2);
        CHECK_THAT(lower(r.output), ContainsSubstring("usage"));
        CHECK_THAT(r.output, ContainsSubstring("--definitely-not-a-flag"));
    }
}

TEST_CASE("run_benchmarks --quick writes one valid result file per scene", "[scripts]") {
    if (!need_bin("wfbench")) SKIP("build/wfbench does not exist: build it first (cmake --build build --target wfbench)");
    fs::remove("results/bench-cornell.json");
    fs::remove("results/bench-divergence.json");
    Run r = run("bash scripts/run_benchmarks.sh --quick");
    INFO(r.output);
    REQUIRE(r.code == 0);
    check_bench_file("cornell");
    check_bench_file("divergence");
}

TEST_CASE("run_benchmarks is idempotent: a second run overwrites with the same schema", "[scripts]") {
    if (!need_bin("wfbench")) SKIP("build/wfbench does not exist: build it first");
    fs::remove("results/bench-cornell.json");
    REQUIRE(run("bash scripts/run_benchmarks.sh --quick").code == 0);
    Json first = json_parse(read_text("results/bench-cornell.json"));
    fs::remove("results/bench-cornell.json");  // a stale file would hide a script that skips the write
    Run r = run("bash scripts/run_benchmarks.sh --quick");
    INFO(r.output);
    REQUIRE(r.code == 0);
    REQUIRE(fs::exists("results/bench-cornell.json"));
    Json second = json_parse(read_text("results/bench-cornell.json"));
    CHECK(keys_of(first) == keys_of(second));
    CHECK(second.at("schema_version").num == 1);
    CHECK(second.at("scene_hash").num == first.at("scene_hash").num);  // same scene, same hash
    CHECK(second.at("conditions").arr.size() == first.at("conditions").arr.size());
    // overwrite, not append: still one JSON document (json_parse rejects trailing garbage)
    CHECK_NOTHROW(json_parse(read_text("results/bench-cornell.json")));
}

TEST_CASE("run_sanitizers --quick writes the documented schema and an exit code matching error_count", "[scripts]") {
    if (!need_bin("wf")) SKIP("build/wf does not exist: build it first");
    fs::remove("results/sanitizers.json");
    Run r = run("bash scripts/run_sanitizers.sh --quick");
    INFO(r.output);
    REQUIRE(fs::exists("results/sanitizers.json"));
    Json j = json_parse(read_text("results/sanitizers.json"));
    CHECK(j.at("schema_version").num == 1);
    REQUIRE(j.at("runs").arr.size() >= 2);
    bool mega = false, wave = false, asan = false, all_zero = true;
    for (auto& run_j : j.at("runs").arr) {
        for (const char* k : {"tool", "command", "exit_code", "error_count", "first_failure"}) CHECK(run_j.has(k));
        CHECK(run_j.at("tool").type == Json::String);
        CHECK(run_j.at("command").type == Json::String);
        CHECK(run_j.at("exit_code").type == Json::Number);
        CHECK(run_j.at("error_count").type == Json::Number);
        const std::string cmd = run_j.at("command").str, tool = lower(run_j.at("tool").str);
        mega |= cmd.find("--mode megakernel") != std::string::npos;
        wave |= cmd.find("--mode wavefront") != std::string::npos;
        asan |= tool.find("asan") != std::string::npos || tool.find("address") != std::string::npos;
        if (run_j.at("error_count").num != 0) {
            all_zero = false;
            CHECK(!run_j.at("first_failure").str.empty());  // a failure must say what failed
        }
    }
    CHECK(mega);
    CHECK(wave);
    CHECK(asan);
    CHECK(r.code == (all_zero ? 0 : 1));  // exit 0 iff every error_count is 0, else exactly 1
}

TEST_CASE("profile_ncu --quick exits 0 and records either csv profiles or error files naming ncu", "[scripts]") {
    if (!need_bin("wf")) SKIP("build/wf does not exist: build it first");
    fs::remove_all("results/profiles");
    Run r = run("bash scripts/profile_ncu.sh --quick");
    INFO(r.output);
    REQUIRE(r.code == 0);  // unavailable counters are evidence, not failure
    REQUIRE(fs::exists("results/profiles/summary.json"));
    Json j = json_parse(read_text("results/profiles/summary.json"));
    CHECK(j.at("schema_version").num == 1);
    REQUIRE(j.at("available").type == Json::Bool);
    CHECK(j.at("command").type == Json::String);
    CHECK(j.at("message").type == Json::String);
    for (const char* mode : {"megakernel", "wavefront"}) {
        INFO(mode);
        if (j.at("available").b) {
            CHECK(fs::exists(std::string("results/profiles/") + mode + ".csv"));
        } else {
            fs::path e = std::string("results/profiles/") + mode + ".error.txt";
            REQUIRE(fs::exists(e));
            std::string t = read_text(e);
            CHECK_THAT(t, ContainsSubstring("ncu"));
            CHECK(!j.at("command").str.empty());
            CHECK_THAT(t, ContainsSubstring(j.at("command").str));  // the exact failing command is recorded
            CHECK(!j.at("message").str.empty());
        }
    }
}

TEST_CASE("with WF_BUILD_DIR at an empty directory every script fails and names the missing binary", "[scripts]") {
    // Directory name deliberately avoids the letters 'wf' so the name check cannot pass by accident.
    fs::path d = fs::temp_directory_path() / ("emptybuild_" + std::to_string((long)::getpid()));
    fs::remove_all(d);
    fs::create_directories(d);
    struct C { const char* script; const char* binary; } cases[3] = {
        {"scripts/run_benchmarks.sh", "wfbench"}, {"scripts/run_sanitizers.sh", "wf"}, {"scripts/profile_ncu.sh", "wf"}};
    for (auto& c : cases) {
        INFO(c.script);
        Run r = run("WF_BUILD_DIR='" + d.string() + "' bash " + c.script + " --quick");
        CHECK(r.code != 0);
        // the message must contain <dir>/<binary> not followed by more letters (so "wf" is not satisfied by "wfbench")
        std::string needle = d.string() + "/" + c.binary;
        size_t pos = r.output.find(needle);
        bool ok = false;
        while (pos != std::string::npos) {
            size_t end = pos + needle.size();
            if (end >= r.output.size() || !std::isalnum((unsigned char)r.output[end])) { ok = true; break; }
            pos = r.output.find(needle, pos + 1);
        }
        INFO(r.output);
        CHECK(ok);
    }
    fs::remove_all(d);
}

TEST_CASE("result files never embed an absolute home path", "[scripts]") {
    // Produce at least one fresh artifact set so the scan is not vacuous.
    if (!need_bin("wf")) SKIP("build/wf does not exist: build it first");
    REQUIRE(run("bash scripts/profile_ncu.sh --quick").code == 0);
    auto files = result_files();
    REQUIRE(!files.empty());
    for (auto& p : files) {
        INFO(p.string());
        std::string t = read_text(p);
        CHECK(t.find("/home/") == std::string::npos);
        CHECK(t.find("/Users/") == std::string::npos);
    }
    for (const char* s : kScripts) {
        INFO(s);
        CHECK(read_text(s).find("/home/") == std::string::npos);  // no hard-coded home path in the scripts either
    }
}
