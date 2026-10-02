// Tests for wf::parse_args / usage / mode_name (include/wf/config.hpp).
// Oracle: the contract comments in config.hpp (documented flags, defaults, error list).
// Every error test builds an otherwise valid command line, so the single offending flag is the only
// possible cause of the rejection.
#include "test_framework.hpp"
#include "wf/config.hpp"

#include <string>
#include <vector>

using wf::Config;
using wf::ConfigError;
using wf::Mode;

namespace {

// Builds argv = {"wf", "render", args...} and parses it.
Config parse(const std::vector<std::string>& args, bool with_render = true) {
    std::vector<std::string> store{"wf"};
    if (with_render) store.push_back("render");
    store.insert(store.end(), args.begin(), args.end());
    std::vector<const char*> argv;
    for (auto& s : store) argv.push_back(s.c_str());
    return wf::parse_args(static_cast<int>(argv.size()), argv.data());
}

// Returns what() if parse_args threw exactly a ConfigError, "<no throw>" or "<wrong type>" otherwise.
std::string error_of(const std::vector<std::string>& args, bool with_render = true) {
    try {
        (void)parse(args, with_render);
    } catch (const ConfigError& e) {
        return e.what();
    } catch (...) {
        return "<wrong type>";
    }
    return "<no throw>";
}

bool names(const std::string& msg, const std::string& flag) { return msg.find(flag) != std::string::npos; }

// Given a command line with one bad flag, then ConfigError is thrown and its message names that flag.
#define CHECK_REJECTS(flag, ...)                                                     \
    do {                                                                              \
        std::string _m = error_of(__VA_ARGS__);                                       \
        if (_m == "<no throw>" || _m == "<wrong type>") {                             \
            std::printf("  FAIL %s:%d %s: expected ConfigError, got %s\n", __FILE__, __LINE__, flag, _m.c_str()); \
            ++wftest::failures();                                                     \
        } else if (!names(_m, flag)) {                                                \
            std::printf("  FAIL %s:%d message '%s' does not name %s\n", __FILE__, __LINE__, _m.c_str(), flag); \
            ++wftest::failures();                                                     \
        }                                                                             \
    } while (0)

}  // namespace

// ---------------------------------------------------------------- defaults and happy path

TEST(defaults_when_no_flags_given) {
    // Oracle: default member initialisers documented in config.hpp.
    Config c = parse({});
    CHECK(c.scene == "cornell");
    CHECK(c.width == 1280);
    CHECK(c.height == 720);
    CHECK(c.spp == 64);
    CHECK(c.seed == 1234u);
    CHECK(c.max_depth == 8);
    CHECK(c.mode == Mode::Wavefront);
    CHECK(c.report_path.empty());
    CHECK(c.out_prefix.empty());
    CHECK(c.threads == 0);
    CHECK(c.reps == 1);
    CHECK(c.warmup == 0);
}

TEST(every_documented_flag_is_parsed_into_its_own_field) {
    // All values differ from the defaults and from each other where types allow
    // (width != height catches a swap, spp != max-depth != reps != warmup catches crossed fields).
    Config c = parse({"--scene", "assets/bunny.obj", "--width", "320", "--height", "200", "--spp", "17",
                      "--seed", "4242", "--max-depth", "5", "--mode", "cpu", "--report", "out/r.json",
                      "--out", "out/img", "--threads", "3", "--reps", "4", "--warmup", "2"});
    CHECK(c.scene == "assets/bunny.obj");
    CHECK(c.width == 320);
    CHECK(c.height == 200);
    CHECK(c.spp == 17);
    CHECK(c.seed == 4242u);
    CHECK(c.max_depth == 5);
    CHECK(c.mode == Mode::Cpu);
    CHECK(c.report_path == "out/r.json");
    CHECK(c.out_prefix == "out/img");
    CHECK(c.threads == 3);
    CHECK(c.reps == 4);
    CHECK(c.warmup == 2);
}

TEST(omitted_flags_keep_defaults_when_only_one_flag_is_given) {
    Config c = parse({"--spp", "5"});
    CHECK(c.spp == 5);
    CHECK(c.scene == "cornell");
    CHECK(c.width == 1280);
    CHECK(c.height == 720);
    CHECK(c.seed == 1234u);
    CHECK(c.max_depth == 8);
    CHECK(c.mode == Mode::Wavefront);
    CHECK(c.reps == 1);
    CHECK(c.warmup == 0);
    CHECK(c.threads == 0);
}

TEST(each_mode_string_parses_to_its_enum_value) {
    CHECK(parse({"--mode", "cpu"}).mode == Mode::Cpu);
    CHECK(parse({"--mode", "megakernel"}).mode == Mode::Megakernel);
    CHECK(parse({"--mode", "wavefront"}).mode == Mode::Wavefront);
}

TEST(flag_order_does_not_matter) {
    Config a = parse({"--width", "10", "--height", "20", "--spp", "3"});
    Config b = parse({"--spp", "3", "--height", "20", "--width", "10"});
    CHECK(a.width == b.width && a.height == b.height && a.spp == b.spp);
    CHECK(a.width == 10 && a.height == 20 && a.spp == 3);
}

TEST(repeated_flag_last_occurrence_wins) {
    CHECK(parse({"--spp", "4", "--spp", "9"}).spp == 9);
    CHECK(parse({"--mode", "cpu", "--mode", "megakernel"}).mode == Mode::Megakernel);
    CHECK(parse({"--scene", "a", "--scene", "b"}).scene == "b");
}

TEST(range_boundaries_are_accepted) {
    // The documented ranges are closed: width/height 1..8192, spp >= 1, max-depth 1..64, reps >= 1, warmup >= 0.
    CHECK(parse({"--width", "1"}).width == 1);
    CHECK(parse({"--width", "8192"}).width == 8192);
    CHECK(parse({"--height", "1"}).height == 1);
    CHECK(parse({"--height", "8192"}).height == 8192);
    CHECK(parse({"--spp", "1"}).spp == 1);
    CHECK(parse({"--max-depth", "1"}).max_depth == 1);
    CHECK(parse({"--max-depth", "64"}).max_depth == 64);
    CHECK(parse({"--reps", "1"}).reps == 1);
    CHECK(parse({"--warmup", "0"}).warmup == 0);
}

// ---------------------------------------------------------------- errors

TEST(width_outside_1_to_8192_is_rejected_naming_width) {
    CHECK_REJECTS("--width", {"--width", "0"});
    CHECK_REJECTS("--width", {"--width", "8193"});
    CHECK_REJECTS("--width", {"--width", "-5"});
}

TEST(height_outside_1_to_8192_is_rejected_naming_height) {
    CHECK_REJECTS("--height", {"--height", "0"});
    CHECK_REJECTS("--height", {"--height", "8193"});
}

TEST(spp_below_one_is_rejected_naming_spp) {
    CHECK_REJECTS("--spp", {"--spp", "0"});
    CHECK_REJECTS("--spp", {"--spp", "-3"});
}

TEST(max_depth_outside_1_to_64_is_rejected_naming_max_depth) {
    CHECK_REJECTS("--max-depth", {"--max-depth", "0"});
    CHECK_REJECTS("--max-depth", {"--max-depth", "65"});
}

TEST(reps_below_one_is_rejected_naming_reps) {
    CHECK_REJECTS("--reps", {"--reps", "0"});
}

TEST(negative_warmup_is_rejected_naming_warmup) {
    CHECK_REJECTS("--warmup", {"--warmup", "-1"});
}

TEST(unknown_mode_is_rejected_naming_mode) {
    CHECK_REJECTS("--mode", {"--mode", "gpu"});
    CHECK_REJECTS("--mode", {"--mode", ""});
}

TEST(non_numeric_value_is_rejected_naming_each_numeric_flag) {
    CHECK_REJECTS("--width", {"--width", "abc"});
    CHECK_REJECTS("--height", {"--height", "abc"});
    CHECK_REJECTS("--spp", {"--spp", "abc"});
    CHECK_REJECTS("--seed", {"--seed", "abc"});
    CHECK_REJECTS("--max-depth", {"--max-depth", "abc"});
    CHECK_REJECTS("--threads", {"--threads", "abc"});
    CHECK_REJECTS("--reps", {"--reps", "abc"});
    CHECK_REJECTS("--warmup", {"--warmup", "abc"});
    CHECK_REJECTS("--width", {"--width", ""});
}

TEST(flag_without_value_at_end_of_line_is_rejected_naming_the_flag) {
    CHECK_REJECTS("--scene", {"--scene"});
    CHECK_REJECTS("--width", {"--width"});
    CHECK_REJECTS("--height", {"--height"});
    CHECK_REJECTS("--spp", {"--spp"});
    CHECK_REJECTS("--seed", {"--seed"});
    CHECK_REJECTS("--max-depth", {"--max-depth"});
    CHECK_REJECTS("--mode", {"--mode"});
    CHECK_REJECTS("--report", {"--report"});
    CHECK_REJECTS("--out", {"--out"});
    CHECK_REJECTS("--threads", {"--threads"});
    CHECK_REJECTS("--reps", {"--reps"});
    CHECK_REJECTS("--warmup", {"--warmup"});
}

TEST(missing_value_after_valid_flags_names_the_trailing_flag) {
    // The valid prefix must not mask or mis-attribute the error.
    CHECK_REJECTS("--spp", {"--width", "64", "--spp"});
}

TEST(unknown_flag_is_rejected_naming_that_flag) {
    CHECK_REJECTS("--bogus", {"--bogus", "1"});
    CHECK_REJECTS("--bogus", {"--width", "64", "--bogus"});
}

TEST(first_argument_must_be_render) {
    // Valid flags, wrong command: only argv[1] can be the cause.
    CHECK(error_of({"--width", "64"}, /*with_render=*/false) != "<no throw>");
    CHECK(error_of({"--width", "64"}, false) != "<wrong type>");
    std::vector<const char*> only_prog{"wf"};
    bool threw = false;
    try { (void)wf::parse_args(1, only_prog.data()); } catch (const ConfigError&) { threw = true; }
    CHECK(threw);
    std::vector<const char*> bench{"wf", "bench"};
    threw = false;
    try { (void)wf::parse_args(2, bench.data()); } catch (const ConfigError&) { threw = true; }
    CHECK(threw);
}

TEST(rejected_parse_does_not_affect_the_next_parse) {
    // No hidden global state: after a failure, a valid line yields pure defaults plus its own flag.
    (void)error_of({"--width", "0"});
    Config c = parse({"--spp", "2"});
    CHECK(c.width == 1280);
    CHECK(c.spp == 2);
}

// ---------------------------------------------------------------- usage / mode_name

TEST(usage_mentions_the_command_and_every_documented_flag) {
    std::string u = wf::usage();
    CHECK(u.find("render") != std::string::npos);
    const char* flags[] = {"--scene", "--width", "--height", "--spp", "--seed", "--max-depth",
                           "--mode", "--report", "--out", "--threads", "--reps", "--warmup"};
    for (const char* f : flags) {
        if (u.find(f) == std::string::npos) {
            std::printf("  FAIL usage() does not mention %s\n", f);
            ++wftest::failures();
        }
    }
}

TEST(mode_name_gives_documented_strings_and_round_trips_through_parse) {
    CHECK(wf::mode_name(Mode::Cpu) == "cpu");
    CHECK(wf::mode_name(Mode::Megakernel) == "megakernel");
    CHECK(wf::mode_name(Mode::Wavefront) == "wavefront");
    const Mode all[] = {Mode::Cpu, Mode::Megakernel, Mode::Wavefront};
    for (Mode m : all) CHECK(parse({"--mode", wf::mode_name(m)}).mode == m);
}
