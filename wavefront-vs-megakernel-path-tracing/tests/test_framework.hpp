// Minimal test framework: TEST(name) { CHECK(cond); CHECK_NEAR(a,b,tol); }  Run all with main().
#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace wftest {
struct Case { const char* name; std::function<void()> fn; };
inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }
inline int& failures() { static int f = 0; return f; }
struct Reg { Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };
}  // namespace wftest

#define TEST(name) \
    static void test_##name(); \
    static wftest::Reg reg_##name(#name, test_##name); \
    static void test_##name()

#define CHECK(cond) do { if (!(cond)) { std::printf("  FAIL %s:%d CHECK(%s)\n", __FILE__, __LINE__, #cond); ++wftest::failures(); } } while (0)
#define CHECK_NEAR(a, b, tol) do { double _a = (a), _b = (b), _t = (tol); \
    if (!(std::fabs(_a - _b) <= _t)) { std::printf("  FAIL %s:%d CHECK_NEAR(%s=%.9g, %s=%.9g, tol=%.3g)\n", __FILE__, __LINE__, #a, _a, #b, _b, _t); ++wftest::failures(); } } while (0)
#define CHECK_THROWS(expr) do { bool _t = false; try { (void)(expr); } catch (...) { _t = true; } \
    if (!_t) { std::printf("  FAIL %s:%d expected exception: %s\n", __FILE__, __LINE__, #expr); ++wftest::failures(); } } while (0)

int main() {
    int run = 0, bad = 0;
    for (auto& c : wftest::registry()) {
        int before = wftest::failures();
        try { c.fn(); } catch (const std::exception& e) { std::printf("  FAIL exception: %s\n", e.what()); ++wftest::failures(); }
        catch (...) { std::printf("  FAIL unknown exception\n"); ++wftest::failures(); }
        bool ok = wftest::failures() == before;
        std::printf("[%s] %s\n", ok ? " ok " : "FAIL", c.name);
        ++run; if (!ok) ++bad;
    }
    std::printf("%d tests, %d failed\n", run, bad);
    return bad ? 1 : 0;
}
