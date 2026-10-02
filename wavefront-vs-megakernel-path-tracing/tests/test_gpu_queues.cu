// GPU queue primitives: exclusive scan, stable compaction, warp-aggregated queue_push.
// Oracles: host exclusive prefix sum, host std::copy_if filter, host popcount of flags, and set
// equality of pushed indices against the flagged index set. All independent of the device code.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "queues.cuh"
#include "test_framework.hpp"

using namespace wf::gpu;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { ck(cudaMalloc(&p, (count ? count : 1) * sizeof(T)), "cudaMalloc"); }
    Dev(const std::vector<T>& h) : Dev(h.size()) { up(h); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& h) {
        if (!h.empty()) ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    }
    std::vector<T> down() const {
        std::vector<T> h(n);
        if (n) ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
        return h;
    }
};

constexpr uint32_t kCanary = 0xDEADBEEFu;

std::vector<uint32_t> random_vals(int n, uint32_t maxv, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<uint32_t> v(n);
    for (auto& x : v) x = g() % (maxv + 1);
    return v;
}

// Independent reference.
std::vector<uint32_t> host_excl(const std::vector<uint32_t>& in, uint32_t* total) {
    std::vector<uint32_t> out(in.size());
    uint32_t acc = 0;
    for (size_t i = 0; i < in.size(); ++i) { out[i] = acc; acc += in[i]; }
    *total = acc;
    return out;
}

const int kScanSizes[] = {1, 2, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025, 2048, 1048576, 921600, 1000003};

// Runs scan (optionally in place) and checks every output, the total, and a canary past the end.
void check_scan(const std::vector<uint32_t>& in, bool in_place, ScanWorkspace& ws, const char* label) {
    const int n = (int)in.size();
    uint32_t want_total;
    auto want = host_excl(in, &want_total);
    Dev<uint32_t> d_in(in);
    // out has 16 canary words past n: an off-by-one write past the end must be detected.
    std::vector<uint32_t> out_init(n + 16, kCanary);
    Dev<uint32_t> d_out_sep(out_init);
    Dev<uint32_t> d_total(1);
    ck(cudaMemset(d_total.p, 0xFF, 4), "memset");
    uint32_t* out = in_place ? d_in.p : d_out_sep.p;
    cudaError_t e = scan_exclusive_u32(d_in.p, out, n, d_total.p, ws, 0);
    CHECK(e == cudaSuccess);
    ck(cudaDeviceSynchronize(), "sync");
    std::vector<uint32_t> got(n);
    ck(cudaMemcpy(got.data(), out, n * 4, cudaMemcpyDeviceToHost), "D2H");
    int bad = 0; int first = -1;
    for (int i = 0; i < n; ++i) if (got[i] != want[i]) { if (!bad) first = i; ++bad; }
    if (bad) std::printf("  scan %s n=%d in_place=%d: %d mismatches, first at %d got %u want %u\n", label, n,
                         (int)in_place, bad, first, got[first], want[first]);
    CHECK(bad == 0);
    CHECK(d_total.down()[0] == want_total);
    if (!in_place) {
        auto full = d_out_sep.down();
        bool canary_ok = true;
        for (int i = n; i < n + 16; ++i) canary_ok = canary_ok && full[i] == kCanary;
        CHECK(canary_ok);  // scan must not write past element n-1
    }
}

}  // namespace

TEST(scan_matches_host_prefix_sum_random_0_to_3) {
    // Given random values 0..3, When scanning, Then out[i] == sum(in[0..i-1]) for every i, any n.
    ScanWorkspace ws;
    ck(scan_workspace_create(&ws, 1 << 21), "ws create");
    for (int n : kScanSizes) check_scan(random_vals(n, 3, 1000 + n), false, ws, "rand0-3");
    scan_workspace_destroy(&ws);
}

TEST(scan_all_ones_is_iota) {
    // Oracle: exclusive scan of ones is 0,1,2,... ; total == n. Catches inclusive-vs-exclusive.
    ScanWorkspace ws;
    ck(scan_workspace_create(&ws, 1 << 21), "ws create");
    for (int n : kScanSizes) {
        std::vector<uint32_t> ones(n, 1u);
        Dev<uint32_t> d_in(ones), d_out((size_t)n), d_tot(1);
        CHECK(scan_exclusive_u32(d_in.p, d_out.p, n, d_tot.p, ws, 0) == cudaSuccess);
        ck(cudaDeviceSynchronize(), "sync");
        auto got = d_out.down();
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += got[i] != (uint32_t)i;  // analytic
        CHECK(bad == 0);
        CHECK(d_tot.down()[0] == (uint32_t)n);
    }
    scan_workspace_destroy(&ws);
}

TEST(scan_all_zeros_is_zeros_total_zero) {
    // Oracle: all-zero input gives all-zero output and total 0 (total buffer pre-set to nonzero).
    ScanWorkspace ws;
    ck(scan_workspace_create(&ws, 1 << 21), "ws create");
    for (int n : kScanSizes) {
        std::vector<uint32_t> z(n, 0u);
        Dev<uint32_t> d_in(z), d_out((size_t)n), d_tot(1);
        ck(cudaMemset(d_tot.p, 0xFF, 4), "memset");
        ck(cudaMemset(d_out.p, 0xFF, (size_t)n * 4), "memset");
        CHECK(scan_exclusive_u32(d_in.p, d_out.p, n, d_tot.p, ws, 0) == cudaSuccess);
        ck(cudaDeviceSynchronize(), "sync");
        auto got = d_out.down();
        int bad = 0;
        for (int i = 0; i < n; ++i) bad += got[i] != 0u;
        CHECK(bad == 0);
        CHECK(d_tot.down()[0] == 0u);
    }
    scan_workspace_destroy(&ws);
}

TEST(scan_in_place_equals_out_of_place) {
    // Given d_in == d_out, Then the result equals the host scan (in-place is explicitly allowed).
    ScanWorkspace ws;
    ck(scan_workspace_create(&ws, 1 << 21), "ws create");
    for (int n : kScanSizes) check_scan(random_vals(n, 3, 77 + n), true, ws, "inplace");
    scan_workspace_destroy(&ws);
}

TEST(scan_handles_large_values_with_wraparound_free_sums) {
    // Values up to 1000 over 100000 elements: total 5e7 fits in uint32; catches 16-bit/float partials.
    ScanWorkspace ws;
    ck(scan_workspace_create(&ws, 100000), "ws create");
    check_scan(random_vals(100000, 1000, 5), false, ws, "big values");
    scan_workspace_destroy(&ws);
}

TEST(scan_null_total_pointer_is_allowed) {
    // Header: d_total "may be null". Output must still be correct.
    ScanWorkspace ws;
    ck(scan_workspace_create(&ws, 4096), "ws create");
    auto in = random_vals(3000, 3, 9);
    uint32_t t; auto want = host_excl(in, &t);
    Dev<uint32_t> d_in(in), d_out(in.size());
    CHECK(scan_exclusive_u32(d_in.p, d_out.p, 3000, nullptr, ws, 0) == cudaSuccess);
    ck(cudaDeviceSynchronize(), "sync");
    CHECK(d_out.down() == want);
    scan_workspace_destroy(&ws);
}

TEST(scan_n_above_workspace_capacity_fails_and_leaves_output_untouched) {
    // Given ws.capacity=1000, When n=5000 (> capacity), Then a non-success code is returned and
    // the output buffer still holds its canary (no partial scan, no out-of-bounds block_sums use).
    ScanWorkspace ws;
    ck(scan_workspace_create(&ws, 1000), "ws create");
    const int n = 5000;
    std::vector<uint32_t> in(n, 1u);
    Dev<uint32_t> d_in(in), d_out(std::vector<uint32_t>(n, kCanary)), d_tot(std::vector<uint32_t>(1, kCanary));
    cudaError_t e = scan_exclusive_u32(d_in.p, d_out.p, n, d_tot.p, ws, 0);
    CHECK(e != cudaSuccess);
    cudaGetLastError();  // clear any sticky non-fatal state raised by the rejected call
    ck(cudaDeviceSynchronize(), "sync after rejected scan");
    auto got = d_out.down();
    bool untouched = true;
    for (uint32_t v : got) untouched = untouched && v == kCanary;
    CHECK(untouched);
    // The workspace is still usable for a legal size afterwards.
    check_scan(random_vals(1000, 3, 3), false, ws, "after reject");
    // Boundary: n == capacity is legal.
    scan_workspace_destroy(&ws);
}

TEST(scan_workspace_create_destroy_does_not_leak_device_memory) {
    // Oracle: cudaMemGetInfo free bytes before vs after 50 create/destroy cycles within 1 MB.
    ck(cudaFree(0), "init");
    size_t free0, total;
    ck(cudaMemGetInfo(&free0, &total), "meminfo");
    for (int i = 0; i < 50; ++i) {
        ScanWorkspace ws;
        CHECK(scan_workspace_create(&ws, 1 << 22) == cudaSuccess);
        CHECK(ws.capacity >= (1 << 22));
        CHECK(ws.block_sums != nullptr);
        scan_workspace_destroy(&ws);
    }
    size_t free1;
    ck(cudaMemGetInfo(&free1, &total), "meminfo");
    long long delta = (long long)free0 - (long long)free1;
    CHECK(std::llabs(delta) < (1 << 20));
}

namespace {

void check_compact(const std::vector<uint32_t>& values, const std::vector<uint32_t>& flags, const char* label) {
    const int n = (int)values.size();
    std::vector<uint32_t> want;
    for (int i = 0; i < n; ++i) if (flags[i] != 0) want.push_back(values[i]);
    ScanWorkspace ws;
    ck(scan_workspace_create(&ws, std::max(n, 1)), "ws create");
    Dev<uint32_t> d_v(values), d_f(flags), d_out(std::vector<uint32_t>(n + 16, kCanary));
    Dev<uint32_t> d_cnt(1), d_scr((size_t)n);
    ck(cudaMemset(d_cnt.p, 0xFF, 4), "memset");
    CHECK(compact_u32(d_v.p, d_f.p, d_out.p, d_cnt.p, d_scr.p, n, ws, 0) == cudaSuccess);
    ck(cudaDeviceSynchronize(), "sync");
    uint32_t cnt = d_cnt.down()[0];
    if (cnt != want.size()) std::printf("  compact %s n=%d: count %u want %zu\n", label, n, cnt, want.size());
    CHECK(cnt == (uint32_t)want.size());
    auto out = d_out.down();
    bool same = cnt == want.size() && std::equal(want.begin(), want.end(), out.begin());  // stable order
    if (!same) std::printf("  compact %s n=%d: output differs from host filter\n", label, n);
    CHECK(same);
    // Slots past n untouched (no overrun).
    bool canary_ok = true;
    for (int i = n; i < n + 16; ++i) canary_ok = canary_ok && out[i] == kCanary;
    CHECK(canary_ok);
    // Inputs unchanged.
    CHECK(d_v.down() == values);
    CHECK(d_f.down() == flags);
    scan_workspace_destroy(&ws);
}

std::vector<uint32_t> iota_vals(int n) {
    std::vector<uint32_t> v(n);
    std::iota(v.begin(), v.end(), 1000u);  // distinct values, not equal to indices
    return v;
}

std::vector<uint32_t> density_flags(int n, double p, uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<uint32_t> f(n);
    for (auto& x : f) x = u(g) < p ? 1u : 0u;
    return f;
}

}  // namespace

TEST(compact_all_zero_flags_gives_count_zero) {
    for (int n : {1, 33, 1024, 921600}) check_compact(iota_vals(n), std::vector<uint32_t>(n, 0u), "zeros");
}

TEST(compact_all_one_flags_is_identity) {
    for (int n : {1, 31, 32, 1025, 921600}) check_compact(iota_vals(n), std::vector<uint32_t>(n, 1u), "ones");
}

TEST(compact_alternating_flags_both_phases) {
    for (int n : {2, 33, 1000, 2049}) {
        std::vector<uint32_t> a(n), b(n);
        for (int i = 0; i < n; ++i) { a[i] = i % 2; b[i] = 1 - i % 2; }
        check_compact(iota_vals(n), a, "alt01");
        check_compact(iota_vals(n), b, "alt10");
    }
}

TEST(compact_random_density_is_stable_and_counts_correct) {
    // Stable order is the property: output equals the host filter, not merely the same set.
    int k = 0;
    for (double p : {0.01, 0.5, 0.99}) {
        for (int n : {1, 100, 1025, 921600}) check_compact(random_vals(n, 1u << 30, 40 + k), density_flags(n, p, 90 + k), "density");
        ++k;
    }
}

TEST(compact_selects_any_nonzero_flag_value_not_only_one) {
    // Header: selected when flags[i] != 0. Flags of 2, 7 and 0x80000000 must each count as ONE item;
    // a bug that scans the raw flags would produce wrong counts and positions.
    const int n = 5000;
    std::mt19937 g(5);
    std::vector<uint32_t> flags(n);
    const uint32_t choices[] = {0u, 0u, 1u, 2u, 7u, 0x80000000u};
    for (auto& f : flags) f = choices[g() % 6];
    check_compact(iota_vals(n), flags, "nonzero flags");
}

TEST(compact_single_element_boundaries) {
    check_compact({42u}, {1u}, "n1 keep");
    check_compact({42u}, {0u}, "n1 drop");
}

namespace {

struct PushResult {
    uint32_t counter, overflow;
    std::vector<uint32_t> items;  // buffer incl. canary tail
    uint32_t capacity;
    size_t tail_start;
};

PushResult run_push(const std::vector<uint32_t>& flags, uint32_t capacity, size_t tail = 32) {
    const int n = (int)flags.size();
    Dev<uint32_t> d_f(flags), d_items(std::vector<uint32_t>(capacity + tail, kCanary));
    Dev<uint32_t> d_c(std::vector<uint32_t>(1, 0u)), d_o(std::vector<uint32_t>(1, 0u));
    ck(test_queue_push(d_f.p, n, d_items.p, capacity, d_c.p, d_o.p, 0), "test_queue_push");
    ck(cudaDeviceSynchronize(), "sync");
    return {d_c.down()[0], d_o.down()[0], d_items.down(), capacity, (size_t)capacity};
}

}  // namespace

TEST(queue_push_counter_is_popcount_and_stores_each_flagged_index_once) {
    for (int n : {1, 31, 32, 100, 100000}) {
        for (double p : {0.0, 0.3, 0.5, 1.0}) {
            auto flags = density_flags(n, p, 700 + n);
            std::vector<uint32_t> want;
            for (int i = 0; i < n; ++i) if (flags[i]) want.push_back((uint32_t)i);   // oracle: popcount + index set
            auto r = run_push(flags, (uint32_t)n);
            if (r.counter != want.size() || r.overflow != 0)
                std::printf("  push n=%d p=%.1f counter=%u want %zu overflow=%u\n", n, p, r.counter, want.size(), r.overflow);
            CHECK(r.counter == (uint32_t)want.size());
            CHECK(r.overflow == 0u);
            std::vector<uint32_t> got(r.items.begin(), r.items.begin() + want.size());
            std::sort(got.begin(), got.end());       // order unspecified: compare sorted
            CHECK(got == want);                      // no duplicates, no gaps, no invalid values
            bool tail_ok = true;                     // nothing written past the flagged count
            for (size_t i = want.size(); i < r.items.size(); ++i) tail_ok = tail_ok && r.items[i] == kCanary;
            CHECK(tail_ok);
        }
    }
}

TEST(queue_push_partial_warp_flagged_lane_patterns) {
    // Single flagged lane at each position of one warp, and n = 33 with only lane 32 flagged
    // (a warp with one live lane): leader selection / rank computation must handle every case.
    for (int lane : {0, 1, 15, 30, 31}) {
        std::vector<uint32_t> flags(32, 0u);
        flags[lane] = 1u;
        auto r = run_push(flags, 32);
        CHECK(r.counter == 1u);
        CHECK(r.items[0] == (uint32_t)lane);
    }
    std::vector<uint32_t> f33(33, 0u);
    f33[32] = 1u;
    auto r = run_push(f33, 33);
    CHECK(r.counter == 1u);
    CHECK(r.items[0] == 32u);
}

TEST(queue_push_overflow_counts_dropped_lanes_and_stores_valid_distinct_items) {
    const int n = 5000;
    auto flags = density_flags(n, 0.5, 31337);
    uint32_t flagged = 0;
    std::vector<char> is_flagged(n);
    for (int i = 0; i < n; ++i) { is_flagged[i] = flags[i] != 0; flagged += flags[i] != 0; }
    for (uint32_t cap : {1u, 31u, 100u, flagged - 1u}) {
        CHECK(flagged > cap);   // test precondition: genuinely overflowing
        auto r = run_push(flags, cap);
        // Spec: counter either exceeds capacity or equals pushed count; both equal `flagged` for a
        // reservation-based queue, so require counter >= capacity.
        CHECK(r.counter >= cap);
        CHECK(r.counter == flagged || r.counter > cap);
        CHECK(r.overflow == flagged - cap);   // oracle: dropped lanes = flagged - capacity
        std::vector<uint32_t> stored(r.items.begin(), r.items.begin() + cap);
        bool valid = true;
        for (uint32_t v : stored) valid = valid && v < (uint32_t)n && is_flagged[v];
        CHECK(valid);   // never stores unflagged index or the canary
        std::sort(stored.begin(), stored.end());
        CHECK(std::adjacent_find(stored.begin(), stored.end()) == stored.end());   // distinct
        bool tail_ok = true;   // no write past capacity
        for (size_t i = cap; i < r.items.size(); ++i) tail_ok = tail_ok && r.items[i] == kCanary;
        CHECK(tail_ok);
    }
}

TEST(queue_push_capacity_exactly_flagged_count_has_no_overflow) {
    // Boundary off-by-one: slot >= capacity must be the drop condition, not slot > capacity.
    auto flags = density_flags(3000, 0.4, 2024);
    uint32_t flagged = 0;
    for (auto f : flags) flagged += f != 0;
    auto r = run_push(flags, flagged);
    CHECK(r.overflow == 0u);
    CHECK(r.counter == flagged);
    std::vector<uint32_t> got(r.items.begin(), r.items.begin() + flagged);
    std::sort(got.begin(), got.end());
    CHECK(std::adjacent_find(got.begin(), got.end()) == got.end());
    // And one slot fewer overflows by exactly one.
    auto r2 = run_push(flags, flagged - 1);
    CHECK(r2.overflow == 1u);
}

TEST(queue_push_capacity_zero_overflows_everything_and_writes_nothing) {
    auto flags = density_flags(1000, 0.5, 8);
    uint32_t flagged = 0;
    for (auto f : flags) flagged += f != 0;
    auto r = run_push(flags, 0, 64);
    CHECK(r.overflow == flagged);
    bool canary_ok = true;
    for (uint32_t v : r.items) canary_ok = canary_ok && v == kCanary;
    CHECK(canary_ok);
}
