// GPU queue primitives, boundaries and failure (building blocks for the wavefront arm).
// Oracles: host exclusive prefix sum, host stable filter, host popcount of the flags, set equality
// of pushed indices with the flagged index set, canary words to prove no out-of-bounds writes,
// and cudaMemGetInfo deltas for leaks. All independent of the device code.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "queues.cuh"
#include "wf/renderer.hpp"
#include "wf/scene.hpp"

using namespace wf::gpu;

namespace {

constexpr uint32_t kCanary = 0xDEADBEEFu;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { ck(cudaMalloc(&p, (count ? count : 1) * sizeof(T)), "cudaMalloc"); }
    explicit Dev(const std::vector<T>& h) : Dev(h.size()) { up(h); }
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

struct Workspace {
    ScanWorkspace ws;
    explicit Workspace(int capacity) { ck(scan_workspace_create(&ws, capacity), "scan_workspace_create"); }
    ~Workspace() { scan_workspace_destroy(&ws); }
};

// Independent reference: exclusive prefix sum with uint32 wraparound.
std::vector<uint32_t> host_excl(const std::vector<uint32_t>& in, uint32_t* total) {
    std::vector<uint32_t> out(in.size());
    uint32_t acc = 0;
    for (size_t i = 0; i < in.size(); ++i) { out[i] = acc; acc += in[i]; }
    *total = acc;
    return out;
}

enum class Fill { Random, Ones, Zero };

std::vector<uint32_t> make_input(int n, Fill f, uint32_t seed) {
    std::mt19937 g(seed);
    std::vector<uint32_t> v(n);
    for (auto& x : v) x = f == Fill::Random ? g() % 1000u : (f == Fill::Ones ? 1u : 0u);
    return v;
}

std::vector<uint32_t> random_flags(int n, double density, uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    // Non-zero flag values other than 1: the contract is "flags[i] != 0", not "== 1".
    const uint32_t nz[] = {1u, 2u, 7u, 0x80000000u, 0xFFFFFFFFu};
    std::vector<uint32_t> f(n, 0u);
    for (auto& x : f) if (u(g) < density) x = nz[g() % 5];
    return f;
}

}  // namespace

TEST_CASE("scan_exclusive_u32 equals the host prefix sum across block-boundary sizes", "[gpu_primitives]") {
    // Given n around warp (32) and block (1024) boundaries and large sizes, with random, all-ones
    // and all-zero inputs, then out[i] == sum(in[0..i-1]), *d_total == sum(in), and out[n] (one
    // word past the end) is untouched.
    const int sizes[] = {1, 2, 31, 32, 33, 1023, 1024, 1025, 921600, 1000003};
    for (int n : sizes) {
        for (Fill f : {Fill::Random, Fill::Ones, Fill::Zero}) {
            INFO("n=" << n << " fill=" << (int)f);
            std::vector<uint32_t> in = make_input(n, f, 1234u + n);
            uint32_t want_total = 0;
            std::vector<uint32_t> want = host_excl(in, &want_total);
            Dev<uint32_t> d_in(in), d_out(std::vector<uint32_t>(n + 1, kCanary)), d_total(std::vector<uint32_t>(1, 0xFFFFFFFFu));
            Workspace w(n);
            REQUIRE(scan_exclusive_u32(d_in.p, d_out.p, n, d_total.p, w.ws, 0) == cudaSuccess);
            REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
            std::vector<uint32_t> got = d_out.down();
            REQUIRE(got[n] == kCanary);
            got.resize(n);
            CHECK(got == want);
            CHECK(d_total.down()[0] == want_total);
        }
    }
}

TEST_CASE("in-place scan, null total, and n above capacity is rejected without side effects", "[gpu_primitives]") {
    // (a) in == out gives the same result as the out-of-place host oracle and the right total;
    // (b) d_total == nullptr is accepted; (c) n > ws.capacity returns an error, leaves the output
    // buffer untouched, and the workspace still works afterwards.
    for (int n : {1, 1025, 921600}) {
        INFO("in-place n=" << n);
        std::vector<uint32_t> in = make_input(n, Fill::Random, 99u + n);
        uint32_t want_total = 0;
        std::vector<uint32_t> want = host_excl(in, &want_total);
        Dev<uint32_t> d(in), d_total(std::vector<uint32_t>(1, 0xFFFFFFFFu));
        Workspace w(n);
        REQUIRE(scan_exclusive_u32(d.p, d.p, n, d_total.p, w.ws, 0) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        CHECK(d.down() == want);
        CHECK(d_total.down()[0] == want_total);
    }
    {
        std::vector<uint32_t> in = make_input(2048, Fill::Ones, 5);
        uint32_t t = 0;
        std::vector<uint32_t> want = host_excl(in, &t);
        Dev<uint32_t> d_in(in), d_out(2048);
        Workspace w(2048);
        REQUIRE(scan_exclusive_u32(d_in.p, d_out.p, 2048, nullptr, w.ws, 0) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        CHECK(d_out.down() == want);
    }
    {
        const int cap = 1000, n = 1001;   // exactly one over capacity
        std::vector<uint32_t> in = make_input(n, Fill::Ones, 6);
        Dev<uint32_t> d_in(in), d_out(std::vector<uint32_t>(n, kCanary)), d_total(std::vector<uint32_t>(1, 0xFFFFFFFFu));
        Workspace w(cap);
        cudaError_t e = scan_exclusive_u32(d_in.p, d_out.p, n, d_total.p, w.ws, 0);
        cudaGetLastError();   // clear any sticky non-fatal error so later cases start clean
        cudaStreamSynchronize(0);
        CHECK(e != cudaSuccess);
        CHECK(d_out.down() == std::vector<uint32_t>(n, kCanary));     // rejected: nothing written
        CHECK(d_total.down()[0] == 0xFFFFFFFFu);
        // n == capacity is legal on the same workspace (boundary is strict '>').
        REQUIRE(scan_exclusive_u32(d_in.p, d_out.p, cap, d_total.p, w.ws, 0) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        CHECK(d_total.down()[0] == (uint32_t)cap);
    }
}

TEST_CASE("compact_u32 is stable, counts exactly, and writes only selected slots", "[gpu_primitives]") {
    // Given densities 0/1/50/100% and flag values other than 1, then out[0..count) equals the host
    // stable filter, *d_count == popcount(flags != 0), flags/values are unchanged, and out past
    // count keeps its canary.
    const int n = 70001;   // several blocks, not a multiple of 32
    std::vector<uint32_t> values(n);
    for (int i = 0; i < n; ++i) values[i] = (uint32_t)i * 2654435761u + 17u;   // distinct enough to expose reordering
    for (double density : {0.0, 0.01, 0.5, 1.0}) {
        INFO("density=" << density);
        std::vector<uint32_t> flags = random_flags(n, density, 4242u);
        std::vector<uint32_t> want;
        for (int i = 0; i < n; ++i) if (flags[i] != 0) want.push_back(values[i]);
        Dev<uint32_t> d_vals(values), d_flags(flags), d_out(std::vector<uint32_t>(n + 1, kCanary)),
            d_count(std::vector<uint32_t>(1, 777u)), d_scratch(n);
        Workspace w(n);
        REQUIRE(compact_u32(d_vals.p, d_flags.p, d_out.p, d_count.p, d_scratch.p, n, w.ws, 0) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        const uint32_t count = d_count.down()[0];
        REQUIRE(count == (uint32_t)want.size());
        std::vector<uint32_t> out = d_out.down();
        CHECK(std::vector<uint32_t>(out.begin(), out.begin() + count) == want);
        bool tail_ok = true;
        for (size_t i = count; i < out.size(); ++i) tail_ok = tail_ok && out[i] == kCanary;
        CHECK(tail_ok);
        CHECK(d_vals.down() == values);
        CHECK(d_flags.down() == flags);
    }
}

TEST_CASE("queue_push gives every flagged lane a unique slot with no gaps", "[gpu_primitives]") {
    // Given capacity >= flagged, then counter == popcount, overflow == 0, and the first `counter`
    // items are exactly the flagged lane indices (set equality, so unique and no gap), for sizes
    // that are not warp multiples and for sparse, dense and full flags.
    struct Case { int n; double density; };
    for (Case c : {Case{100003, 0.5}, Case{100003, 0.01}, Case{4099, 1.0}, Case{33, 0.5}, Case{1, 1.0}}) {
        INFO("n=" << c.n << " density=" << c.density);
        std::vector<uint32_t> flags = random_flags(c.n, c.density, 777u + c.n);
        std::vector<uint32_t> want;
        for (int i = 0; i < c.n; ++i) if (flags[i] != 0) want.push_back((uint32_t)i);
        const uint32_t cap = (uint32_t)c.n;
        Dev<uint32_t> d_flags(flags), d_items(std::vector<uint32_t>(cap + 8, kCanary)),
            d_counter(std::vector<uint32_t>(1, 0u)), d_overflow(std::vector<uint32_t>(1, 0u));
        REQUIRE(test_queue_push(d_flags.p, c.n, d_items.p, cap, d_counter.p, d_overflow.p, 0) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        CHECK(d_counter.down()[0] == (uint32_t)want.size());
        CHECK(d_overflow.down()[0] == 0u);
        std::vector<uint32_t> items = d_items.down();
        std::vector<uint32_t> got(items.begin(), items.begin() + want.size());
        std::sort(got.begin(), got.end());   // order inside the queue is unspecified
        CHECK(got == want);
        bool tail_ok = true;
        for (size_t i = want.size(); i < items.size(); ++i) tail_ok = tail_ok && items[i] == kCanary;
        CHECK(tail_ok);
    }
}

TEST_CASE("queue_push overflow is counted and never writes past capacity", "[gpu_primitives]") {
    // Given F flagged lanes and capacity < F, then overflow == F - capacity exactly, the queue
    // holds `capacity` distinct valid flagged indices, and words past capacity keep their canary.
    // Capacity 0 drops everything: overflow == F and nothing is written.
    struct Case { int n; double density; uint32_t cap; };
    for (Case c : {Case{1000, 1.0, 300u}, Case{5000, 0.3, 100u}, Case{1000, 1.0, 999u}, Case{1000, 1.0, 0u}, Case{70, 0.5, 0u}}) {
        INFO("n=" << c.n << " density=" << c.density << " cap=" << c.cap);
        std::vector<uint32_t> flags = random_flags(c.n, c.density, 31u + c.n);
        std::vector<uint32_t> flagged;
        for (int i = 0; i < c.n; ++i) if (flags[i] != 0) flagged.push_back((uint32_t)i);
        const uint32_t F = (uint32_t)flagged.size();
        REQUIRE(F > c.cap);   // scenario really overflows
        const size_t alloc = c.cap + 64;   // 64 canary words past capacity
        Dev<uint32_t> d_flags(flags), d_items(std::vector<uint32_t>(alloc, kCanary)),
            d_counter(std::vector<uint32_t>(1, 0u)), d_overflow(std::vector<uint32_t>(1, 0u));
        REQUIRE(test_queue_push(d_flags.p, c.n, d_items.p, c.cap, d_counter.p, d_overflow.p, 0) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(0) == cudaSuccess);
        CHECK(d_overflow.down()[0] == F - c.cap);
        CHECK(d_counter.down()[0] >= c.cap);   // reservations are counted (>= capacity); exact value not pinned
        std::vector<uint32_t> items = d_items.down();
        bool tail_ok = true;
        for (size_t i = c.cap; i < alloc; ++i) tail_ok = tail_ok && items[i] == kCanary;
        CHECK(tail_ok);   // no write past capacity
        std::vector<uint32_t> stored(items.begin(), items.begin() + c.cap);
        std::sort(stored.begin(), stored.end());
        CHECK(std::adjacent_find(stored.begin(), stored.end()) == stored.end());   // distinct
        bool valid = true;   // every stored item is a flagged lane index (and not the canary)
        for (uint32_t v : stored) valid = valid && std::binary_search(flagged.begin(), flagged.end(), v);
        CHECK(valid);
    }
}

TEST_CASE("GpuRenderer constructor throws std::runtime_error on an unfinalized scene", "[gpu_primitives]") {
    // Given a valid scene whose BVH nodes were removed (the single defect), the constructor must
    // throw std::runtime_error; the same scene untouched constructs fine (control, so the throw
    // cannot come from anything else).
    wf::Scene good = wf::make_scene("cornell", 16, 16);
    REQUIRE(!good.nodes.empty());
    REQUIRE_NOTHROW(wf::GpuRenderer(good, wf::GpuMode::Megakernel));
    wf::Scene bad = good;
    bad.nodes.clear();
    REQUIRE(!bad.spheres.empty() + !bad.triangles.empty());
    REQUIRE_THROWS_AS(wf::GpuRenderer(bad, wf::GpuMode::Megakernel), std::runtime_error);
    // The failed construction left the device usable: a good renderer still renders.
    wf::GpuRenderer ok(good, wf::GpuMode::Megakernel);
    wf::RenderParams p;
    p.spp = 1;
    wf::RenderStats st;
    wf::RenderResult r = ok.render(p, &st);
    CHECK(r.rgb.size() == (size_t)3 * 16 * 16);
}

TEST_CASE("query_cuda_device reports the GTX 1650 Ti facts", "[gpu_primitives]") {
    // Target hardware: compute capability 7.5, 4 GB VRAM.
    wf::CudaDeviceFacts f = wf::query_cuda_device();
    CHECK(f.compute_capability == "7.5");
    CHECK(!f.name.empty());
    CHECK(f.total_memory_bytes > 3000000000ull);
    CHECK(f.free_memory_bytes > 0u);
    CHECK(f.free_memory_bytes <= f.total_memory_bytes);   // free memory cannot exceed total
}

TEST_CASE("scan workspace create/destroy does not leak device memory", "[gpu_primitives]") {
    // 30 create/scan/destroy cycles at capacity 1e8 (~390 KB of block sums each): a leak of the
    // block-sum array would lose ~11 MB; the tolerance is 1 MB (cudaMemGetInfo delta).
    ck(cudaFree(0), "context init");
    {   // warm-up so one-time allocator/pool growth is excluded from the measurement
        Workspace w(100000000);
    }
    size_t free_before = 0, total = 0, free_after = 0;
    ck(cudaMemGetInfo(&free_before, &total), "MemGetInfo");
    for (int i = 0; i < 30; ++i) {
        Workspace w(100000000);
        REQUIRE(w.ws.capacity >= 100000000);   // ws records the capacity it was sized for
    }
    ck(cudaDeviceSynchronize(), "sync");
    ck(cudaMemGetInfo(&free_after, &total), "MemGetInfo");
    const long long leaked = (long long)free_before - (long long)free_after;
    INFO("free before=" << free_before << " after=" << free_after << " leaked=" << leaked);
    CHECK(leaked < (1ll << 20));
}
