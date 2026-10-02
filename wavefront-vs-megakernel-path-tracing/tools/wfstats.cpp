// wfstats: collects the measurements behind the analysis figures and writes one JSON file.
//   wfstats <out.json> [section ...]     sections: convergence furnace variance survival divergence scaling bvh rng
// Every number comes from running the renderers, the BVH builder or the RNG in this repository.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "wf/bvh.hpp"
#include "wf/path_core.hpp"
#include "wf/renderer.hpp"
#include "wf/rng.hpp"
#include "wf/scene.hpp"

namespace {
using namespace wf;

std::string num(double v) { char b[48]; std::snprintf(b, sizeof b, "%.9g", std::isfinite(v) ? v : 0.0); return b; }
std::string arr(const std::vector<double>& v) { std::string s = "["; for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + num(v[i]); return s + "]"; }
std::string q(const std::string& s) { return "\"" + s + "\""; }

double lum(const float* c) { return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; }
double mean_of(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size(); }
double sd_of(const std::vector<double>& v) { if (v.size() < 2) return 0; double m = mean_of(v), s = 0; for (double x : v) s += (x - m) * (x - m); return std::sqrt(s / (v.size() - 1)); }
double median_of(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

struct Fit { double a = 0, b = 0, r2 = 0, se_b = 0; };   // y = a + b x
Fit ols(const std::vector<double>& x, const std::vector<double>& y) {
    const size_t n = x.size(); double mx = mean_of(x), my = mean_of(y), sxx = 0, sxy = 0, syy = 0;
    for (size_t i = 0; i < n; ++i) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); syy += (y[i] - my) * (y[i] - my); }
    Fit f; f.b = sxy / sxx; f.a = my - f.b * mx; f.r2 = syy > 0 ? sxy * sxy / (sxx * syy) : 1;
    double rss = 0; for (size_t i = 0; i < n; ++i) { double e = y[i] - f.a - f.b * x[i]; rss += e * e; }
    f.se_b = n > 2 ? std::sqrt(rss / (n - 2) / sxx) : 0;
    return f;
}
std::string fit_json(const Fit& f) { return "{\"a\":" + num(f.a) + ",\"b\":" + num(f.b) + ",\"r2\":" + num(f.r2) + ",\"se_b\":" + num(f.se_b) + "}"; }

RenderParams params(int spp, uint32_t seed, int depth = 8) { RenderParams p; p.spp = spp; p.seed = seed; p.max_depth = depth; return p; }

// ---- 1. convergence of the estimator against a high-sample reference ----
std::string convergence() {
    Scene sc = make_scene("cornell", 160, 90);
    GpuRenderer gpu(sc, GpuMode::Megakernel);
    RenderStats st;
    RenderResult ref = gpu.render(params(16384, 99991), &st);
    const size_t n = (size_t)ref.width * ref.height;
    std::vector<double> ref_l(n); double ref_var = 0;
    for (size_t i = 0; i < n; ++i) { ref_l[i] = lum(&ref.rgb[3 * i]); ref_var += ref.lum_var[i]; }
    const double floor_mse = ref_var / n / 16384.0;
    auto one = [&](RenderResult& r, double* mse, double* pred) {
        double m = 0, v = 0;
        for (size_t i = 0; i < n; ++i) { double d = lum(&r.rgb[3 * i]) - ref_l[i]; m += d * d; v += r.lum_var[i]; }
        *mse = m / n; *pred = v / n / r.spp;
    };
    std::vector<double> spps, rm, rs, pm, cm, cs;
    for (int spp = 1; spp <= 1024; spp *= 2) {
        std::vector<double> e, pr;
        for (int rep = 0; rep < 6; ++rep) { RenderResult r = gpu.render(params(spp, 100 + rep), &st); double m, p; one(r, &m, &p); e.push_back(std::sqrt(std::max(m - floor_mse, 0.0))); pr.push_back(std::sqrt(p)); }
        spps.push_back(spp); rm.push_back(mean_of(e)); rs.push_back(sd_of(e)); pm.push_back(mean_of(pr));
        if (spp <= 64) {
            std::vector<double> ce;
            for (int rep = 0; rep < 3; ++rep) { RenderResult r = render_cpu(sc, params(spp, 200 + rep), &st); double m, p; one(r, &m, &p); ce.push_back(std::sqrt(std::max(m - floor_mse, 0.0))); }
            cm.push_back(mean_of(ce)); cs.push_back(sd_of(ce));
        }
    }
    std::vector<double> lx, ly; for (size_t i = 0; i < spps.size(); ++i) { lx.push_back(std::log2(spps[i])); ly.push_back(std::log2(rm[i])); }
    Fit f = ols(lx, ly);
    return "\"convergence\":{\"scene\":\"cornell\",\"width\":160,\"height\":90,\"reference_spp\":16384,\"reference_floor_rmse\":" + num(std::sqrt(floor_mse)) +
           ",\"spp\":" + arr(spps) + ",\"rmse_mean\":" + arr(rm) + ",\"rmse_sd\":" + arr(rs) + ",\"predicted\":" + arr(pm) +
           ",\"cpu_rmse_mean\":" + arr(cm) + ",\"cpu_rmse_sd\":" + arr(cs) + ",\"loglog_fit\":" + fit_json(f) + "}";
}

// ---- 2. bias: furnace radiance against the geometric series ----
std::string furnace() {
    Scene sc = make_scene("furnace", 64, 64);
    const double rho = sc.materials[0].albedo.x, le = sc.materials[0].emission.x;
    GpuRenderer mk(sc, GpuMode::Megakernel), wf(sc, GpuMode::Wavefront);
    RenderStats st;
    std::vector<double> depth, cpu, cpu_se, mkv, mk_se, wfv, wf_se, a0, a1;
    auto summarize = [&](const RenderResult& r, double* m, double* se) {
        const size_t n = (size_t)r.width * r.height; double s = 0, v = 0;
        for (size_t i = 0; i < n; ++i) { s += lum(&r.rgb[3 * i]); v += r.lum_var[i]; }
        *m = s / n; *se = std::sqrt(v / r.spp) / n;
    };
    for (int d = 1; d <= 16; ++d) {
        double m, se;
        summarize(render_cpu(sc, params(64, 7, d), &st), &m, &se); cpu.push_back(m); cpu_se.push_back(se);
        summarize(mk.render(params(64, 7, d), &st), &m, &se); mkv.push_back(m); mk_se.push_back(se);
        summarize(wf.render(params(64, 7, d), &st), &m, &se); wfv.push_back(m); wf_se.push_back(se);
        double s0 = 0, s1 = 0; for (int k = 0; k < d; ++k) s0 += std::pow(rho, k); s1 = s0 + std::pow(rho, d);
        depth.push_back(d); a0.push_back(le * s0); a1.push_back(le * s1);
    }
    return "\"furnace\":{\"albedo\":" + num(rho) + ",\"emission\":" + num(le) + ",\"limit\":" + num(le / (1 - rho)) + ",\"depth\":" + arr(depth) +
           ",\"cpu\":" + arr(cpu) + ",\"cpu_se\":" + arr(cpu_se) + ",\"megakernel\":" + arr(mkv) + ",\"megakernel_se\":" + arr(mk_se) +
           ",\"wavefront\":" + arr(wfv) + ",\"wavefront_se\":" + arr(wf_se) + ",\"series_to_depth_minus_1\":" + arr(a0) + ",\"series_to_depth\":" + arr(a1) + "}";
}

// ---- 3. per-pixel variance concentration (Lorenz curve) ----
std::string variance() {
    std::string out = "\"variance\":{";
    bool first = true;
    for (const char* name : {"cornell", "divergence", "trianglehv"}) {
        Scene sc = make_scene(name, 320, 180);
        GpuRenderer g(sc, GpuMode::Megakernel); RenderStats st;
        RenderResult r = g.render(params(64, 5), &st);
        std::vector<double> v(r.lum_var.begin(), r.lum_var.end()); std::sort(v.begin(), v.end());
        const size_t n = v.size(); double tot = 0; for (double x : v) tot += x;
        std::vector<double> lor; double cum = 0; size_t idx = 0;
        for (int k = 0; k <= 100; ++k) { size_t upto = (size_t)((double)n * k / 100.0); while (idx < upto) cum += v[idx++]; lor.push_back(tot > 0 ? cum / tot : 0); }
        double gini = 0; for (int k = 0; k < 100; ++k) gini += (lor[k] + lor[k + 1]) / 2 * 0.01; gini = 1 - 2 * gini;
        double top1 = 0; for (size_t i = n - n / 100; i < n; ++i) top1 += v[i];
        double mu = tot / n, m2 = 0, m4 = 0; for (double x : v) { double d = x - mu; m2 += d * d; m4 += d * d * d * d; } m2 /= n; m4 /= n;
        out += std::string(first ? "" : ",") + q(name) + ":{\"lorenz\":" + arr(lor) + ",\"gini\":" + num(gini) + ",\"top1pct_share\":" + num(top1 / tot) +
               ",\"mean_var\":" + num(mu) + ",\"median_var\":" + num(v[n / 2]) + ",\"max_over_mean\":" + num(v.back() / mu) + ",\"kurtosis\":" + num(m2 > 0 ? m4 / (m2 * m2) : 0) + "}";
        first = false;
    }
    return out + "}";
}

// ---- 4. path survival from extension-ray counts at increasing depth caps ----
std::string survival() {
    std::string out = "\"survival\":{";
    bool first = true;
    for (const char* name : {"cornell", "divergence", "trianglehv"}) {
        Scene sc = make_scene(name, 320, 180);
        GpuRenderer g(sc, GpuMode::Megakernel);
        const double N = 320.0 * 180.0 * 8.0;
        std::vector<double> e;
        for (int d = 0; d <= 16; ++d) { RenderParams p = params(8, 3, d); p.count_rays = true; RenderStats st; g.render(p, &st); e.push_back((double)st.extension_rays); }
        out += std::string(first ? "" : ",") + q(name) + ":{\"paths\":" + num(N) + ",\"extension_rays_by_cap\":" + arr(e) + "}";
        first = false;
    }
    return out + "}";
}

// ---- 5. material mix seen by warp-sized pixel tiles (primary hits, host side) ----
std::string divergence() {
    std::string out = "\"divergence\":{";
    bool first = true;
    for (const char* name : {"cornell", "divergence", "trianglehv"}) {
        const int W = 320, H = 180;
        Scene sc = make_scene(name, W, H);
        SceneView view = sc.view();
        std::vector<int> cat((size_t)W * H);
        std::map<int, double> count;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            PathState ps = init_path(view, x, y, 0, 1234); HitRecord h;
            int c = -1;
            if (intersect(view, ps.ray, &h) && h.prim >= 0) c = view.materials[prim_material(view, h.prim)].type;
            cat[(size_t)y * W + x] = c; count[c] += 1;
        }
        const double n = (double)W * H; double H_bits = 0, iid = 0; std::vector<double> p;
        for (auto& kv : count) { double pi = kv.second / n; p.push_back(pi); H_bits -= pi * std::log2(pi); iid += 1 - std::pow(1 - pi, 32); }
        double tiles = 0, distinct = 0;
        for (int ty = 0; ty + 2 <= H; ty += 2) for (int tx = 0; tx + 16 <= W; tx += 16) {
            std::set<int> s; for (int j = 0; j < 2; ++j) for (int i = 0; i < 16; ++i) s.insert(cat[(size_t)(ty + j) * W + tx + i]);
            distinct += (double)s.size(); tiles += 1;
        }
        std::vector<double> kinds; for (auto& kv : count) kinds.push_back(kv.first);
        out += std::string(first ? "" : ",") + q(name) + ":{\"categories\":" + arr(kinds) + ",\"probability\":" + arr(p) + ",\"entropy_bits\":" + num(H_bits) +
               ",\"expected_distinct_iid\":" + num(iid) + ",\"mean_distinct_per_warp\":" + num(distinct / tiles) + "}";
        first = false;
    }
    return out + "}";
}

// ---- 7. render time against pixel count ----
std::string scaling() {
    std::string out = "\"scaling\":{";
    bool first_scene = true;
    const int widths[] = {128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1600, 1920};
    for (const char* name : {"cornell", "divergence"}) {
        std::vector<double> px, mkm, wfm;
        for (int w : widths) {
            int h = (w * 9 / 16) & ~1;
            Scene sc = make_scene(name, w, h);
            GpuRenderer a(sc, GpuMode::Megakernel), b(sc, GpuMode::Wavefront);
            RenderStats st; RenderParams p = params(4, 11);
            a.render(p, &st); b.render(p, &st);
            std::vector<double> ta, tb;
            for (int r = 0; r < 5; ++r) { a.render(p, &st); ta.push_back(st.render_ms); b.render(p, &st); tb.push_back(st.render_ms); }
            px.push_back((double)w * h); mkm.push_back(median_of(ta)); wfm.push_back(median_of(tb));
        }
        std::vector<double> mpx; for (double v : px) mpx.push_back(v / 1e6);
        Fit fm = ols(mpx, mkm), fw = ols(mpx, wfm);
        out += std::string(first_scene ? "" : ",") + q(name) + ":{\"megapixels\":" + arr(mpx) + ",\"megakernel_ms\":" + arr(mkm) + ",\"wavefront_ms\":" + arr(wfm) +
               ",\"megakernel_fit\":" + fit_json(fm) + ",\"wavefront_fit\":" + fit_json(fw) + ",\"spp\":4}";
        first_scene = false;
    }
    return out + "}";
}

// ---- 8. BVH quality ----
float area(Vec3 lo, Vec3 hi) { float dx = hi.x - lo.x, dy = hi.y - lo.y, dz = hi.z - lo.z; return 2.0f * (dx * dy + dy * dz + dz * dx); }

std::string bvh_stats() {
    std::string out = "\"bvh\":{";
    bool first = true;
    for (const char* name : {"cornell", "divergence", "trianglehv"}) {
        Scene sc = make_scene(name, 1280, 720);
        const auto& nd = sc.nodes;
        const double root = area(nd[0].lo, nd[0].hi);
        std::vector<double> leaf_depth_hist(64, 0), leaf_size(5, 0), ratio_sum(64, 0), ratio_cnt(64, 0);
        double sah = 0; int max_depth = 0, internal = 0, leaves = 0;
        struct Item { int node, depth; }; std::vector<Item> stack{{0, 0}};
        while (!stack.empty()) {
            Item it = stack.back(); stack.pop_back();
            const BvhNode& n = nd[it.node]; double a = area(n.lo, n.hi) / root;
            max_depth = std::max(max_depth, it.depth);
            if (n.prim_count > 0) { ++leaves; sah += a * n.prim_count; leaf_depth_hist[std::min(it.depth, 63)] += 1; leaf_size[std::min(n.prim_count, 4)] += 1; }
            else {
                ++internal; sah += a;
                int l = it.node + 1, r = nd[l].skip;
                for (int c : {l, r}) { ratio_sum[std::min(it.depth, 63)] += area(nd[c].lo, nd[c].hi) / std::max(area(n.lo, n.hi), 1e-20f); ratio_cnt[std::min(it.depth, 63)] += 1; stack.push_back({c, it.depth + 1}); }
            }
        }
        std::vector<double> dh(leaf_depth_hist.begin(), leaf_depth_hist.begin() + max_depth + 1), rr;
        for (int d = 0; d < max_depth; ++d) rr.push_back(ratio_cnt[d] > 0 ? ratio_sum[d] / ratio_cnt[d] : 0);
        out += std::string(first ? "" : ",") + q(name) + ":{\"primitives\":" + num(sc.n_prims()) + ",\"nodes\":" + num((double)nd.size()) + ",\"internal\":" + num(internal) + ",\"leaves\":" + num(leaves) +
               ",\"max_depth\":" + num(max_depth) + ",\"sah_cost\":" + num(sah) + ",\"sah_per_primitive\":" + num(sah / sc.n_prims()) + ",\"leaf_depth_hist\":" + arr(dh) +
               ",\"leaf_size_hist\":" + arr(std::vector<double>(leaf_size.begin() + 1, leaf_size.end())) + ",\"child_to_parent_area_by_depth\":" + arr(rr) + "}";
        first = false;
    }
    return out + "}";
}

// ---- 9. counter-based RNG quality ----
std::string rng_stats() {
    // Avalanche of make_path_key over its 96 input bits (seed, pixel, sample) onto its 64 output bits.
    const int trials = 20000; uint64_t s = 88172645463325252ull;
    auto next = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    std::vector<double> flip(96 * 64, 0.0);
    for (int t = 0; t < trials; ++t) {
        uint32_t in[3] = {(uint32_t)next(), (uint32_t)next(), (uint32_t)next()};
        PathKey base = make_path_key(in[0], in[1], in[2]);
        uint64_t b0 = ((uint64_t)base.k1 << 32) | base.k0;
        for (int bit = 0; bit < 96; ++bit) {
            uint32_t m[3] = {in[0], in[1], in[2]}; m[bit / 32] ^= 1u << (bit % 32);
            PathKey k = make_path_key(m[0], m[1], m[2]); uint64_t d = (((uint64_t)k.k1 << 32) | k.k0) ^ b0;
            for (int ob = 0; ob < 64; ++ob) flip[bit * 64 + ob] += (double)((d >> ob) & 1);
        }
    }
    double maxdev = 0, mean = 0; for (double& f : flip) { f /= trials; mean += f / flip.size(); maxdev = std::max(maxdev, std::fabs(f - 0.5)); }
    const double se = 0.5 / std::sqrt((double)trials);
    // Stream tests on u_i = rng(key(seed, pixel=i, sample 0), depth 0, kDimBsdfU)
    const int N = 1 << 20; std::vector<float> u(N);
    for (int i = 0; i < N; ++i) u[i] = rng(make_path_key(1234, (uint32_t)i, 0), 0, kDimBsdfU);
    double mu = 0; for (float x : u) mu += x; mu /= N; double var = 0; for (float x : u) var += (x - mu) * (x - mu); var /= N;
    std::vector<double> acf;
    for (int lag = 1; lag <= 64; ++lag) { double c = 0; for (int i = 0; i + lag < N; ++i) c += (u[i] - mu) * (u[i + lag] - mu); acf.push_back(c / (N - lag) / var); }
    std::vector<float> sorted(u); std::sort(sorted.begin(), sorted.end()); double D = 0;
    for (int i = 0; i < N; i += 16) D = std::max({D, std::fabs((i + 1.0) / N - sorted[i]), std::fabs((double)i / N - sorted[i])});
    // 16x16 serial test on consecutive pixels (u_i, u_{i+1}), repeated over 32 seeds to see whether one value is chance.
    const int B = 16; std::vector<double> chi2s;
    for (uint32_t seed = 1; seed <= 32; ++seed) {
        std::vector<double> cell(B * B, 0.0); float prev = rng(make_path_key(seed, 0, 0), 0, kDimBsdfU);
        for (int i = 1; i < N; ++i) { float cur = rng(make_path_key(seed, (uint32_t)i, 0), 0, kDimBsdfU); cell[std::min(B - 1, (int)(prev * B)) * B + std::min(B - 1, (int)(cur * B))] += 1; prev = cur; }
        double c2 = 0, expect = (double)(N - 1) / (B * B); for (double c : cell) c2 += (c - expect) * (c - expect) / expect;
        chi2s.push_back(c2);
    }
    double chi2 = chi2s[0];
    return "\"rng\":{\"trials\":" + num(trials) + ",\"avalanche\":" + arr(flip) + ",\"avalanche_mean\":" + num(mean) + ",\"avalanche_max_dev\":" + num(maxdev) +
           ",\"avalanche_se\":" + num(se) + ",\"stream_samples\":" + num(N) + ",\"acf\":" + arr(acf) + ",\"acf_band95\":" + num(1.96 / std::sqrt((double)N)) +
           ",\"ks_d\":" + num(D) + ",\"ks_critical_5pct\":" + num(1.358 / std::sqrt((double)N)) + ",\"pair_chi2\":" + num(chi2) + ",\"pair_chi2_by_seed\":" + arr(chi2s) + ",\"pair_dof\":" + num(B * B - 1) + "}";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) { std::cerr << "Usage: wfstats <out.json> [section ...]\n"; return 2; }
        std::set<std::string> want; for (int i = 2; i < argc; ++i) want.insert(argv[i]);
        auto on = [&](const char* n) { return want.empty() || want.count(n); };
        std::vector<std::string> parts; const auto t0 = std::chrono::steady_clock::now();
        auto run = [&](const char* n, std::string (*f)()) { if (!on(n)) return; parts.push_back(f()); std::cerr << n << " done (" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s)\n"; };
        run("convergence", convergence); run("furnace", furnace); run("variance", variance); run("survival", survival);
        run("divergence", divergence); run("scaling", scaling); run("bvh", bvh_stats); run("rng", rng_stats);
        CudaDeviceFacts dev = query_cuda_device();
        std::string json = "{\"schema_version\":1,\"device\":" + q(dev.name) + ",";
        for (size_t i = 0; i < parts.size(); ++i) json += (i ? "," : "") + parts[i];
        json += "}\n";
        std::ofstream f(argv[1]); if (!f) throw std::runtime_error(std::string("cannot write ") + argv[1]);
        f << json; std::cout << "wrote " << argv[1] << " (" << json.size() << " bytes)\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 2; }
}
