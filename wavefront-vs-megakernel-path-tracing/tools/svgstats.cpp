// svgstats: draws the nine analysis figures from measured data.
//   svgstats <out-dir> <stats.json> <ncu-dir>
// stats.json comes from wfstats; <ncu-dir> holds <scene>-<mode>.summary.json produced by ncusum from Nsight Compute CSVs.
// Writes fig-01-convergence.svg ... fig-09-rng.svg. Light and dark colours are CSS variables.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "wf/json.hpp"

namespace {
using wf::Json;
using Pts = std::vector<std::pair<double, double>>;

std::string read_all(const std::string& path) { std::ifstream f(path); if (!f) throw std::runtime_error("cannot read " + path); std::ostringstream s; s << f.rdbuf(); return s.str(); }
std::vector<double> vec(const Json& j) { std::vector<double> v; for (const Json& e : j.arr) v.push_back(e.num); return v; }
std::string f(double v, const char* fmt = "%.1f") { char b[64]; std::snprintf(b, sizeof b, fmt, v); return b; }
std::string esc(const std::string& s) { std::string o; for (char c : s) { if (c == '&') o += "&amp;"; else if (c == '<') o += "&lt;"; else if (c == '>') o += "&gt;"; else o += c; } return o; }
std::string col(int i) { return "var(--s" + std::to_string(i) + ")"; }
const char* kScenes[3] = {"cornell", "divergence", "trianglehv"};

struct Fit { double a, b, r2; };
Fit ols(const std::vector<double>& x, const std::vector<double>& y) {
    double n = (double)x.size(), mx = 0, my = 0; for (size_t i = 0; i < x.size(); ++i) { mx += x[i]; my += y[i]; } mx /= n; my /= n;
    double sxx = 0, sxy = 0, syy = 0; for (size_t i = 0; i < x.size(); ++i) { sxx += (x[i] - mx) * (x[i] - mx); sxy += (x[i] - mx) * (y[i] - my); syy += (y[i] - my) * (y[i] - my); }
    return {my - sxy / sxx * mx, sxy / sxx, syy > 0 ? sxy * sxy / (sxx * syy) : 1.0};
}

const char* kStyle =
    "<style>:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100}"
    "@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}"
    "text{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;fill:var(--ink)}.sub{fill:var(--ink2)}</style>";

std::string replace_all(std::string s, const std::string& a, const std::string& b) { size_t p = 0; while ((p = s.find(a, p)) != std::string::npos) { s.replace(p, a.size(), b); p += b.size(); } return s; }
std::string pretty(std::string s) {
    for (auto& r : std::vector<std::pair<const char*, const char*>>{{"sigma^2", "\u03c3\u00b2"}, {"N^-1/2", "N^(\u2212\u00bd)"}, {"sigma", "\u03c3"}, {"rho", "\u03c1"}, {"alpha", "\u03b1"}, {"beta", "\u03b2"}, {"+/-", "\u00b1"}, {">=", "\u2265"}, {"<=", "\u2264"},
                                                                    {"sqrt(", "\u221a("}, {"-1/2", "\u2212\u00bd"}, {"R^2", "R\u00b2"}, {"sigma^2", "\u03c3\u00b2"}})
        s = replace_all(s, r.first, r.second);
    return s;
}
std::vector<std::string> wrap(const std::string& s, size_t width) {
    std::vector<std::string> out; std::istringstream in(s); std::string w, line;
    while (in >> w) { if (!line.empty() && line.size() + 1 + w.size() > width) { out.push_back(line); line.clear(); } line += (line.empty() ? "" : " ") + w; }
    if (!line.empty()) out.push_back(line); return out;
}

struct Svg {
    std::ostringstream o; int W, H; double max_y = 0;
    Svg(int w, int h, const std::string& title, const std::string& desc) : W(w), H(h) {
        o << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 " << w << " @H@\" role=\"img\" aria-labelledby=\"t d\"><title id=\"t\">" << esc(title)
          << "</title><desc id=\"d\">" << esc(desc) << "</desc>" << kStyle << "<rect width=\"" << w << "\" height=\"@H@\" fill=\"var(--bg)\"/>";
    }
    void text(double x, double y, const std::string& s, int size = 12, const char* anchor = "start", bool sub = false, bool bold = false, double rot = 0) {
        max_y = std::max(max_y, y + 10);
        o << "<text x=\"" << f(x) << "\" y=\"" << f(y) << "\" font-size=\"" << size << "\" text-anchor=\"" << anchor << "\"" << (sub ? " class=\"sub\"" : "") << (bold ? " font-weight=\"600\"" : "");
        if (rot != 0) o << " transform=\"rotate(" << f(rot) << ' ' << f(x) << ' ' << f(y) << ")\"";
        o << ">" << esc(pretty(s)) << "</text>";
    }
    void line(double x1, double y1, double x2, double y2, const std::string& stroke = "var(--grid)", double w = 1, const char* dash = nullptr) {
        o << "<line x1=\"" << f(x1) << "\" y1=\"" << f(y1) << "\" x2=\"" << f(x2) << "\" y2=\"" << f(y2) << "\" stroke=\"" << stroke << "\" stroke-width=\"" << w << "\""; if (dash) o << " stroke-dasharray=\"" << dash << "\""; o << "/>";
    }
    void rect(double x, double y, double w, double h, const std::string& fill, double rx = 0, double op = 1, const std::string& tip = "") {
        o << "<rect x=\"" << f(x, "%.2f") << "\" y=\"" << f(y, "%.2f") << "\" width=\"" << f(w, "%.2f") << "\" height=\"" << f(h, "%.2f") << "\" rx=\"" << rx << "\" fill=\"" << fill << "\" fill-opacity=\"" << op << "\"";
        if (tip.empty()) o << "/>"; else o << "><title>" << esc(tip) << "</title></rect>";
    }
    void circle(double cx, double cy, double r, const std::string& fill, const std::string& stroke = "none", const std::string& tip = "") {
        o << "<circle cx=\"" << f(cx) << "\" cy=\"" << f(cy) << "\" r=\"" << r << "\" fill=\"" << fill << "\" stroke=\"" << stroke << "\" stroke-width=\"2\"";
        if (tip.empty()) o << "/>"; else o << "><title>" << esc(tip) << "</title></circle>";
    }
    void square(double cx, double cy, double r, const std::string& stroke, const std::string& tip = "") {   // hollow square marker
        o << "<rect x=\"" << f(cx - r) << "\" y=\"" << f(cy - r) << "\" width=\"" << f(2 * r) << "\" height=\"" << f(2 * r) << "\" fill=\"var(--bg)\" stroke=\"" << stroke << "\" stroke-width=\"2\">";
        if (!tip.empty()) o << "<title>" << esc(tip) << "</title>"; o << "</rect>";
    }
    void poly(const Pts& p, const std::string& stroke, double w = 2, const char* dash = nullptr) {
        o << "<polyline fill=\"none\" stroke=\"" << stroke << "\" stroke-width=\"" << w << "\" stroke-linejoin=\"round\" stroke-linecap=\"round\""; if (dash) o << " stroke-dasharray=\"" << dash << "\"";
        o << " points=\""; for (auto& q : p) o << f(q.first, "%.2f") << ',' << f(q.second, "%.2f") << ' '; o << "\"/>";
    }
    void area(const Pts& p, const std::string& fill, double op) {
        o << "<polygon fill=\"" << fill << "\" fill-opacity=\"" << op << "\" points=\""; for (auto& q : p) o << f(q.first, "%.2f") << ',' << f(q.second, "%.2f") << ' '; o << "\"/>";
    }
    void header(const std::string& title, const std::string& sub) { text(24, 32, title, 18, "start", false, true); text(24, 53, sub, 12, "start", true); }
    void caption(double y, const std::vector<std::string>& lines) { double yy = y; for (auto& ln : lines) for (auto& w : wrap(ln, 150)) { text(24, yy, w, 11, "start", true); yy += 15; } }
    std::string str() { return replace_all(o.str() + "</svg>\n", "@H@", std::to_string((int)std::ceil(max_y + 18))); }
};

struct Ax {
    double x, y, w, h, x0, x1, y0, y1; bool lx = false, ly = false;
    double X(double v) const { double t = lx ? (std::log10(v) - std::log10(x0)) / (std::log10(x1) - std::log10(x0)) : (v - x0) / (x1 - x0); return x + t * w; }
    double Y(double v) const { double t = ly ? (std::log10(v) - std::log10(y0)) / (std::log10(y1) - std::log10(y0)) : (v - y0) / (y1 - y0); return y + h - t * h; }
};
using Ticks = std::vector<std::pair<double, std::string>>;
Ticks lin_ticks(double lo, double hi, double step, const char* fmt = "%g") { Ticks t; for (double v = std::ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) t.push_back({v, f(std::fabs(v) < 1e-12 ? 0 : v, fmt)}); return t; }
void frame(Svg& s, const Ax& a, const Ticks& xt, const Ticks& yt, const std::string& xl, const std::string& yl, bool vgrid = false) {
    for (auto& t : yt) { s.line(a.x, a.Y(t.first), a.x + a.w, a.Y(t.first)); s.text(a.x - 8, a.Y(t.first) + 4, t.second, 11, "end", true); }
    if (vgrid) for (auto& t : xt) s.line(a.X(t.first), a.y, a.X(t.first), a.y + a.h);
    for (auto& t : xt) s.text(a.X(t.first), a.y + a.h + 18, t.second, 11, "middle", true);
    s.line(a.x, a.y + a.h, a.x + a.w, a.y + a.h, "var(--ink2)", 1);
    s.text(a.x + a.w / 2, a.y + a.h + 38, xl, 12, "middle", true);
    s.text(a.x - 46, a.y + a.h / 2, yl, 12, "middle", true, false, -90);
}
void legend(Svg& s, double x, double y, const std::vector<std::pair<std::string, int>>& items, double gap = 150) {
    for (auto& it : items) { s.rect(x, y - 10, 12, 12, col(it.second), 2); s.text(x + 18, y, it.first, 12); x += gap; }
}

// ---- ncu data ----
struct Kernel { std::string name; double launches = 1; std::map<std::string, double> m; std::map<std::string, std::string> unit; };
std::vector<Kernel> load_ncu(const std::string& path) {
    Json j = wf::json_parse(read_all(path)); std::vector<Kernel> ks;
    for (const Json& k : j.at("kernels").arr) {
        Kernel x; x.name = k.at("name").str; x.name = x.name.substr(0, x.name.find('(')); x.launches = k.at("invocations").num;
        for (const Json& m : k.at("metrics").arr) if (m.at("value").type == Json::Number) { x.m[m.at("name").str] = m.at("value").num; x.unit[m.at("name").str] = m.at("unit").str; }
        ks.push_back(x);
    }
    return ks;
}
double unit_scale_time(const std::string& u) { if (u == "second" || u == "s") return 1; if (u == "msecond" || u == "ms") return 1e-3; if (u == "usecond" || u == "us") return 1e-6; return 1e-9; }
double get(const Kernel& k, const char* name, double d = 0) { auto it = k.m.find(name); return it == k.m.end() ? d : it->second; }
const char* kTime = "gpu__time_duration.sum";
double kernel_seconds(const Kernel& k) { return get(k, kTime) * unit_scale_time(k.unit.count(kTime) ? k.unit.at(kTime) : "nsecond"); }
double lane_efficiency(const std::vector<Kernel>& ks) {   // instruction-weighted active lanes per warp instruction, over 32
    double num = 0, den = 0;
    for (const Kernel& k : ks) { double w = get(k, "smsp__inst_executed.sum") * k.launches; num += w * get(k, "smsp__thread_inst_executed_per_inst_executed.ratio"); den += w; }
    return den > 0 ? num / den / 32.0 : 0;
}

void write(const std::string& dir, const std::string& name, const std::string& text) { std::ofstream o(dir + "/" + name); if (!o) throw std::runtime_error("cannot write " + dir + "/" + name); o << text; }

// ---- figure 1: convergence ----
std::string fig1(const Json& J) {
    const Json& c = J.at("convergence");
    auto spp = vec(c.at("spp")), rm = vec(c.at("rmse_mean")), rs = vec(c.at("rmse_sd")), pr = vec(c.at("predicted")), cm = vec(c.at("cpu_rmse_mean")), cs = vec(c.at("cpu_rmse_sd"));
    std::vector<double> lx, ly, lx2, ly2;
    for (size_t i = 0; i < spp.size(); ++i) { lx.push_back(std::log10(spp[i])); ly.push_back(std::log10(rm[i])); if (spp[i] >= 16) { lx2.push_back(lx.back()); ly2.push_back(ly.back()); } }
    Fit all = ols(lx, ly), hi = ols(lx2, ly2);
    Svg s(900, 600, "Monte Carlo convergence", "RMSE of the megakernel and CPU reference against a 16384-sample reference as the sample count grows.");
    s.header("Error falls as N^-1/2: fitted slope " + f(hi.b, "%.3f") + " for N >= 16 (all N: " + f(all.b, "%.3f") + ")",
             "Luminance RMSE on cornell (160x90) against a 16384-spp reference, 6 seeds per point, reference noise removed");
    Ax a{90, 110, 760, 360, 0.7, 1500, 0.0025, 0.16, true, true};
    Ticks xt, yt; for (double v : {1, 4, 16, 64, 256, 1024}) xt.push_back({v, f(v, "%.0f")}); for (double v : {0.003, 0.01, 0.03, 0.1}) yt.push_back({v, f(v, "%g")});
    frame(s, a, xt, yt, "samples per pixel N (log scale)", "RMSE of luminance (log scale)", true);
    legend(s, 90, 90, {{"megakernel (mean +/- sd)", 1}, {"CPU reference", 2}, {"predicted sqrt(mean sample variance / N)", 3}}, 230);
    Pts ideal, pred; for (double x : {1.0, 1024.0}) ideal.push_back({a.X(x), a.Y(rm[10] * std::sqrt(1024.0 / x))});
    s.poly(ideal, "var(--ink2)", 1.5, "6 4");
    for (size_t i = 1; i < spp.size(); ++i) pred.push_back({a.X(spp[i]), a.Y(pr[i])});
    s.poly(pred, col(3), 2.5);
    for (size_t i = 0; i < spp.size(); ++i) {
        s.line(a.X(spp[i]), a.Y(std::max(rm[i] - rs[i], 1e-4)), a.X(spp[i]), a.Y(rm[i] + rs[i]), "var(--ink2)", 1.5);
        s.circle(a.X(spp[i]), a.Y(rm[i]), 5, col(1), "var(--bg)", "megakernel N=" + f(spp[i], "%.0f") + ": RMSE " + f(rm[i], "%.5f") + " +/- " + f(rs[i], "%.5f"));
    }
    for (size_t i = 0; i < cm.size(); ++i) s.square(a.X(spp[i]), a.Y(cm[i]), 4.5, col(2), "CPU N=" + f(spp[i], "%.0f") + ": RMSE " + f(cm[i], "%.5f"));
    s.text(a.X(1024) - 4, a.Y(rm[10] * 1.0) - 62, "ideal slope -1/2", 11, "end", true);
    s.text(a.X(1.2), a.Y(0.0035), "RMSE = sigma / sqrt(N)", 13, "start", false, true);
    s.caption(548, {"Estimator: F_N = (1/N) sum f(X_i)/p(X_i) with independent counter-based streams, so Var[F_N] = sigma^2 / N and RMSE is proportional to N^-1/2.",
                    "Below N = 16 the curve is flatter than -1/2 (heavy-tailed samples: the sample variance is itself noisy); from N = 16 the fit is " + f(hi.b, "%.3f") + " (R^2 " + f(hi.r2, "%.4f") + ").",
                    "The predicted line uses only the renderer's own per-pixel variance estimate, so it checks the variance bookkeeping independently of the reference image."});
    return s.str();
}

// ---- figure 2: bias against the geometric series ----
std::string fig2(const Json& J) {
    const Json& c = J.at("furnace");
    auto d = vec(c.at("depth")), ser = vec(c.at("series_to_depth")), cpu = vec(c.at("cpu")), mk = vec(c.at("megakernel")), wf = vec(c.at("wavefront")), se = vec(c.at("megakernel_se"));
    double rho = c.at("albedo").num, lim = c.at("limit").num;
    Svg s(900, 640, "Furnace test", "Mean radiance of a closed Lambertian furnace against the truncated geometric series for three renderers.");
    double worst = 0; for (size_t i = 0; i < d.size(); ++i) worst = std::max({worst, std::fabs(cpu[i] - ser[i]), std::fabs(mk[i] - ser[i]), std::fabs(wf[i] - ser[i])});
    s.header("Bias stays inside the sampling error: |residual| <= " + f(worst * 1e3, "%.1f") + "e-3 at every depth",
             "Furnace scene, albedo rho = " + f(rho, "%.2f") + ", emission 1, 64x64 pixels, 64 spp; truncated sum L_D = sum_{k=0..D} rho^k, limit 1/(1-rho) = " + f(lim, "%.0f"));
    Ax a{90, 110, 760, 200, 1, 16, 1.4, 2.1};
    frame(s, a, lin_ticks(1, 16, 1, "%.0f"), lin_ticks(1.4, 2.0, 0.2, "%.1f"), "", "mean radiance L_D", true);
    s.line(a.x, a.Y(lim), a.x + a.w, a.Y(lim), "var(--ink2)", 1.5, "6 4"); s.text(a.x + a.w - 4, a.Y(lim) - 6, "limit 1/(1-rho) = 2", 11, "end", true);
    Pts sp; for (size_t i = 0; i < d.size(); ++i) sp.push_back({a.X(d[i]), a.Y(ser[i])}); s.poly(sp, "var(--ink2)", 1.5);
    for (size_t i = 0; i < d.size(); ++i) {
        s.square(a.X(d[i]) - 6, a.Y(cpu[i]), 3.5, col(2), "CPU D=" + f(d[i], "%.0f") + ": " + f(cpu[i], "%.5f"));
        s.circle(a.X(d[i]), a.Y(mk[i]), 3.8, col(1), "var(--bg)", "megakernel D=" + f(d[i], "%.0f") + ": " + f(mk[i], "%.5f"));
        s.circle(a.X(d[i]) + 6, a.Y(wf[i]), 3.8, col(3), "var(--bg)", "wavefront D=" + f(d[i], "%.0f") + ": " + f(wf[i], "%.5f"));
    }
    legend(s, 90, 90, {{"CPU reference", 2}, {"megakernel", 1}, {"wavefront", 3}}, 150); s.line(560, 84, 585, 84, "var(--ink2)", 1.5); s.text(591, 88, "series sum_{k<=D} rho^k", 12);
    // residuals in units of 1e-3
    double rmax = 0; for (size_t i = 0; i < d.size(); ++i) rmax = std::max({rmax, std::fabs(cpu[i] - ser[i]), std::fabs(mk[i] - ser[i]), std::fabs(wf[i] - ser[i]), 1.96 * se[i]}); rmax = std::ceil(rmax * 1e3 + 0.5);
    Ax b{90, 380, 760, 170, 1, 16, -rmax, rmax};
    frame(s, b, lin_ticks(1, 16, 1, "%.0f"), lin_ticks(-rmax, rmax, rmax / 2, "%.1f"), "maximum path depth D", "residual (x 1e-3)", true);
    s.line(b.x, b.Y(0), b.x + b.w, b.Y(0), "var(--ink2)", 1.5);
    Pts up, dn, band; for (size_t i = 0; i < d.size(); ++i) up.push_back({b.X(d[i]), b.Y(1.96e3 * se[i])}); for (size_t i = d.size(); i-- > 0;) dn.push_back({b.X(d[i]), b.Y(-1.96e3 * se[i])});
    band = up; band.insert(band.end(), dn.begin(), dn.end()); s.area(band, "var(--ink2)", 0.16);
    for (size_t i = 0; i < d.size(); ++i) {
        s.square(b.X(d[i]) - 6, b.Y((cpu[i] - ser[i]) * 1e3), 3.5, col(2)); s.circle(b.X(d[i]), b.Y((mk[i] - ser[i]) * 1e3), 3.8, col(1), "var(--bg)");
        s.circle(b.X(d[i]) + 6, b.Y((wf[i] - ser[i]) * 1e3), 3.8, col(3), "var(--bg)");
    }
    s.text(b.x + b.w - 4, b.y + 14, "shaded: +/- 1.96 standard errors of the megakernel mean", 11, "end", true);
    s.caption(612, {"With max depth D the estimator returns the first D+1 terms of the Neumann series, so E[L_D] = sum rho^k; Russian roulette (from depth 3) reweights survivors by 1/q and must leave this mean unchanged.",
                    "All three arms use the same random numbers, so their residuals coincide: this shows the GPU arms add no error of their own, not three independent confirmations. The common residual drifts to about -1e-3, inside the 95% band (+/- 2.3e-3) but monotone; a longer run would be needed to tell a tiny roulette or numerical bias from chance."});
    return s.str();
}

// ---- figure 3: variance concentration ----
std::string fig3(const Json& J) {
    const Json& v = J.at("variance");
    Svg s(900, 600, "Variance concentration", "Lorenz curves of per-pixel variance for the three scenes.");
    const Json& t = v.at("trianglehv");
    s.header("The top 1% of pixels hold " + f(100 * v.at("cornell").at("top1pct_share").num, "%.0f") + "%, " + f(100 * v.at("divergence").at("top1pct_share").num, "%.0f") + "% and " + f(100 * t.at("top1pct_share").num, "%.0f") + "% of the variance",
             "Lorenz curve of the per-pixel luminance variance (320x180 pixels, 64 spp); Gini coefficient and excess kurtosis from the same data");
    Ax a{90, 110, 560, 380, 0, 1, 0, 1};
    frame(s, a, lin_ticks(0, 1, 0.2, "%.1f"), lin_ticks(0, 1, 0.2, "%.1f"), "fraction of pixels, sorted by variance (low to high)", "cumulative share of total variance", true);
    s.line(a.X(0), a.Y(0), a.X(1), a.Y(1), "var(--ink2)", 1.5, "6 4"); s.text(a.X(0.55), a.Y(0.5) + 18, "equal variance everywhere", 11, "start", true);
    int ci = 1; std::vector<std::pair<std::string, int>> leg;
    for (const char* name : kScenes) {
        const Json& e = v.at(name); auto lor = vec(e.at("lorenz")); Pts p; for (size_t k = 0; k < lor.size(); ++k) p.push_back({a.X(k / 100.0), a.Y(lor[k])});
        s.poly(p, col(ci), 3); leg.push_back({name, ci});
        double ky = 140 + 74 * (ci - 1);
        s.rect(680, ky - 14, 12, 12, col(ci), 2); s.text(698, ky - 3, name, 13, "start", false, true);
        s.text(698, ky + 16, "Gini " + f(e.at("gini").num, "%.3f") + ", top 1% = " + f(100 * e.at("top1pct_share").num, "%.1f") + "%", 11, "start", true);
        s.text(698, ky + 32, "max/mean " + f(e.at("max_over_mean").num, "%.0f") + ", kurtosis " + f(e.at("kurtosis").num, "%.0f"), 11, "start", true);
        ++ci;
    }
    s.caption(565, {"The sample variance of a pixel is sigma_i^2. A Gini coefficient near 1 means almost all of the image error sits in a few pixels: bright caustic or glancing-light paths (fireflies).",
                    "The excess kurtosis (1e2 to 1e4, against 0 for a Gaussian) shows why the mean error converges slowly even though the RMSE obeys N^-1/2: a few rare, very bright paths dominate the sum.",
                    "Practical consequence: the megakernel/wavefront comparison must be made at equal sample counts, never equal visual noise; clamping or better light sampling would change the tail, not the estimator's bias law."});
    return s.str();
}

// ---- figure 4: path survival ----
std::string fig4(const Json& J) {
    const Json& sv = J.at("survival");
    Svg s(900, 620, "Path survival", "Fraction of paths still alive after each bounce for three scenes with geometric tail fits.");
    double s8lo = 1, s8hi = 0;
    for (const char* name : kScenes) { auto e = vec(sv.at(name).at("extension_rays_by_cap")); double S8 = (e[8] - e[7]) / sv.at(name).at("paths").num; s8lo = std::min(s8lo, S8); s8hi = std::max(s8hi, S8); }
    (void)s8lo;
    s.header("By bounce 8 at most " + f(100 * s8hi, "%.1f") + "% of paths are still alive, so wavefront queues drain",
             "S_k = fraction of paths that trace a ray at bounce k = (E_k - E_(k-1)) / N from extension-ray counts at depth caps; 320x180, 8 spp");
    Ax a{90, 110, 560, 380, 0.5, 16.5, 0.00002, 2.0, false, true};
    Ticks yt; for (double v : {0.0001, 0.001, 0.01, 0.1, 1.0}) yt.push_back({v, f(v, "%g")});
    frame(s, a, lin_ticks(1, 16, 3, "%.0f"), yt, "bounce k", "S_k (log scale)", true);
    int ci = 1; std::vector<std::string> info;
    for (const char* name : kScenes) {
        auto e = vec(sv.at(name).at("extension_rays_by_cap")); double N = sv.at(name).at("paths").num;
        std::vector<double> k, ln; Pts p; int kmax = 0;
        for (int d = 1; d <= 16; ++d) { double S = (e[d] - e[d - 1]) / N; if (S * N < 30) continue; kmax = d; p.push_back({a.X(d), a.Y(S)}); if (d >= 4) { k.push_back(d); ln.push_back(std::log(S)); } }
        s.poly(p, col(ci), 2.5); for (auto& q : p) s.circle(q.first, q.second, 3.5, col(ci), "var(--bg)");
        Fit fit = ols(k, ln); double r = std::exp(fit.b), perpath = e[16] / N;
        Pts g; for (int d = 4; d <= kmax; ++d) g.push_back({a.X(d), a.Y(std::exp(fit.a + fit.b * d))}); s.poly(g, "var(--ink2)", 1.2, "4 4");
        double ky = 140 + 84 * (ci - 1);
        s.rect(680, ky - 14, 12, 12, col(ci), 2); s.text(698, ky - 3, name, 13, "start", false, true);
        s.text(698, ky + 16, "tail ratio r = " + f(r, "%.3f") + " per bounce", 11, "start", true); s.text(698, ky + 32, "mean rays per path " + f(perpath, "%.2f"), 11, "start", true);
        s.text(698, ky + 48, "S_1 = " + f((e[1] - e[0]) / N, "%.2f") + "  (primary hit rate)", 11, "start", true);
        ++ci;
    }
    s.text(a.X(15.8), a.Y(0.00004), "dashed: geometric fit S_k = S_0 r^k, k >= 4", 11, "end", true);
    s.caption(585, {"Each Lambertian bounce either terminates (miss, emitter, depth cap) or survives Russian roulette with probability q = clamp(max throughput, 0.05, 0.95) from depth 3, so the tail is geometric.",
                    "In the wavefront the active queue at bounce k has N S_k entries: late passes launch few threads, so fixed per-pass costs (launch, scan, host synchronisation) are amortised over ever fewer paths.",
                    "The megakernel keeps those threads resident and lets finished lanes idle; this occupancy decay is one concrete reason the wavefront loses at modest path counts. Points with fewer than 30 counted rays (S_k N < 30) are omitted."});
    return s.str();
}

// ---- figure 5: SIMT efficiency model ----
std::string fig5(const Json& J, const std::string& ncu) {
    const Json& d = J.at("divergence");
    Svg s(900, 620, "SIMT efficiency", "Predicted and measured fraction of active lanes per warp for the megakernel and the wavefront.");
    double eff[3][2];
    for (int i = 0; i < 3; ++i) { eff[i][0] = lane_efficiency(load_ncu(ncu + "/" + kScenes[i] + "-megakernel.summary.json")); eff[i][1] = lane_efficiency(load_ncu(ncu + "/" + kScenes[i] + "-wavefront.summary.json")); }
    s.header("Wavefront lane efficiency: " + f(100 * (eff[0][1] - eff[0][0]), "%+.0f") + " points on cornell, " + f(100 * (eff[1][1] - eff[1][0]), "%+.0f") + " on divergence, " + f(100 * (eff[2][1] - eff[2][0]), "%+.0f") + " on trianglehv",
             "Active lanes / 32: material-mix models vs Nsight Compute thread_inst_executed_per_inst_executed (instruction-weighted)");
    Ax a{90, 120, 760, 340, -0.5, 2.5, 0, 1.0};
    Ticks xt; for (int i = 0; i < 3; ++i) xt.push_back({(double)i, kScenes[i]});
    frame(s, a, xt, lin_ticks(0, 1, 0.2, "%.1f"), "", "fraction of 32 lanes active");
    legend(s, 90, 96, {{"model: iid mix", 1}, {"model: 16x2 tiles", 2}, {"measured megakernel", 3}, {"measured wavefront", 4}}, 190);
    for (int i = 0; i < 3; ++i) {
        const Json& e = d.at(kScenes[i]);
        double v[4] = {1.0 / e.at("expected_distinct_iid").num, 1.0 / e.at("mean_distinct_per_warp").num,
                       eff[i][0], eff[i][1]};
        const char* names[4] = {"iid model", "tile model", "megakernel", "wavefront"};
        for (int b = 0; b < 4; ++b) {
            double x = a.X(i) + (b - 2) * 40 + 2, h = v[b] * a.h;
            s.rect(x, a.Y(0) - h, 36, h, col(b + 1), 3, 1, std::string(kScenes[i]) + " " + names[b] + ": " + f(v[b], "%.3f"));
            s.text(x + 18, a.Y(0) - h - 6, f(v[b], "%.2f"), 11, "middle");
        }
        s.text(a.X(i), a.Y(0) + 38, "entropy " + f(e.at("entropy_bits").num, "%.2f") + " bits, " + f(e.at("mean_distinct_per_warp").num, "%.2f") + " materials per warp", 11, "middle", true);
    }
    s.caption(552, {"Materials seen by a warp: if the 32 lanes drew materials independently with probabilities p_i, the expected number of distinct branches is E[K] = sum_i (1 - (1-p_i)^32), and serialised execution gives efficiency 1/E[K].",
                    "Real warps are spatially coherent (a 16x2 pixel tile usually sees one or two materials), so the tile estimate sits well above the iid bound. Primary hits only; secondary bounces decohere further.",
                    "The measured megakernel efficiency is far below the material-only estimate because BVH traversal runs a different number of iterations per lane; the wavefront does not remove that (kernel_extend stays divergent), and on trianglehv, where traversal dominates the run time, it ends up no better than the megakernel."});
    return s.str();
}

// ---- figure 6: roofline ----
std::string fig6(const std::string& ncu) {
    auto mk = load_ncu(ncu + "/divergence-megakernel.summary.json"), wf = load_ncu(ncu + "/divergence-wavefront.summary.json"), dev = load_ncu(ncu + "/device.summary.json");
    const double sms = get(dev[0], "device__attribute_multiprocessor_count", 16), clk = get(dev[0], "sm__cycles_elapsed.avg.per_second", 1.35e9);
    const double bw = get(dev[0], "dram__bytes.sum.peak_sustained", 32) * get(dev[0], "dram__cycles_elapsed.avg.per_second", 6e9);
    const double peak = sms * 64.0 * 2.0 * clk, knee = peak / bw;   // FP32 FMA lanes per SM on Turing
    struct Pt { std::string name; double share, gf, ai; };
    auto collect = [&](const std::vector<Kernel>& ks) {
        std::vector<Pt> v; double tot = 0; for (const Kernel& k : ks) tot += kernel_seconds(k) * k.launches;
        for (const Kernel& k : ks) {
            double t = kernel_seconds(k), fl = get(k, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum") + get(k, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum") + 2 * get(k, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum");
            double by = std::max(get(k, "dram__bytes.sum"), 1.0); if (t <= 0 || fl <= 0) continue;
            v.push_back({k.name, t * k.launches / tot, fl / t / 1e9, fl / by});
        }
        std::sort(v.begin(), v.end(), [](const Pt& a, const Pt& b) { return a.share > b.share; }); return v;
    };
    auto pm = collect(mk), pw = collect(wf);
    double left = 0; for (const Pt& p : pw) if (p.ai < knee) left += p.share;
    Svg s(900, 640, "Roofline", "Arithmetic intensity and floating-point throughput of every kernel against the device roofline.");
    s.header(f(100 * left, "%.0f") + "% of wavefront kernel time is left of the ridge point; the megakernel is right of it",
             "divergence, 1280x720, 1 spp. FP32 peak " + f(peak / 1e9, "%.0f") + " GFLOP/s (" + f(sms, "%.0f") + " SMs, " + f(clk / 1e9, "%.2f") + " GHz), DRAM peak " + f(bw / 1e9, "%.0f") + " GB/s, ridge " + f(knee, "%.0f") + " FLOP/B");
    Ax a{90, 110, 560, 380, 0.01, 1000, 0.1, peak / 1e9 * 2, true, true};
    Ticks xt, yt; for (double v : {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}) xt.push_back({v, f(v, "%g")}); for (double v : {0.1, 1.0, 10.0, 100.0, 1000.0}) yt.push_back({v, f(v, "%g")});
    frame(s, a, xt, yt, "arithmetic intensity (FLOP per DRAM byte)", "achieved GFLOP/s", true);
    Pts roof; for (double x : {0.01, knee, 1000.0}) roof.push_back({a.X(x), a.Y(std::min(peak, bw * x) / 1e9)}); s.poly(roof, "var(--ink2)", 2.5);
    s.text(a.X(knee) + 6, a.Y(peak / 1e9) - 8, "compute roof", 11, "start", true); s.text(a.X(0.012), a.Y(bw * 0.012 / 1e9) - 10, "memory roof (slope = DRAM bandwidth)", 11, "start", true);
    legend(s, 90, 90, {{"megakernel", 1}, {"wavefront kernels (area = share of kernel time)", 2}}, 150);
    auto draw = [&](const std::vector<Pt>& v, int color, bool number) {
        int n = 0;
        for (const Pt& p : v) {
            double x = a.X(std::min(std::max(p.ai, 0.0105), 990.0)), y = a.Y(std::max(p.gf, 0.11)), r = 4 + 22 * std::sqrt(p.share);
            s.circle(x, y, r, col(color), "var(--bg)", p.name + ": " + f(p.gf, "%.1f") + " GFLOP/s, " + f(p.ai, "%.2f") + " FLOP/B, " + f(100 * p.share, "%.1f") + "% of kernel time");
            if (number && p.share > 0.02) s.text(x, y + 4, std::to_string(++n), 11, "middle", false, true);
        }
    };
    draw(pw, 2, true); draw(pm, 1, false);
    s.text(a.X(pm[0].ai) , a.Y(pm[0].gf) + 40, "megakernel", 11, "middle", true);
    double ky = 130; s.text(680, ky, "wavefront kernels", 12, "start", false, true); ky += 22; int n = 0;
    for (const Pt& p : pw) { if (p.share <= 0.02) continue; s.text(680, ky, std::to_string(++n) + "  " + p.name, 11); s.text(680, ky + 14, f(100 * p.share, "%.0f") + "% of time, " + f(p.gf, "%.0f") + " GFLOP/s, " + f(p.ai, "%.2f") + " FLOP/B", 10, "start", true); ky += 34; }
    s.caption(550, {"Roofline model: attainable P = min(P_peak, B x AI), ridge point AI* = P_peak / B. FLOP = FADD + FMUL + 2 FFMA thread instructions (contraction is disabled in this build, so FFMA is rare); bytes are DRAM traffic from Nsight Compute.",
                    "Both roofs come from profiler counters (SM count, SM clock, DRAM bytes per cycle and DRAM clock), not from a data sheet.",
                    "No kernel reaches 10% of peak FLOP/s: path tracing is dominated by integer, branch and memory-latency work (intersection tests, pointer chasing), which a FLOP roofline does not credit. Kernels left of AI* (classify, shading) are limited by DRAM; the wavefront's extra state traffic moves work to that side."});
    return s.str();
}

// ---- figure 7: scaling law ----
std::string fig7(const Json& J) {
    const Json& sc = J.at("scaling");
    Svg s(900, 700, "Scaling with pixel count", "Render time against megapixels with linear fits, and the resulting speedup curve for two scenes.");
    const Json& dv = sc.at("divergence");
    double bm = dv.at("megakernel_fit").at("b").num, bw = dv.at("wavefront_fit").at("b").num, aw = dv.at("wavefront_fit").at("a").num, am = dv.at("megakernel_fit").at("a").num;
    s.header("T = alpha + beta P: the wavefront pays a fixed " + f(aw, "%.0f") + " ms plus a steeper slope",
             "4 spp, median of 5 timed runs per size, 16:9 frames from 128 to 1920 px wide; divergence asymptote beta_mk / beta_wf = " + f(bm / bw, "%.2f"));
    Ax a{90, 110, 760, 230, 0, 2.2, 0, 0};
    double ymax = 0; for (const char* n : {"cornell", "divergence"}) for (double v : vec(sc.at(n).at("wavefront_ms"))) ymax = std::max(ymax, v); a.y1 = std::ceil(ymax / 50) * 50;
    frame(s, a, lin_ticks(0, 2, 0.5, "%.1f"), lin_ticks(0, a.y1, a.y1 / 4, "%.0f"), "megapixels P", "render time (ms)", true);
    legend(s, 90, 90, {{"megakernel", 1}, {"wavefront", 2}}, 130); s.text(360, 90, "solid + filled: divergence     dashed + hollow: cornell", 12, "start", true);
    for (int si = 0; si < 2; ++si) {
        const char* name = si == 0 ? "divergence" : "cornell"; const Json& e = sc.at(name);
        auto px = vec(e.at("megapixels"));
        for (int arm = 0; arm < 2; ++arm) {
            auto y = vec(e.at(arm == 0 ? "megakernel_ms" : "wavefront_ms")); const Json& ft = e.at(arm == 0 ? "megakernel_fit" : "wavefront_fit");
            Pts l = {{a.X(0), a.Y(ft.at("a").num)}, {a.X(2.1), a.Y(ft.at("a").num + ft.at("b").num * 2.1)}}; s.poly(l, col(arm + 1), 2, si ? "6 4" : nullptr);
            for (size_t i = 0; i < px.size(); ++i) { if (si == 0) s.circle(a.X(px[i]), a.Y(y[i]), 4, col(arm + 1), "var(--bg)", std::string(name) + " " + (arm ? "wavefront" : "megakernel") + " " + f(px[i], "%.2f") + " MP: " + f(y[i], "%.1f") + " ms"); else s.square(a.X(px[i]), a.Y(y[i]), 3.5, col(arm + 1)); }
            if (si == 0) s.text(a.X(2.1) - 4, a.Y(ft.at("a").num + ft.at("b").num * 2.1) - 8, f(ft.at("a").num, "%.1f") + " + " + f(ft.at("b").num, "%.1f") + " P  (R^2 " + f(ft.at("r2").num, "%.3f") + ")", 11, "end", true);
        }
    }
    Ax b{90, 420, 760, 190, 0, 2.2, 0, 1.5};
    frame(s, b, lin_ticks(0, 2, 0.5, "%.1f"), lin_ticks(0, 1.5, 0.3, "%.1f"), "megapixels P", "speedup T_mk / T_wf", true);
    s.line(b.x, b.Y(1.3), b.x + b.w, b.Y(1.3), "var(--ink2)", 1.5, "6 4"); s.text(b.x + b.w - 4, b.Y(1.3) - 6, "target 1.30", 11, "end", true);
    s.line(b.x, b.Y(1.0), b.x + b.w, b.Y(1.0), "var(--ink2)", 1, "2 4");
    for (int si = 0; si < 2; ++si) {
        const char* name = si == 0 ? "divergence" : "cornell"; const Json& e = sc.at(name); auto px = vec(e.at("megapixels")), m = vec(e.at("megakernel_ms")), w = vec(e.at("wavefront_ms"));
        const Json& fm = e.at("megakernel_fit"); const Json& fw = e.at("wavefront_fit"); Pts curve;
        for (double p = 0.05; p <= 2.1; p += 0.05) curve.push_back({b.X(p), b.Y((fm.at("a").num + fm.at("b").num * p) / (fw.at("a").num + fw.at("b").num * p))});
        s.poly(curve, col(3), 2, si ? "6 4" : nullptr);
        for (size_t i = 0; i < px.size(); ++i) { if (si == 0) s.circle(b.X(px[i]), b.Y(m[i] / w[i]), 4, col(3), "var(--bg)", std::string(name) + " " + f(px[i], "%.2f") + " MP: speedup " + f(m[i] / w[i], "%.2f")); else s.square(b.X(px[i]), b.Y(m[i] / w[i]), 3.5, col(3)); }
        double asym = fm.at("b").num / fw.at("b").num; s.text(b.X(1.2), b.Y(asym) + (si ? 26 : -12), std::string(name) + " limit beta_mk/beta_wf = " + f(asym, "%.2f"), 11, "start", true);
    }
    (void)am;
    s.caption(668, {"Model: T_arm(P) = alpha + beta P. alpha collects per-frame fixed costs (launches, scans, host synchronisation per bounce); beta is the cost per pixel. The speedup S(P) = (alpha_m + beta_m P)/(alpha_w + beta_w P) tends to beta_m/beta_w.",
                    "Both asymptotes are below 1: even with infinitely many pixels to hide fixed costs the wavefront's per-pixel work (state read/write, classification, compaction) exceeds the megakernel's, so more resolution cannot rescue it."});
    return s.str();
}

// ---- figure 8: BVH quality ----
std::string fig8(const Json& J) {
    const Json& b = J.at("bvh");
    Svg s(900, 640, "BVH quality", "Leaf depth distribution and child to parent surface-area ratios of the SAH BVH for three scenes.");
    const Json& t = b.at("trianglehv");
    s.header("SAH splits halve surface area per level: depth " + f(t.at("max_depth").num, "%.0f") + " for " + f(t.at("primitives").num, "%.0f") + " primitives",
             "Binned-SAH BVH with at most 4 primitives per leaf, stackless skip-link layout; statistics from the built node arrays");
    Ax a{90, 110, 360, 290, 0, 20, 0, 0.6};
    frame(s, a, lin_ticks(0, 20, 4, "%.0f"), lin_ticks(0, 0.6, 0.2, "%.1f"), "leaf depth", "fraction of leaves", true);
    Ax c{500, 110, 350, 290, 0, 20, 0, 1.0};
    frame(s, c, lin_ticks(0, 20, 4, "%.0f"), lin_ticks(0, 1.0, 0.2, "%.1f"), "parent depth", "mean child / parent surface area", true);
    s.line(c.x, c.Y(0.5), c.x + c.w, c.Y(0.5), "var(--ink2)", 1.5, "6 4"); s.text(c.x + c.w - 4, c.Y(0.5) - 6, "ideal halving 0.5", 11, "end", true);
    int ci = 1;
    for (const char* name : kScenes) {
        const Json& e = b.at(name); auto dh = vec(e.at("leaf_depth_hist")), rr = vec(e.at("child_to_parent_area_by_depth"));
        double tot = 0; for (double v : dh) tot += v; Pts p, q;
        for (size_t d = 0; d < dh.size(); ++d) { p.push_back({a.X((double)d), a.Y(dh[d] / tot)}); }
        for (size_t d = 0; d < rr.size(); ++d) q.push_back({c.X((double)d), c.Y(rr[d])});
        s.poly(p, col(ci), 2.5); for (auto& m : p) s.circle(m.first, m.second, 3, col(ci), "var(--bg)");
        s.poly(q, col(ci), 2.5); for (auto& m : q) s.circle(m.first, m.second, 3, col(ci), "var(--bg)");
        ++ci;
    }
    legend(s, 90, 90, {{kScenes[0], 1}, {kScenes[1], 2}, {kScenes[2], 3}}, 130);
    s.text(90, 500, "scene", 12, "start", false, true); s.text(260, 500, "primitives", 12, "end", false, true); s.text(350, 500, "nodes", 12, "end", false, true); s.text(430, 500, "depth", 12, "end", false, true);
    s.text(540, 500, "log2 leaves", 12, "end", false, true); s.text(640, 500, "SAH cost", 12, "end", false, true); s.text(850, 500, "leaves by size 1/2/3/4", 12, "end", false, true);
    ci = 1;
    for (const char* name : kScenes) {
        const Json& e = b.at(name); double y = 524 + 24 * (ci - 1); auto ls = vec(e.at("leaf_size_hist"));
        s.rect(90, y - 11, 12, 12, col(ci), 2); s.text(108, y, name, 12);
        s.text(260, y, f(e.at("primitives").num, "%.0f"), 12, "end"); s.text(350, y, f(e.at("nodes").num, "%.0f"), 12, "end"); s.text(430, y, f(e.at("max_depth").num, "%.0f"), 12, "end");
        s.text(540, y, f(std::log2(e.at("leaves").num), "%.1f"), 12, "end"); s.text(640, y, f(e.at("sah_cost").num, "%.1f"), 12, "end");
        s.text(850, y, f(ls[0], "%.0f") + " / " + f(ls[1], "%.0f") + " / " + f(ls[2], "%.0f") + " / " + f(ls[3], "%.0f"), 12, "end");
        ++ci;
    }
    s.caption(600, {"SAH cost C = sum over internal nodes of SA(n)/SA(root) + sum over leaves of SA(l)/SA(root) x (primitives in l): the expected number of node and primitive tests for a ray that hits the root box.",
                    "A tree of N leaves needs at least log2(N) levels; the measured maximum depth is a few levels above that bound, so SAH splits are close to balanced while still adapting to the geometry.",
                    "The per-level area ratio sits near 1/2 (children are roughly half the parent's surface), the signature of good spatial splits; the root split is the least even one (ratio at depth 0 is highest for divergence)."});
    return s.str();
}

// ---- figure 9: RNG quality ----
Pts dummy;
std::string fig9(const Json& J) {
    const Json& r = J.at("rng");
    auto av = vec(r.at("avalanche")), acf = vec(r.at("acf")), chi = vec(r.at("pair_chi2_by_seed"));
    double se = r.at("avalanche_se").num, band = r.at("acf_band95").num, dof = r.at("pair_dof").num;
    double zmax = 0; for (double v : av) zmax = std::max(zmax, std::fabs((v - 0.5) / se));
    double ks = r.at("ks_d").num, kc = r.at("ks_critical_5pct").num;
    Svg s(900, 720, "Counter-based RNG quality", "Avalanche matrix, serial correlation and pair chi-square for the counter-based random generator.");
    double pbonf = std::min(1.0, 6144.0 * std::erfc(zmax / std::sqrt(2.0)));
    s.header("Path-key hash: mean bit-flip probability " + f(r.at("avalanche_mean").num, "%.4f") + ", KS D " + f(ks * 1e3, "%.2f") + "e-3 < " + f(kc * 1e3, "%.2f") + "e-3 critical",
             "20000 trials x 96 input bits; worst of 6144 cells: " + f(zmax, "%.1f") + " standard errors (Bonferroni p = " + f(pbonf, "%.2f") + ")");
    const double cw = 7, ch = 4.4, hx = 90, hy = 100;
    s.text(hx, hy - 8, "A  bit-flip probability, z = (p - 0.5)/SE", 12, "start", false, true);
    for (int i = 0; i < 96; ++i) for (int o = 0; o < 64; ++o) {
        double z = std::max(-4.5, std::min(4.5, (av[i * 64 + o] - 0.5) / se)), t = std::fabs(z) / 4.5; const bool pos = z > 0;
        double base[3] = {222, 221, 216}, tgt[3] = {pos ? 235.0 : 42.0, pos ? 104.0 : 120.0, pos ? 52.0 : 214.0}; char hex[16];
        std::snprintf(hex, sizeof hex, "#%02x%02x%02x", (int)(base[0] + (tgt[0] - base[0]) * t), (int)(base[1] + (tgt[1] - base[1]) * t), (int)(base[2] + (tgt[2] - base[2]) * t));
        s.rect(hx + o * cw, hy + i * ch, cw - 0.4, ch - 0.3, hex, 0, 1, "input bit " + std::to_string(i) + " -> output bit " + std::to_string(o) + ": p = " + f(av[i * 64 + o], "%.4f") + " (z " + f(z, "%.2f") + ")");
    }
    s.text(hx - 6, hy + 16 * ch, "seed", 11, "end", true); s.text(hx - 6, hy + 48 * ch, "pixel", 11, "end", true); s.text(hx - 6, hy + 80 * ch, "sample", 11, "end", true);
    s.line(hx - 3, hy + 32 * ch, hx - 3 + 3, hy + 32 * ch, "var(--ink2)"); s.line(hx, hy + 32 * ch, hx + 64 * cw, hy + 32 * ch, "var(--bg)", 1.5); s.line(hx, hy + 64 * ch, hx + 64 * cw, hy + 64 * ch, "var(--bg)", 1.5);
    s.text(hx + 32 * cw, hy + 96 * ch + 16, "output bit (0..63 of k0 | k1)", 11, "middle", true);
    // colour key
    for (int i = 0; i < 9; ++i) { double z = -4 + i, t = std::fabs(z) / 4.5; const bool pos = z > 0; double base[3] = {222, 221, 216}, tgt[3] = {pos ? 235.0 : 42.0, pos ? 104.0 : 120.0, pos ? 52.0 : 214.0}; char hex[16];
        std::snprintf(hex, sizeof hex, "#%02x%02x%02x", (int)(base[0] + (tgt[0] - base[0]) * t), (int)(base[1] + (tgt[1] - base[1]) * t), (int)(base[2] + (tgt[2] - base[2]) * t)); s.rect(hx + i * 22, hy + 96 * ch + 30, 22, 10, hex); }
    s.text(hx, hy + 96 * ch + 54, "z = -4", 10, "start", true); s.text(hx + 99, hy + 96 * ch + 54, "0", 10, "middle", true); s.text(hx + 198, hy + 96 * ch + 54, "+4", 10, "end", true);
    // B: chi-square by seed
    const double bx = 625, by = 120, bw = 245, bh = 190;
    s.text(bx, by - 18, "B  16x16 serial test, 32 seeds", 12, "start", false, true);
    auto wh = [&](double z) { double k = dof, c = 2.0 / (9.0 * k); return k * std::pow(1.0 - c + z * std::sqrt(c), 3); };
    double lo = wh(-1.96), hi = wh(1.96), mid = wh(0); double ymin = std::min(lo, *std::min_element(chi.begin(), chi.end())) - 15, ymax = std::max(hi, *std::max_element(chi.begin(), chi.end())) + 15;
    Ax B{bx, by, bw, bh, 0, 33, ymin, ymax}; frame(s, B, lin_ticks(0, 32, 8, "%.0f"), lin_ticks(std::ceil(ymin / 25) * 25, ymax, 25, "%.0f"), "seed", "chi-square (255 dof)");
    s.rect(B.x, B.Y(hi), B.w, B.Y(lo) - B.Y(hi), "var(--ink2)", 0, 0.16); s.line(B.x, B.Y(mid), B.x + B.w, B.Y(mid), "var(--ink2)", 1, "5 4");
    int outside = 0; for (size_t i = 0; i < chi.size(); ++i) { if (chi[i] < lo || chi[i] > hi) ++outside; s.circle(B.X(i + 1.0), B.Y(chi[i]), 3.8, col(1), "var(--bg)", "seed " + std::to_string(i + 1) + ": chi2 " + f(chi[i], "%.1f")); }
    s.text(bx, by + bh + 52, std::to_string(outside) + " of 32 outside the 95% band (1.6 expected)", 11, "start", true);
    // C: autocorrelation
    const double cx = 625, cy = 440, cwid = 245, chh = 140;
    s.text(cx, cy - 18, "C  autocorrelation, lags 1..64", 12, "start", false, true);
    double am = 0; for (double v : acf) am = std::max(am, std::fabs(v)); double yr = std::max(band * 1.6, am * 1.15);
    Ax C{cx, cy, cwid, chh, 0, 65, -yr, yr}; frame(s, C, lin_ticks(0, 64, 16, "%.0f"), lin_ticks(-yr, yr, yr, "%.4f"), "lag", "");
    s.text(C.x - 62, C.y + C.h / 2, "r(lag)", 12, "middle", true, false, -90);
    s.rect(C.x, C.Y(band), C.w, C.Y(-band) - C.Y(band), "var(--ink2)", 0, 0.16);
    for (size_t i = 0; i < acf.size(); ++i) s.line(C.X(i + 1.0), C.Y(0), C.X(i + 1.0), C.Y(acf[i]), col(2), 2);
    s.text(cx, cy + chh + 52, "largest |r| = " + f(am, "%.4f") + ", 95% band +/- " + f(band, "%.4f"), 11, "start", true);
    s.caption(700, {"A: flipping any one input bit should flip each output bit with probability 1/2 (strict avalanche); cells are coloured by deviation in standard errors sqrt(1/(4 x trials)). The worst cell has Bonferroni p = " + f(pbonf, "%.2f") + " over 6144 cells: borderline, consistent with chance but worth repeating with more trials.",
                    "B: consecutive-pixel streams (u_i, u_i+1) binned 16x16 give chi-square near 255 if independent; the shaded band is the 95% interval of chi-square(255). C: autocorrelation of the stream over 2^20 draws against the +/- 1.96/sqrt(N) band.",
                    "Counter-based design: every draw is a pure function of (seed, pixel, sample, depth, dim), so megakernel and wavefront consume identical random numbers regardless of execution order; these tests probe the quality of that mapping."});
    return s.str();
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) { std::cerr << "Usage: svgstats <out-dir> <stats.json> <ncu-dir>\n"; return 2; }
        const std::string dir = argv[1], ncu = argv[3]; Json J = wf::json_parse(read_all(argv[2]));
        write(dir, "fig-01-convergence.svg", fig1(J)); write(dir, "fig-02-bias.svg", fig2(J)); write(dir, "fig-03-variance.svg", fig3(J));
        write(dir, "fig-04-survival.svg", fig4(J)); write(dir, "fig-05-simt-efficiency.svg", fig5(J, ncu)); write(dir, "fig-06-roofline.svg", fig6(ncu));
        write(dir, "fig-07-scaling.svg", fig7(J)); write(dir, "fig-08-bvh.svg", fig8(J)); write(dir, "fig-09-rng.svg", fig9(J));
        std::cout << "wrote 9 figures to " << dir << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 2; }
}
