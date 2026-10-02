// svgplot: draws the benchmark charts as SVG from the raw result files.
//   svgplot <out-dir> <bench-cornell.json> <bench-divergence.json> <bench-trianglehv.json> [stage-times-log]
// Writes timing.svg (median with p10-p90 whiskers), speedup.svg (against the 1.30 target) and, when a
// stage-time log is given, stages.svg (instrumented stage breakdown). Light and dark colours are CSS variables.
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

struct SceneData {
    std::string name;
    double mk_med = 0, mk_p10 = 0, mk_p90 = 0, wf_med = 0, wf_p10 = 0, wf_p90 = 0, speedup = 0;
};

std::string read_all(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}

std::string num(double v, const char* fmt = "%.1f") { char b[64]; std::snprintf(b, sizeof b, fmt, v); return b; }

const char* kStyle =
    "<style>:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100}"
    "@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--grid:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}"
    "text{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;fill:var(--ink)}.sub{fill:var(--ink2)}.axis{stroke:var(--grid);stroke-width:1}"
    ".whisk{stroke:var(--ink2);stroke-width:1.5}.ref{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 4}</style>";

struct Svg {
    std::ostringstream o; int w, h;
    Svg(int w_, int h_, const std::string& title, const std::string& desc) : w(w_), h(h_) {
        o << "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 " << w << ' ' << h << "\" role=\"img\" aria-labelledby=\"t d\">"
          << "<title id=\"t\">" << title << "</title><desc id=\"d\">" << desc << "</desc>" << kStyle
          << "<rect width=\"" << w << "\" height=\"" << h << "\" fill=\"var(--bg)\"/>";
    }
    void text(double x, double y, const std::string& s, int size, const char* anchor = "start", bool sub = false, bool bold = false) {
        o << "<text x=\"" << num(x) << "\" y=\"" << num(y) << "\" font-size=\"" << size << "\" text-anchor=\"" << anchor << "\""
          << (sub ? " class=\"sub\"" : "") << (bold ? " font-weight=\"600\"" : "") << ">" << s << "</text>";
    }
    void line(double x1, double y1, double x2, double y2, const char* cls) {
        o << "<line x1=\"" << num(x1) << "\" y1=\"" << num(y1) << "\" x2=\"" << num(x2) << "\" y2=\"" << num(y2) << "\" class=\"" << cls << "\"/>";
    }
    void bar(double x, double y, double bw, double bh, const char* var, const std::string& tip) {
        // 4px rounded data end at the top, anchored to the baseline.
        double r = std::min(4.0, std::min(bw / 2, bh));
        o << "<path fill=\"var(" << var << ")\" d=\"M" << num(x) << ' ' << num(y + bh) << "V" << num(y + r) << "Q" << num(x) << ' ' << num(y) << ' ' << num(x + r) << ' ' << num(y)
          << "H" << num(x + bw - r) << "Q" << num(x + bw) << ' ' << num(y) << ' ' << num(x + bw) << ' ' << num(y + r) << "V" << num(y + bh) << "Z\"><title>" << tip << "</title></path>";
    }
    std::string str() { return o.str() + "</svg>\n"; }
};

double nice_max(double v) { double e = std::pow(10.0, std::floor(std::log10(v))); for (double m : {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0}) if (m * e >= v) return m * e; return 10 * e; }

std::string timing_svg(const std::vector<SceneData>& d) {
    const int W = 760, H = 440, L = 78, R = 24, T = 96, B = 64;
    Svg s(W, H, "Render time per scene", "Median render time of the megakernel and the wavefront on three scenes with 10th to 90th percentile whiskers.");
    s.text(L, 30, "The wavefront was slower than the megakernel on all three scenes", 17, "start", false, true);
    s.text(L, 50, "Median of 5 timed runs, 1280x720, 64 samples per pixel, depth 8; whiskers show p10 to p90", 12, "start", true);
    double ymax = 0; for (auto& x : d) ymax = std::max({ymax, x.mk_p90, x.wf_p90}); ymax = nice_max(ymax * 1.08);
    double pw = W - L - R, ph = H - T - B;
    auto Y = [&](double v) { return T + ph - v / ymax * ph; };
    for (int i = 0; i <= 5; ++i) { double v = ymax * i / 5; s.line(L, Y(v), W - R, Y(v), "axis"); s.text(L - 8, Y(v) + 4, num(v, "%.0f"), 11, "end", true); }
    s.text(18, T + ph / 2, "", 1);
    // legend (two series: always present)
    double lx = L; for (auto [name, var] : std::vector<std::pair<std::string, std::string>>{{"megakernel", "--s1"}, {"wavefront", "--s2"}}) {
        s.o << "<rect x=\"" << lx << "\" y=\"66\" width=\"12\" height=\"12\" rx=\"2\" fill=\"var(" << var << ")\"/>"; s.text(lx + 18, 77, name, 12); lx += 110; }
    double gw = pw / d.size(), bw = 52;
    for (size_t i = 0; i < d.size(); ++i) {
        double cx = L + gw * (i + 0.5);
        struct Bar { double med, p10, p90; const char* var; const char* nm; double x; } bars[2] = {{d[i].mk_med, d[i].mk_p10, d[i].mk_p90, "--s1", "megakernel", cx - bw - 1}, {d[i].wf_med, d[i].wf_p10, d[i].wf_p90, "--s2", "wavefront", cx + 1}};
        for (auto& b : bars) {
            s.bar(b.x, Y(b.med), bw, Y(0) - Y(b.med), b.var, d[i].name + " " + b.nm + ": median " + num(b.med) + " ms, p10 " + num(b.p10) + ", p90 " + num(b.p90));
            double mx = b.x + bw / 2; s.line(mx, Y(b.p10), mx, Y(b.p90), "whisk"); s.line(mx - 5, Y(b.p10), mx + 5, Y(b.p10), "whisk"); s.line(mx - 5, Y(b.p90), mx + 5, Y(b.p90), "whisk");
            s.text(mx, Y(b.p90) - 7, num(b.med, "%.0f"), 12, "middle");
        }
        s.text(cx, H - B + 22, d[i].name, 13, "middle");
    }
    s.text(L, H - 14, "Render time in milliseconds (lower is better)", 11, "start", true);
    return s.str();
}

std::string speedup_svg(const std::vector<SceneData>& d) {
    const int W = 760, H = 420, L = 78, R = 24, T = 84, B = 60;
    Svg s(W, H, "Speedup of the wavefront over the megakernel", "Speedup per scene against the 1.30 target; every scene is below 1.");
    s.text(L, 30, "Speedup of the wavefront over the megakernel (target 1.30)", 17, "start", false, true);
    s.text(L, 50, "Megakernel median time divided by wavefront median time; above 1 means the wavefront is faster", 12, "start", true);
    double ymax = 1.6, pw = W - L - R, ph = H - T - B;
    auto Y = [&](double v) { return T + ph - v / ymax * ph; };
    for (int i = 0; i <= 4; ++i) { double v = 0.4 * i; s.line(L, Y(v), W - R, Y(v), "axis"); s.text(L - 8, Y(v) + 4, num(v, "%.1f"), 11, "end", true); }
    double gw = pw / d.size(), bw = 84;
    for (size_t i = 0; i < d.size(); ++i) {
        double cx = L + gw * (i + 0.5);
        s.bar(cx - bw / 2, Y(d[i].speedup), bw, Y(0) - Y(d[i].speedup), "--s1", d[i].name + ": speedup " + num(d[i].speedup, "%.2f"));
        s.text(cx, Y(d[i].speedup) - 8, num(d[i].speedup, "%.2f"), 14, "middle", false, true);
        s.text(cx, H - B + 22, d[i].name, 13, "middle");
    }
    s.line(L, Y(1.0), W - R, Y(1.0), "ref"); s.text(W - R, Y(1.0) - 6, "equal time (1.00)", 11, "end", true);
    s.line(L, Y(1.3), W - R, Y(1.3), "ref"); s.text(W - R, Y(1.3) - 6, "target (1.30)", 11, "end", true);
    return s.str();
}

struct Stages { std::string name; double extend = 0, classify = 0, lambert = 0, metal = 0, dielectric = 0, shadow = 0, compact = 0, other = 0, total = 0; };

std::vector<Stages> parse_stages(const std::string& text) {
    std::vector<Stages> out; std::istringstream in(text); std::string line; Stages cur; bool have = false;
    while (std::getline(in, line)) {
        char name[64]; double plain, tot;
        if (std::sscanf(line.c_str(), "%63s spp4 plain %lf ms | instrumented total %lf ms", name, &plain, &tot) == 3) { cur = Stages(); cur.name = name; cur.total = tot; have = true; continue; }
        double a, b, c, d2, e, f, g, h;
        if (have && std::sscanf(line.c_str(), "extend %lf classify %lf lambert %lf metal %lf dielectric %lf shadow %lf compact %lf other %lf", &a, &b, &c, &d2, &e, &f, &g, &h) == 8) {
            cur.extend = a; cur.classify = b; cur.lambert = c; cur.metal = d2; cur.dielectric = e; cur.shadow = f; cur.compact = g; cur.other = h; out.push_back(cur); have = false; }
    }
    return out;
}

std::string stages_svg(const std::vector<Stages>& st) {
    const int W = 760, H = 330, L = 110, R = 24, T = 100;
    Svg s(W, H, "Where the wavefront time goes", "Instrumented stage breakdown per scene grouped into traversal, shading, shadow rays and bookkeeping.");
    s.text(L - 100, 30, "Where the wavefront time goes", 17, "start", false, true);
    s.text(L - 100, 50, "One instrumented run, 1280x720, 4 samples per pixel; stage times include synchronisation, read the proportions", 12, "start", true);
    struct G { const char* nm; const char* var; double Stages::*unused; } groups[4] = {{"traversal", "--s1", nullptr}, {"shading", "--s2", nullptr}, {"shadow rays", "--s3", nullptr}, {"bookkeeping (classify, compact, other)", "--s4", nullptr}};
    double lx = L - 100; double ly = 76;
    for (int g = 0; g < 4; ++g) { s.o << "<rect x=\"" << lx << "\" y=\"" << ly - 10 << "\" width=\"12\" height=\"12\" rx=\"2\" fill=\"var(" << groups[g].var << ")\"/>"; s.text(lx + 18, ly, groups[g].nm, 12); lx += (g == 3 ? 0 : 18 + 8.0 * std::string(groups[g].nm).size() + 22); }
    double xmax = 0; for (auto& x : st) xmax = std::max(xmax, x.extend + x.classify + x.lambert + x.metal + x.dielectric + x.shadow + x.compact + x.other); xmax = nice_max(xmax * 1.05);
    double pw = W - L - R, rowh = 40;
    for (int i = 0; i <= 5; ++i) { double v = xmax * i / 5, x = L + v / xmax * pw; s.line(x, T - 6, x, T + rowh * st.size() + 6, "axis"); s.text(x, T + rowh * st.size() + 24, num(v, "%.0f"), 11, "middle", true); }
    for (size_t i = 0; i < st.size(); ++i) {
        double vals[4] = {st[i].extend, st[i].lambert + st[i].metal + st[i].dielectric, st[i].shadow, st[i].classify + st[i].compact + st[i].other};
        double x = L, y = T + rowh * i;
        s.text(L - 10, y + 17, st[i].name, 13, "end");
        for (int g = 0; g < 4; ++g) {
            double w = vals[g] / xmax * pw; if (w <= 0) continue;
            s.o << "<rect x=\"" << num(x) << "\" y=\"" << num(y) << "\" width=\"" << num(std::max(0.0, w - 2)) << "\" height=\"26\" rx=\"3\" fill=\"var(" << groups[g].var << ")\"><title>" << st[i].name << " " << groups[g].nm << ": " << num(vals[g]) << " ms</title></rect>";
            if (w > 38) s.text(x + (w - 2) / 2, y + 18, num(vals[g], "%.0f"), 12, "middle");
            x += w;
        }
    }
    s.text(L, H - 12, "Milliseconds in one instrumented render (spp 4)", 11, "start", true);
    return s.str();
}

void write(const std::string& path, const std::string& text) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << text;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5) { std::cerr << "Usage: svgplot <out-dir> <bench.json> <bench.json> <bench.json> [stage-times-log]\n"; return 2; }
        std::string dir = argv[1];
        std::vector<SceneData> scenes;
        for (int i = 2; i < 5; ++i) {
            Json j = wf::json_parse(read_all(argv[i]));
            if (j.at("width").num != 1280 || j.at("height").num != 720 || j.at("spp").num != 64) throw std::runtime_error(std::string(argv[i]) + " is not a full-protocol result (1280x720, spp 64)");
            SceneData s; s.name = j.at("scene_name").str; s.speedup = j.at("speedup").num;
            for (const Json& c : j.at("conditions").arr) {
                const bool mk = c.at("mode").str == "megakernel";
                (mk ? s.mk_med : s.wf_med) = c.at("median").num; (mk ? s.mk_p10 : s.wf_p10) = c.at("p10").num; (mk ? s.mk_p90 : s.wf_p90) = c.at("p90").num;
            }
            scenes.push_back(s);
        }
        write(dir + "/timing.svg", timing_svg(scenes));
        write(dir + "/speedup.svg", speedup_svg(scenes));
        if (argc >= 6) write(dir + "/stages.svg", stages_svg(parse_stages(read_all(argv[5]))));
        std::cout << "wrote charts to " << dir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
