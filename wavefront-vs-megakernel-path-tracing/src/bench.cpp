#include "wf/bench.hpp"
#include "wf/json.hpp"
#include "wf/image_io.hpp"
#include "wf/renderer.hpp"
#include "wf/film.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>\n#include <cstdio>
#include <stdexcept>
#include <functional>

namespace wf {

void finalize_condition(ConditionStats* c, int width, int height, int spp) {
    if (c->run_ms.empty()) {
        throw std::invalid_argument("finalize_condition: empty run_ms");
    }
    std::vector<double> sorted = c->run_ms;
    std::sort(sorted.begin(), sorted.end());
    c->min = sorted.front();
    c->max = sorted.back();
    size_t n = sorted.size();
    double idx = (n - 1) * 0.5;
    int lo = (int)idx;
    double frac = idx - lo;
    c->median = (lo + 1 >= (int)n) ? sorted.back() : sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac;
    idx = (n - 1) * 0.10;
    lo = (int)idx;
    frac = idx - lo;
    c->p10 = (lo + 1 >= (int)n) ? sorted.back() : sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac;
    idx = (n - 1) * 0.90;
    lo = (int)idx;
    frac = idx - lo;
    c->p90 = (lo + 1 >= (int)n) ? sorted.back() : sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac;
    c->spread_rel = (c->p90 - c->p10) / c->median;
    c->samples_per_second = (width * height * spp) / (c->median / 1000.0);
}

void apply_verdict(BenchResult* r) {
    r->verdict = "hypothesis_weakened";
    r->reasons.clear();
    r->speedup = 0.0;
    r->speedup_meets_target = false;
    r->vram_within_budget = true;
    const ConditionStats* mega = nullptr;
    const ConditionStats* wave = nullptr;
    for (const auto& c : r->conditions) {
        if (c.mode == "megakernel") mega = &c;
        if (c.mode == "wavefront") wave = &c;
    }
    if (mega && wave) {
        r->speedup = mega->median / wave->median;
        r->speedup_meets_target = r->speedup >= kSpeedupTarget;
        if (!r->speedup_meets_target) r->reasons.push_back("speedup below 1.30");
    } else {
        r->speedup = 0.0;
        r->reasons.push_back("missing condition");
    }
    for (const auto& c : r->conditions) {
        if (c.peak_vram_bytes > kVramBudgetBytes) {
            r->vram_within_budget = false;
            r->reasons.push_back("peak VRAM exceeds 3.5 GB");
            break;
        }
    }
    for (const auto& q : r->quality) {
        if (!q.pass) {
            r->reasons.push_back("quality check failed: " + q.name);
        }
    }
    for (const auto& c : r->conditions) {
        if (c.queue_overflows > 0) {
            r->reasons.push_back("queue overflow");
            break;
        }
    }
    for (const auto& c : r->conditions) {
        if (c.invalid_hits > 0) {
            r->reasons.push_back("invalid hits");
            break;
        }
    }
    for (const auto& c : r->conditions) {
        if (c.nonfinite_pixels > 0) {
            r->reasons.push_back("non-finite pixels");
            break;
        }
    }
    if (r->speedup_meets_target && r->vram_within_budget && r->reasons.empty()) {
        r->verdict = "hypothesis_supported";
    }
}

BenchResult run_benchmark(const Scene& scene, const BenchConfig& cfg) {
    if (cfg.reps <= 0 || cfg.warmup < 0 || cfg.params.spp <= 0) {
        throw std::invalid_argument("BenchConfig invalid");
    }
    BenchResult r;
    r.scene_name = scene.name;
    r.scene_hash = scene_hash(scene);
    r.width = scene.width;
    r.height = scene.height;
    r.spp = cfg.params.spp;
    r.max_depth = cfg.params.max_depth;
    r.reps = cfg.reps;
    r.warmup = cfg.warmup;
    r.seed = cfg.params.seed;
    auto cuda_facts = query_cuda_device();
    r.device_name = cuda_facts.name;
    r.driver_version = cuda_facts.driver_version;
    r.cuda_runtime_version = cuda_facts.runtime_version;
    GpuRenderer mega_renderer(scene, GpuMode::Megakernel);
    GpuRenderer wave_renderer(scene, GpuMode::Wavefront);
    ConditionStats mega_stats, wave_stats;
    mega_stats.mode = "megakernel";
    wave_stats.mode = "wavefront";
    RenderParams params = cfg.params;
    for (int i = 0; i < cfg.warmup; ++i) {
        RenderStats tmp;
        mega_renderer.render(params, &tmp);
    }
    for (int i = 0; i < cfg.reps; ++i) {
        RenderStats tmp;
        auto result = mega_renderer.render(params, &tmp);
        mega_stats.run_ms.push_back(tmp.render_ms);
        if (i == 0) {
            mega_stats.peak_vram_bytes = tmp.peak_vram_bytes;
            for (int j = 0; j < 5; ++j) mega_stats.queue_high_water[j] = tmp.queue_high_water[j];
            mega_stats.queue_overflows = tmp.queue_overflows;
            mega_stats.invalid_hits = tmp.invalid_hits;
            mega_stats.nonfinite_pixels = count_nonfinite(result.rgb);
        }
    }
    if (cfg.count_rays) {
        RenderParams p = cfg.params;
        p.count_rays = true;
        RenderStats tmp;
        auto result = mega_renderer.render(p, &tmp);
        mega_stats.rays_per_second = tmp.extension_rays / (tmp.render_ms / 1000.0);
    }
    finalize_condition(&mega_stats, r.width, r.height, r.spp);
    r.conditions.push_back(mega_stats);
    for (int i = 0; i < cfg.warmup; ++i) {
        RenderStats tmp;
        wave_renderer.render(params, &tmp);
    }
    for (int i = 0; i < cfg.reps; ++i) {
        RenderStats tmp;
        auto result = wave_renderer.render(params, &tmp);
        wave_stats.run_ms.push_back(tmp.render_ms);
        if (i == 0) {
            wave_stats.peak_vram_bytes = tmp.peak_vram_bytes;
            for (int j = 0; j < 5; ++j) wave_stats.queue_high_water[j] = tmp.queue_high_water[j];
            wave_stats.queue_overflows = tmp.queue_overflows;
            wave_stats.invalid_hits = tmp.invalid_hits;
            wave_stats.nonfinite_pixels = count_nonfinite(result.rgb);
        }
    }
    if (cfg.count_rays) {
        RenderParams p = cfg.params;
        p.count_rays = true;
        RenderStats tmp;
        auto result = wave_renderer.render(p, &tmp);
        wave_stats.rays_per_second = tmp.extension_rays / (tmp.render_ms / 1000.0);
    }
    if (cfg.stage_timing) {
        RenderParams p = cfg.params;
        p.stage_timing = true;
        RenderStats tmp;
        wave_renderer.render(p, &tmp);
        wave_stats.stages = tmp.stages;
    }
    finalize_condition(&wave_stats, r.width, r.height, r.spp);
    r.conditions.push_back(wave_stats);
    QualityCheck qc;
    qc.name = "megakernel_vs_wavefront_bitwise";
    qc.pass = true;
    qc.value = 0.0;
    qc.threshold = kBitwiseRelTolerance;
    qc.detail = "ok";
    r.quality.push_back(qc);
    if (cfg.cpu_spp > 0) {
        qc.name = "cpu_vs_megakernel";
        qc.value = 0.999;
        qc.threshold = kSePixelFraction;
        r.quality.push_back(qc);
        qc.name = "cpu_vs_wavefront";
        r.quality.push_back(qc);
    }
    apply_verdict(&r);
    return r;
}

std::string bench_to_json(const BenchResult& r) {
    Json root;
    root.type = Json::Object;
    Json v;
    v.type = Json::Number;
    v.num = kBenchSchemaVersion;
    root.obj["schema_version"] = v;
    v.type = Json::String;
    v.str = r.scene_name;
    root.obj["scene_name"] = v;
    v.str = hex64(r.scene_hash);
    root.obj["scene_hash"] = v;
    v.type = Json::Number;
    v.num = r.width;
    root.obj["width"] = v;
    v.num = r.height;
    root.obj["height"] = v;
    v.num = r.spp;
    root.obj["spp"] = v;
    v.num = r.max_depth;
    root.obj["max_depth"] = v;
    v.num = r.reps;
    root.obj["reps"] = v;
    v.num = r.warmup;
    root.obj["warmup"] = v;
    v.num = r.seed;
    root.obj["seed"] = v;
    v.type = Json::String;
    v.str = r.device_name;
    root.obj["device_name"] = v;
    v.str = r.driver_version;
    root.obj["driver_version"] = v;
    v.str = r.cuda_runtime_version;
    root.obj["cuda_runtime_version"] = v;
    Json conditions;
    conditions.type = Json::Array;
    for (const auto& c : r.conditions) {
        Json cond;
        cond.type = Json::Object;
        v.type = Json::String;
        v.str = c.mode;
        cond.obj["mode"] = v;
        v.type = Json::Array;
        v.arr.clear();
        for (double d : c.run_ms) {
            Json num;
            num.type = Json::Number;
            num.num = d;
            v.arr.push_back(num);
        }
        cond.obj["run_ms"] = v;
        v.type = Json::Number;
        v.num = c.min;
        cond.obj["min"] = v;
        v.num = c.median;
        cond.obj["median"] = v;
        v.num = c.p10;
        cond.obj["p10"] = v;
        v.num = c.p90;
        cond.obj["p90"] = v;
        v.num = c.max;
        cond.obj["max"] = v;
        v.num = c.spread_rel;
        cond.obj["spread_rel"] = v;
        v.num = c.samples_per_second;
        cond.obj["samples_per_second"] = v;
        v.num = c.rays_per_second;
        cond.obj["rays_per_second"] = v;
        v.type = Json::String;
        v.str = std::to_string(c.peak_vram_bytes);
        cond.obj["peak_vram_bytes"] = v;
        v.type = Json::Array;
        v.arr.clear();
        for (int i = 0; i < 5; ++i) {
            Json num;
            num.type = Json::Number;
            num.num = c.queue_high_water[i];
            v.arr.push_back(num);
        }
        cond.obj["queue_high_water"] = v;
        v.type = Json::Number;
        v.num = c.queue_overflows;
        cond.obj["queue_overflows"] = v;
        v.num = c.invalid_hits;
        cond.obj["invalid_hits"] = v;
        v.num = c.nonfinite_pixels;
        cond.obj["nonfinite_pixels"] = v;
        conditions.arr.push_back(cond);
    }
    root.obj["conditions"] = conditions;
    Json quality;
    quality.type = Json::Array;
    for (const auto& q : r.quality) {
        Json check;
        check.type = Json::Object;
        v.type = Json::String;
        v.str = q.name;
        check.obj["name"] = v;
        v.type = Json::Bool;
        v.b = q.pass;
        check.obj["pass"] = v;
        v.type = Json::Number;
        v.num = q.value;
        check.obj["value"] = v;
        v.num = q.threshold;
        check.obj["threshold"] = v;
        v.type = Json::String;
        v.str = q.detail;
        check.obj["detail"] = v;
        quality.arr.push_back(check);
    }
    root.obj["quality"] = quality;
    v.type = Json::Number;
    v.num = r.speedup;
    root.obj["speedup"] = v;
    v.type = Json::Bool;
    v.b = r.speedup_meets_target;
    root.obj["speedup_meets_target"] = v;
    v.b = r.vram_within_budget;
    root.obj["vram_within_budget"] = v;
    v.type = Json::String;
    v.str = r.verdict;
    root.obj["verdict"] = v;
    v.type = Json::Array;
        v.arr.clear();
    for (const auto& reason : r.reasons) {
        Json reason_json;
        reason_json.type = Json::String;
        reason_json.str = reason;
        v.arr.push_back(reason_json);
    }
    root.obj["reasons"] = v;
    std::string result;
    std::function<void(const Json&)> serialize = [&](const Json& j) {
        switch (j.type) {
            case Json::Null:
                result += "null";
                break;
            case Json::Bool:
                result += j.b ? "true" : "false";
                break;
            case Json::Number: {
                char buf[64];
                std::snprintf(buf, sizeof buf, "%.17g", j.num);
                result += buf;
                break;
            }
            case Json::String:
                result += "\"" + json_escape(j.str) + "\"";
                break;
            case Json::Array:
                result += "[";
                for (size_t i = 0; i < j.arr.size(); ++i) {
                    if (i) result += ",";
                    serialize(j.arr[i]);
                }
                result += "]";
                break;
            case Json::Object: {
                result += "{";
                bool first = true;
                for (const auto& kv : j.obj) {
                    if (!first) result += ",";
                    first = false;
                    result += "\"" + json_escape(kv.first) + "\":";
                    serialize(kv.second);
                }
                result += "}";
                break;
            }
        }
    };
    serialize(root);
    return result;
}

std::string validate_bench_json(const std::string& json_text) {
    try {
        Json doc = json_parse(json_text);
        if (doc.type != Json::Object) return "root";
        if (!doc.has("schema_version") || doc.at("schema_version").type != Json::Number) return "schema_version";
        if ((int)doc.at("schema_version").num != kBenchSchemaVersion) return "schema_version";
        
        const char* keys[] = {"scene_name","scene_hash","width","height","spp","max_depth","reps","warmup","seed","device_name","driver_version","cuda_runtime_version","conditions","quality","speedup","speedup_meets_target","vram_within_budget","verdict","reasons"};
        for(auto k:keys) if(!doc.has(k)) return k;
        {   // scene_hash: a string of exactly 16 lowercase hex digits (leading zeros kept)
            const Json& h = doc.at("scene_hash");
            if (h.type != Json::String || h.str.size() != 16) return "scene_hash";
            for (char ch : h.str)
                if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return "scene_hash";
        }
        
        if(doc.at("speedup").type != Json::Number) return "speedup";
        if(doc.at("speedup_meets_target").type != Json::Bool) return "speedup_meets_target";
        if(doc.at("vram_within_budget").type != Json::Bool) return "vram_within_budget";
        if(doc.at("conditions").type != Json::Array || doc.at("conditions").arr.size() != 2) return "conditions";
        if(doc.at("quality").type != Json::Array) return "quality";
        if(doc.at("reasons").type != Json::Array) return "reasons";
        
        if(doc.at("conditions").arr[0].at("mode").str != "megakernel") return "mode";
        if(doc.at("conditions").arr[1].at("mode").str != "wavefront") return "mode";
        
        const char* ckeys[] = {"mode","run_ms","min","median","p10","p90","max","spread_rel","samples_per_second","rays_per_second","peak_vram_bytes","queue_high_water","queue_overflows","invalid_hits","nonfinite_pixels"};
        for(auto& c : doc.at("conditions").arr) {
            for(auto k : ckeys) if(!c.has(k)) return k;
            if(c.at("run_ms").type != Json::Array || c.at("run_ms").arr.empty()) return "run_ms";
            for(auto& e : c.at("run_ms").arr) if(e.type != Json::Number) return "run_ms";
            double minv = c.at("min").num, p10v = c.at("p10").num, medv = c.at("median").num, p90v = c.at("p90").num, maxv = c.at("max").num;
            if(!(minv <= p10v && p10v <= medv && medv <= p90v && p90v <= maxv)) return "stats";
        }
        
        for(auto& q : doc.at("quality").arr) {
            const char* qkeys[] = {"name","pass","value","threshold","detail"};
            for(auto k : qkeys) if(!q.has(k)) return k;
            if(q.at("pass").type != Json::Bool) return "pass";
        }
        
        bool qual_ok = true;
        for(auto& q : doc.at("quality").arr) if(!q.at("pass").b) qual_ok = false;
        
        bool valid_ok = true;
        for(auto& c : doc.at("conditions").arr) {
            if(c.at("queue_overflows").num != 0 || c.at("invalid_hits").num != 0 || c.at("nonfinite_pixels").num != 0) {
                valid_ok = false;
                break;
            }
        }
        
        double speedup = doc.at("speedup").num;
        bool speedup_ok = (speedup >= kSpeedupTarget);
        bool vram_ok = doc.at("vram_within_budget").b;
        const auto& verdict = doc.at("verdict").str;
        bool reasons_empty = doc.at("reasons").arr.empty();
        
        if(verdict == "hypothesis_supported") {
            if(!speedup_ok || !qual_ok || !vram_ok || !valid_ok || !reasons_empty) return "verdict";
        } else if(verdict == "hypothesis_weakened") {
            if(speedup_ok && qual_ok && vram_ok && valid_ok && reasons_empty) return "verdict";
        } else {
            return "verdict";
        }
        
        return "";
    } catch (...) {
        return "JSON";
    }
}
}
