// Versioned report.json writer and validator.
#include "wf/report.hpp"
#include "wf/json.hpp"
#include "wf/image_io.hpp"
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>

namespace wf {

double percentile(std::vector<double> v, double p) {
    if (v.empty()) {
        throw std::runtime_error("percentile: empty input");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::runtime_error("percentile: p outside [0,1]");
    }

    std::sort(v.begin(), v.end());

    double idx = p * (v.size() - 1);
    int lo = (int)idx;
    int hi = lo + 1;
    double frac = idx - lo;

    if (hi >= (int)v.size()) {
        return v.back();
    }
    if (lo < 0) {
        return v.front();
    }

    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

std::string report_to_json(const Report& r) {
    Json report;
    report.type = Json::Object;

    // schema_version
    {
        Json sv;
        sv.type = Json::Number;
        sv.num = kReportSchemaVersion;
        report.obj["schema_version"] = sv;
    }

    // git_revision
    {
        Json grev;
        if (r.git_revision.empty()) {
            grev.type = Json::Null;
        } else {
            grev.type = Json::String;
            grev.str = r.git_revision;
        }
        report.obj["git_revision"] = grev;
    }

    // scene
    {
        Json scene;
        scene.type = Json::Object;

        Json sn;
        sn.type = Json::String;
        sn.str = r.scene_name;
        scene.obj["name"] = sn;

        Json sh;
        sh.type = Json::String;
        sh.str = hex64(r.scene_hash);
        scene.obj["hash"] = sh;

        report.obj["scene"] = scene;
    }

    // config
    {
        Json config;
        config.type = Json::Object;

        Json w;
        w.type = Json::Number;
        w.num = r.width;
        config.obj["width"] = w;

        Json h;
        h.type = Json::Number;
        h.num = r.height;
        config.obj["height"] = h;

        Json spp;
        spp.type = Json::Number;
        spp.num = r.spp;
        config.obj["spp"] = spp;

        Json seed;
        seed.type = Json::Number;
        seed.num = r.seed;
        config.obj["seed"] = seed;

        Json maxd;
        maxd.type = Json::Number;
        maxd.num = r.max_depth;
        config.obj["max_depth"] = maxd;

        Json mode;
        mode.type = Json::String;
        mode.str = r.mode;
        config.obj["mode"] = mode;

        report.obj["config"] = config;
    }

    // device
    {
        Json device;
        device.type = Json::Object;

        Json name;
        name.type = Json::String;
        name.str = r.device.name;
        device.obj["name"] = name;

        Json driver;
        driver.type = Json::String;
        driver.str = r.device.driver_version;
        device.obj["driver_version"] = driver;

        Json cuda_rt;
        cuda_rt.type = Json::String;
        cuda_rt.str = r.device.cuda_runtime_version;
        device.obj["cuda_runtime_version"] = cuda_rt;

        Json cuda_drv;
        cuda_drv.type = Json::String;
        cuda_drv.str = r.device.cuda_driver_version;
        device.obj["cuda_driver_version"] = cuda_drv;

        Json cc;
        cc.type = Json::String;
        cc.str = r.device.compute_capability;
        device.obj["compute_capability"] = cc;

        Json mem;
        mem.type = Json::Number;
        mem.num = r.device.total_memory_bytes;
        device.obj["total_memory_bytes"] = mem;

        report.obj["device"] = device;
    }

    // timings_ms
    {
        Json timings;
        timings.type = Json::Object;

        Json runs;
        runs.type = Json::Array;
        for (double t : r.run_ms) {
            Json rt;
            rt.type = Json::Number;
            rt.num = t;
            runs.arr.push_back(rt);
        }
        timings.obj["runs"] = runs;

        if (!r.run_ms.empty()) {
            double minv = *std::min_element(r.run_ms.begin(), r.run_ms.end());
            double medianv = percentile(r.run_ms, 0.5);
            double p10v = percentile(r.run_ms, 0.1);
            double p90v = percentile(r.run_ms, 0.9);
            double maxv = *std::max_element(r.run_ms.begin(), r.run_ms.end());

            Json minj;
            minj.type = Json::Number;
            minj.num = minv;
            timings.obj["min"] = minj;

            Json medj;
            medj.type = Json::Number;
            medj.num = medianv;
            timings.obj["median"] = medj;

            Json p10j;
            p10j.type = Json::Number;
            p10j.num = p10v;
            timings.obj["p10"] = p10j;

            Json p90j;
            p90j.type = Json::Number;
            p90j.num = p90v;
            timings.obj["p90"] = p90j;

            Json maxj;
            maxj.type = Json::Number;
            maxj.num = maxv;
            timings.obj["max"] = maxj;
        }

        report.obj["timings_ms"] = timings;
    }

    // throughput
    {
        Json throughput;
        throughput.type = Json::Object;

        Json sps;
        sps.type = Json::Number;
        sps.num = r.samples_per_second;
        throughput.obj["samples_per_second"] = sps;

        Json rps;
        if (r.rays_per_second > 0.0) {
            rps.type = Json::Number;
            rps.num = r.rays_per_second;
        } else {
            rps.type = Json::Null;
        }
        throughput.obj["rays_per_second"] = rps;

        report.obj["throughput"] = throughput;
    }

    // peak_vram_bytes
    {
        Json pvram;
        pvram.type = Json::Number;
        pvram.num = r.peak_vram_bytes;
        report.obj["peak_vram_bytes"] = pvram;
    }

    // queue_high_water
    {
        Json qhw;
        qhw.type = Json::Object;

        const char* names[] = {"lambert", "metal", "dielectric", "shadow", "active"};
        for (int i = 0; i < 5; ++i) {
            Json qval;
            qval.type = Json::Number;
            qval.num = r.queue_high_water[i];
            qhw.obj[names[i]] = qval;
        }

        report.obj["queue_high_water"] = qhw;
    }

    // outputs
    {
        Json outputs;
        outputs.type = Json::Object;

        Json png;
        png.type = Json::String;
        png.str = r.image_png;
        outputs.obj["png"] = png;

        Json pfm;
        pfm.type = Json::String;
        pfm.str = r.image_pfm;
        outputs.obj["pfm"] = pfm;

        Json phash;
        phash.type = Json::String;
        phash.str = hex64(r.pfm_hash);
        outputs.obj["pfm_hash"] = phash;

        report.obj["outputs"] = outputs;
    }

    // validity
    {
        Json validity;
        validity.type = Json::Object;

        Json nf;
        nf.type = Json::Number;
        nf.num = r.nonfinite_pixels;
        validity.obj["nonfinite_pixels"] = nf;

        Json qo;
        qo.type = Json::Number;
        qo.num = r.queue_overflows;
        validity.obj["queue_overflows"] = qo;

        Json ih;
        ih.type = Json::Number;
        ih.num = r.invalid_hits;
        validity.obj["invalid_hits"] = ih;

        report.obj["validity"] = validity;
    }

    // Serialize to text (simplified)
    std::ostringstream ss;
    std::function<void(const Json&, int)> serialize = [&](const Json& j, int depth) {
        std::string indent(depth * 2, ' ');
        switch (j.type) {
            case Json::Null:
                ss << "null";
                break;
            case Json::Bool:
                ss << (j.b ? "true" : "false");
                break;
            case Json::Number: {
                // Format number
                if (j.num == (long long)j.num) {
                    ss << (long long)j.num;
                } else {
                    ss << std::fixed << std::setprecision(10) << j.num;
                }
                break;
            }
            case Json::String:
                ss << "\"" << json_escape(j.str) << "\"";
                break;
            case Json::Array:
                ss << "[";
                for (size_t i = 0; i < j.arr.size(); ++i) {
                    if (i > 0) ss << ", ";
                    serialize(j.arr[i], 0);
                }
                ss << "]";
                break;
            case Json::Object: {
                ss << "{";
                bool first = true;
                for (const auto& kv : j.obj) {
                    if (!first) ss << ",";
                    ss << " \"" << json_escape(kv.first) << "\": ";
                    serialize(kv.second, 0);
                    first = false;
                }
                ss << " }";
                break;
            }
        }
    };

    serialize(report, 0);
    return ss.str();
}

std::string validate_report_json(const std::string& json_text) {
    try {
        Json j = json_parse(json_text);

        if (j.type != Json::Object) {
            return "root is not an object";
        }

        // Check schema_version
        if (!j.has("schema_version")) return "missing schema_version";
        const Json& sv = j.at("schema_version");
        if (sv.type != Json::Number) return "schema_version is not a number";
        if ((int)sv.num != kReportSchemaVersion) return "schema_version mismatch";

        // Check git_revision
        if (!j.has("git_revision")) return "missing git_revision";
        const Json& gr = j.at("git_revision");
        if (gr.type != Json::Null && gr.type != Json::String) return "git_revision is not null or string";

        // Check scene
        if (!j.has("scene")) return "missing scene";
        const Json& scene = j.at("scene");
        if (scene.type != Json::Object) return "scene is not an object";
        if (!scene.has("name")) return "scene missing name";
        if (scene.at("name").type != Json::String) return "scene.name is not a string";
        if (!scene.has("hash")) return "scene missing hash";
        if (scene.at("hash").type != Json::String) return "scene.hash is not a string";
        std::string hash_str = scene.at("hash").str;
        if (hash_str.size() != 16) return "scene.hash is not 16 hex digits";
        for (char c : hash_str) {
            if (!std::isxdigit((unsigned char)c)) return "scene.hash contains non-hex digit";
        }

        // Check config
        if (!j.has("config")) return "missing config";
        const Json& config = j.at("config");
        if (config.type != Json::Object) return "config is not an object";
        for (const char* key : {"width", "height", "spp", "seed", "max_depth"}) {
            if (!config.has(key)) return std::string("config missing ") + key;
            if (config.at(key).type != Json::Number) return std::string("config.") + key + " is not a number";
        }
        if (!config.has("mode")) return "config missing mode";
        if (config.at("mode").type != Json::String) return "config.mode is not a string";

        // Check device
        if (!j.has("device")) return "missing device";
        const Json& device = j.at("device");
        if (device.type != Json::Object) return "device is not an object";
        for (const char* key : {"name", "driver_version", "cuda_runtime_version", "cuda_driver_version", "compute_capability"}) {
            if (!device.has(key)) return std::string("device missing ") + key;
            if (device.at(key).type != Json::String) return std::string("device.") + key + " is not a string";
        }
        if (!device.has("total_memory_bytes")) return "device missing total_memory_bytes";
        if (device.at("total_memory_bytes").type != Json::Number) return "device.total_memory_bytes is not a number";

        // Check timings_ms
        if (!j.has("timings_ms")) return "missing timings_ms";
        const Json& timings = j.at("timings_ms");
        if (timings.type != Json::Object) return "timings_ms is not an object";
        if (!timings.has("runs")) return "timings_ms missing runs";
        if (timings.at("runs").type != Json::Array) return "timings_ms.runs is not an array";
        if (timings.at("runs").arr.empty()) return "timings_ms.runs is empty";
        for (const auto& run : timings.at("runs").arr) {
            if (run.type != Json::Number) return "timings_ms.runs contains non-number";
        }
        for (const char* key : {"min", "median", "p10", "p90", "max"}) {
            if (!timings.has(key)) return std::string("timings_ms missing ") + key;
            if (timings.at(key).type != Json::Number) return std::string("timings_ms.") + key + " is not a number";
        }
        // Check min <= median <= max
        double minv = timings.at("min").num;
        double medv = timings.at("median").num;
        double maxv = timings.at("max").num;
        if (!(minv <= medv && medv <= maxv)) return "timings_ms: min <= median <= max not satisfied";

        // Check throughput
        if (!j.has("throughput")) return "missing throughput";
        const Json& throughput = j.at("throughput");
        if (throughput.type != Json::Object) return "throughput is not an object";
        if (!throughput.has("samples_per_second")) return "throughput missing samples_per_second";
        if (throughput.at("samples_per_second").type != Json::Number) return "throughput.samples_per_second is not a number";
        if (!throughput.has("rays_per_second")) return "throughput missing rays_per_second";
        const Json& rps = throughput.at("rays_per_second");
        if (rps.type != Json::Null && rps.type != Json::Number) return "throughput.rays_per_second is not null or number";

        // Check peak_vram_bytes
        if (!j.has("peak_vram_bytes")) return "missing peak_vram_bytes";
        if (j.at("peak_vram_bytes").type != Json::Number) return "peak_vram_bytes is not a number";

        // Check queue_high_water
        if (!j.has("queue_high_water")) return "missing queue_high_water";
        const Json& qhw = j.at("queue_high_water");
        if (qhw.type != Json::Object) return "queue_high_water is not an object";
        for (const char* key : {"lambert", "metal", "dielectric", "shadow", "active"}) {
            if (!qhw.has(key)) return std::string("queue_high_water missing ") + key;
            if (qhw.at(key).type != Json::Number) return std::string("queue_high_water.") + key + " is not a number";
        }

        // Check outputs
        if (!j.has("outputs")) return "missing outputs";
        const Json& outputs = j.at("outputs");
        if (outputs.type != Json::Object) return "outputs is not an object";
        if (!outputs.has("png")) return "outputs missing png";
        if (outputs.at("png").type != Json::String) return "outputs.png is not a string";
        if (!outputs.has("pfm")) return "outputs missing pfm";
        if (outputs.at("pfm").type != Json::String) return "outputs.pfm is not a string";
        if (!outputs.has("pfm_hash")) return "outputs missing pfm_hash";
        if (outputs.at("pfm_hash").type != Json::String) return "outputs.pfm_hash is not a string";
        std::string pfm_hash_str = outputs.at("pfm_hash").str;
        if (pfm_hash_str.size() != 16) return "outputs.pfm_hash is not 16 hex digits";
        for (char c : pfm_hash_str) {
            if (!std::isxdigit((unsigned char)c)) return "outputs.pfm_hash contains non-hex digit";
        }

        // Check validity
        if (!j.has("validity")) return "missing validity";
        const Json& validity = j.at("validity");
        if (validity.type != Json::Object) return "validity is not an object";
        for (const char* key : {"nonfinite_pixels", "queue_overflows", "invalid_hits"}) {
            if (!validity.has(key)) return std::string("validity missing ") + key;
            if (validity.at(key).type != Json::Number) return std::string("validity.") + key + " is not a number";
        }

        return "";  // Valid
    } catch (const std::exception& e) {
        return std::string("JSON parse error: ") + e.what();
    }
}

}  // namespace wf
