#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct CsvRow {
    std::vector<std::string> fields;
};

CsvRow parse_csv_line(const std::string& line) {
    CsvRow row;
    std::string field;
    bool in_quotes = false;
    size_t i = 0;
    while (i < line.size()) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                i += 2;
            } else {
                in_quotes = !in_quotes;
                i++;
            }
        } else if (c == ',' && !in_quotes) {
            row.fields.push_back(field);
            field.clear();
            i++;
        } else {
            field += c;
            i++;
        }
    }
    row.fields.push_back(field);
    return row;
}

std::string trim_cr(const std::string& s) {
    if (!s.empty() && s.back() == '\r') {
        return s.substr(0, s.size() - 1);
    }
    return s;
}

double parse_number(const std::string& s) {
    std::string clean;
    for (char c : s) {
        if (c != ',') clean += c;
    }
    return std::stod(clean);
}

bool is_null_value(const std::string& s) {
    std::string lower = s;
    for (char& c : lower) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }
    return lower == "n/a" || s.empty();
}

struct JsonValue {
    enum Type { Null, Number, String, Array, Object } type;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;

    JsonValue() : type(Null) {}
    JsonValue(double n) : type(Number), num_val(n) {}
    JsonValue(const std::string& s) : type(String), str_val(s) {}
};

std::string json_escape(const std::string& s) {
    std::string result;
    for (unsigned char c : s) {
        if (c == '"') result += "\\\"";
        else if (c == '\\') result += "\\\\";
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else if (c < 32) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", (int)c);
            result += buf;
        } else {
            result += c;
        }
    }
    return result;
}

std::string json_dump(const JsonValue& v);

std::string json_dump(const JsonValue& v) {
    std::ostringstream os;
    switch (v.type) {
        case JsonValue::Null:
            os << "null";
            break;
        case JsonValue::Number: {
            os << std::fixed << std::setprecision(17) << v.num_val;
            std::string s = os.str();
            size_t dot = s.find('.');
            if (dot != std::string::npos) {
                while (s.back() == '0') s.pop_back();
                if (s.back() == '.') s.pop_back();
            }
            return s;
        }
        case JsonValue::String:
            return "\"" + json_escape(v.str_val) + "\"";
        case JsonValue::Array: {
            os << "[";
            for (size_t i = 0; i < v.arr_val.size(); i++) {
                if (i > 0) os << ",";
                os << json_dump(v.arr_val[i]);
            }
            os << "]";
            return os.str();
        }
        case JsonValue::Object: {
            os << "{";
            bool first = true;
            for (const auto& kv : v.obj_val) {
                if (!first) os << ",";
                os << "\"" << json_escape(kv.first) << "\":" << json_dump(kv.second);
                first = false;
            }
            os << "}";
            return os.str();
        }
    }
    return os.str();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.csv> [--out <output.json>]\n";
        return 2;
    }

    std::string input_file = argv[1];
    std::string output_file;

    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--out" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    std::ifstream input(input_file, std::ios::binary);
    if (!input) {
        std::cerr << "Error: cannot open file " << input_file << "\n";
        return 2;
    }

    std::string file_content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();

    if (file_content.empty()) {
        std::cerr << "Error: " << input_file << " is empty\n";
        return 2;
    }

    std::vector<std::string> lines;
    std::istringstream iss(file_content);
    std::string line;
    while (std::getline(iss, line)) {
        lines.push_back(trim_cr(line));
    }

    if (lines.empty()) {
        std::cerr << "Error: " << input_file << " is empty\n";
        return 2;
    }

    CsvRow header_row = parse_csv_line(lines[0]);

    // Find required column indices
    int id_idx = -1, kernel_idx = -1, metric_idx = -1, unit_idx = -1, value_idx = -1;
    for (size_t i = 0; i < header_row.fields.size(); i++) {
        const std::string& field = header_row.fields[i];
        if (field == "ID") id_idx = i;
        else if (field == "Kernel Name") kernel_idx = i;
        else if (field == "Metric Name") metric_idx = i;
        else if (field == "Metric Unit") unit_idx = i;
        else if (field == "Metric Value") value_idx = i;
    }

    // Check for missing required columns and report the first missing one
    if (id_idx < 0) {
        std::cerr << "Error: missing required column 'ID' in header\n";
        return 2;
    }
    if (kernel_idx < 0) {
        std::cerr << "Error: missing required column 'Kernel Name' in header\n";
        return 2;
    }
    if (metric_idx < 0) {
        std::cerr << "Error: missing required column 'Metric Name' in header\n";
        return 2;
    }
    if (unit_idx < 0) {
        std::cerr << "Error: missing required column 'Metric Unit' in header\n";
        return 2;
    }
    if (value_idx < 0) {
        std::cerr << "Error: missing required column 'Metric Value' in header\n";
        return 2;
    }

    struct MetricData {
        std::string unit;
        std::vector<std::pair<std::string, double>> samples;
    };

    struct KernelData {
        std::map<std::string, MetricData> metrics;
        std::set<std::string> invocation_ids;
    };

    std::map<std::string, KernelData> kernels;

    for (size_t row_num = 1; row_num < lines.size(); row_num++) {
        const std::string& line = lines[row_num];
        if (line.empty()) continue;

        CsvRow row = parse_csv_line(line);
        if (row.fields.size() < 5) {
            std::cerr << "Error: row " << row_num + 1 << " in " << input_file << " has fewer than 5 fields\n";
            return 2;
        }

        std::string inv_id = row.fields[id_idx];
        std::string kernel_name = row.fields[kernel_idx];
        std::string metric_name = row.fields[metric_idx];
        std::string metric_unit = row.fields[unit_idx];
        std::string metric_value = row.fields[value_idx];

        KernelData& kdata = kernels[kernel_name];
        kdata.invocation_ids.insert(inv_id);

        MetricData& mdata = kdata.metrics[metric_name];
        mdata.unit = metric_unit;

        if (!is_null_value(metric_value)) {
            try {
                double val = parse_number(metric_value);
                mdata.samples.push_back({inv_id, val});
            } catch (...) {
                // Text-valued metrics (for example a cache-preference name) carry no number to summarise.
            }
        }
    }

    JsonValue result;
    result.type = JsonValue::Object;
    result.obj_val["schema_version"] = JsonValue(1.0);

    JsonValue kernels_array;
    kernels_array.type = JsonValue::Array;

    for (const auto& kv : kernels) {
        const std::string& kernel_name = kv.first;
        const KernelData& kdata = kv.second;

        JsonValue kernel_obj;
        kernel_obj.type = JsonValue::Object;
        kernel_obj.obj_val["name"] = JsonValue(kernel_name);
        kernel_obj.obj_val["invocations"] = JsonValue((double)kdata.invocation_ids.size());

        JsonValue metrics_array;
        metrics_array.type = JsonValue::Array;

        for (const auto& mv : kdata.metrics) {
            const std::string& metric_name = mv.first;
            const MetricData& mdata = mv.second;

            JsonValue metric_obj;
            metric_obj.type = JsonValue::Object;
            metric_obj.obj_val["name"] = JsonValue(metric_name);
            metric_obj.obj_val["unit"] = JsonValue(mdata.unit);

            if (mdata.samples.empty()) {
                metric_obj.obj_val["value"].type = JsonValue::Null;
                metric_obj.obj_val["samples"] = JsonValue(0.0);
            } else {
                double sum = 0.0;
                for (const auto& sample : mdata.samples) {
                    sum += sample.second;
                }
                double mean = sum / mdata.samples.size();
                metric_obj.obj_val["value"] = JsonValue(mean);
                metric_obj.obj_val["samples"] = JsonValue((double)mdata.samples.size());
            }

            metrics_array.arr_val.push_back(metric_obj);
        }

        kernel_obj.obj_val["metrics"] = metrics_array;
        kernels_array.arr_val.push_back(kernel_obj);
    }

    result.obj_val["kernels"] = kernels_array;

    JsonValue summary;
    summary.type = JsonValue::Object;

    auto compute_summary = [&kernels](const std::string& metric_name) -> JsonValue {
        std::vector<double> values;
        for (const auto& kv : kernels) {
            const KernelData& kdata = kv.second;
            auto it = kdata.metrics.find(metric_name);
            if (it != kdata.metrics.end() && !it->second.samples.empty()) {
                double sum = 0.0;
                for (const auto& sample : it->second.samples) {
                    sum += sample.second;
                }
                double mean = sum / it->second.samples.size();
                values.push_back(mean);
            }
        }
        if (values.empty()) {
            return JsonValue();
        }
        double sum = 0.0;
        for (double v : values) sum += v;
        return JsonValue(sum / values.size());
    };

    summary.obj_val["achieved_occupancy_pct"] = 
        compute_summary("sm__warps_active.avg.pct_of_peak_sustained_active");
    summary.obj_val["warp_execution_efficiency"] = 
        compute_summary("smsp__thread_inst_executed_per_inst_executed.ratio");
    summary.obj_val["registers_per_thread"] = 
        compute_summary("launch__registers_per_thread");
    summary.obj_val["dram_throughput_pct"] = 
        compute_summary("dram__throughput.avg.pct_of_peak_sustained_elapsed");
    summary.obj_val["sm_throughput_pct"] = 
        compute_summary("sm__throughput.avg.pct_of_peak_sustained_elapsed");

    result.obj_val["summary"] = summary;

    std::string json_output = json_dump(result);

    if (!output_file.empty()) {
        std::ofstream out(output_file, std::ios::binary);
        if (!out) {
            std::cerr << "Error: cannot write to " << output_file << "\n";
            return 2;
        }
        out << json_output << "\n";
        out.close();
    } else {
        std::cout << json_output << "\n";
    }

    return 0;
}
