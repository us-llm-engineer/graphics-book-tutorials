// Tiny JSON value: parse and dump, enough for report.json validation and tests.
#pragma once
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace wf {

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    bool has(const std::string& k) const { return type == Object && obj.count(k) > 0; }
    const Json& at(const std::string& k) const { return obj.at(k); }
};

// Throws std::runtime_error with position info on malformed input (trailing garbage included).
Json json_parse(const std::string& text);
// Escapes for use inside a JSON string: returns the escaped body WITHOUT surrounding quotes.
// \uXXXX escapes in json_parse decode to UTF-8.
std::string json_escape(const std::string& s);

}  // namespace wf
