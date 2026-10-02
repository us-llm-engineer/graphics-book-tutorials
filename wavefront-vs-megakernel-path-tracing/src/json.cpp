// Tiny JSON value: parse and dump.
#include "wf/json.hpp"
#include <cctype>
#include <sstream>

namespace wf {

class JsonParser {
    const std::string& text;
    size_t pos = 0;

    void skip_whitespace() {
        while (pos < text.size() && std::isspace((unsigned char)text[pos])) {
            ++pos;
        }
    }

    void expect(char c) {
        skip_whitespace();
        if (pos >= text.size() || text[pos] != c) {
            throw std::runtime_error("JSON parse error at position " + std::to_string(pos));
        }
        ++pos;
    }

    char peek() {
        skip_whitespace();
        return pos < text.size() ? text[pos] : '\0';
    }

    std::string parse_string() {
        expect('"');
        std::string s;
        while (pos < text.size() && text[pos] != '"') {
            if (text[pos] == '\\' && pos + 1 < text.size()) {
                ++pos;
                switch (text[pos]) {
                    case '"': s += '"'; break;
                    case '\\': s += '\\'; break;
                    case '/': s += '/'; break;
                    case 'b': s += '\b'; break;
                    case 'f': s += '\f'; break;
                    case 'n': s += '\n'; break;
                    case 'r': s += '\r'; break;
                    case 't': s += '\t'; break;
                    case 'u': {
                        if (pos + 4 >= text.size()) throw std::runtime_error("JSON: truncated \\uXXXX");
                        std::string hex = text.substr(pos + 1, 4);
                        unsigned codepoint = std::stoul(hex, nullptr, 16);
                        if (codepoint < 128) {
                            s += (char)codepoint;
                        } else if (codepoint < 2048) {
                            s += (char)(0xC0 | (codepoint >> 6));
                            s += (char)(0x80 | (codepoint & 0x3F));
                        } else {
                            s += (char)(0xE0 | (codepoint >> 12));
                            s += (char)(0x80 | ((codepoint >> 6) & 0x3F));
                            s += (char)(0x80 | (codepoint & 0x3F));
                        }
                        pos += 4;
                        break;
                    }
                    default: throw std::runtime_error("JSON: unknown escape sequence");
                }
            } else {
                s += text[pos];
            }
            ++pos;
        }
        expect('"');
        return s;
    }

    Json parse_value() {
        skip_whitespace();
        if (pos >= text.size()) {
            throw std::runtime_error("JSON parse error: unexpected end");
        }

        char c = text[pos];

        if (c == 'n') {
            if (text.substr(pos, 4) == "null") {
                pos += 4;
                Json j;
                j.type = Json::Null;
                return j;
            }
        } else if (c == 't' || c == 'f') {
            if (text.substr(pos, 4) == "true") {
                pos += 4;
                Json j;
                j.type = Json::Bool;
                j.b = true;
                return j;
            } else if (text.substr(pos, 5) == "false") {
                pos += 5;
                Json j;
                j.type = Json::Bool;
                j.b = false;
                return j;
            }
        } else if (c == '"') {
            Json j;
            j.type = Json::String;
            j.str = parse_string();
            return j;
        } else if (c == '[') {
            ++pos;
            Json j;
            j.type = Json::Array;
            if (peek() != ']') {
                while (true) {
                    j.arr.push_back(parse_value());
                    if (peek() == ']') break;
                    expect(',');
                }
            }
            expect(']');
            return j;
        } else if (c == '{') {
            ++pos;
            Json j;
            j.type = Json::Object;
            if (peek() != '}') {
                while (true) {
                    std::string key = parse_string();
                    expect(':');
                    j.obj[key] = parse_value();
                    if (peek() == '}') break;
                    expect(',');
                }
            }
            expect('}');
            return j;
        } else if (c == '-' || std::isdigit((unsigned char)c)) {
            size_t start = pos;
            if (c == '-') ++pos;
            if (pos >= text.size() || !std::isdigit((unsigned char)text[pos])) {
                throw std::runtime_error("JSON parse error: invalid number");
            }
            while (pos < text.size() && std::isdigit((unsigned char)text[pos])) ++pos;
            if (pos < text.size() && text[pos] == '.') {
                ++pos;
                if (pos >= text.size() || !std::isdigit((unsigned char)text[pos])) {
                    throw std::runtime_error("JSON parse error: invalid number");
                }
                while (pos < text.size() && std::isdigit((unsigned char)text[pos])) ++pos;
            }
            if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
                ++pos;
                if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) ++pos;
                if (pos >= text.size() || !std::isdigit((unsigned char)text[pos])) {
                    throw std::runtime_error("JSON parse error: invalid number");
                }
                while (pos < text.size() && std::isdigit((unsigned char)text[pos])) ++pos;
            }
            Json j;
            j.type = Json::Number;
            j.num = std::stod(text.substr(start, pos - start));
            return j;
        }

        throw std::runtime_error("JSON parse error at position " + std::to_string(pos));
    }

public:
    JsonParser(const std::string& t) : text(t) {}

    Json parse() {
        Json j = parse_value();
        skip_whitespace();
        if (pos < text.size()) {
            throw std::runtime_error("JSON parse error: trailing garbage at position " + std::to_string(pos));
        }
        return j;
    }
};

Json json_parse(const std::string& text) {
    return JsonParser(text).parse();
}

std::string json_escape(const std::string& s) {
    std::string result;
    for (unsigned char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (c < 32) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

}  // namespace wf
