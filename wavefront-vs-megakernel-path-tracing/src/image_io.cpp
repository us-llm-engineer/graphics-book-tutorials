// Minimal image I/O: PNG writer, PFM read/write, FNV-1a hashing.
#include "wf/image_io.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <cstdint>

namespace wf {

// CRC32 for PNG chunks
uint32_t crc32(const uint8_t* data, size_t len, uint32_t crc = 0) {
    crc = ~crc;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

// Adler32 checksum for zlib
uint32_t adler32(const uint8_t* data, size_t len) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void write_be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 24));
    v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}

void write_png(const std::string& path, int width, int height, const std::vector<uint8_t>& rgb8) {
    if ((int)rgb8.size() != 3 * width * height) {
        throw std::runtime_error("write_png: rgb8 size mismatch");
    }

    try {
        std::vector<uint8_t> file_data;

        // PNG signature
        static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
        file_data.insert(file_data.end(), sig, sig + 8);

        // IHDR chunk
        {
            std::vector<uint8_t> ihdr_data;
            write_be32(ihdr_data, (uint32_t)width);
            write_be32(ihdr_data, (uint32_t)height);
            ihdr_data.push_back(8);           // bit depth
            ihdr_data.push_back(2);           // color type (RGB)
            ihdr_data.push_back(0);           // compression (deflate)
            ihdr_data.push_back(0);           // filter (None)
            ihdr_data.push_back(0);           // interlace (none)

            std::vector<uint8_t> chunk_type = {'I', 'H', 'D', 'R'};
            write_be32(file_data, (uint32_t)ihdr_data.size());
            file_data.insert(file_data.end(), chunk_type.begin(), chunk_type.end());
            file_data.insert(file_data.end(), ihdr_data.begin(), ihdr_data.end());

            std::vector<uint8_t> crc_data = chunk_type;
            crc_data.insert(crc_data.end(), ihdr_data.begin(), ihdr_data.end());
            uint32_t crc = crc32(crc_data.data(), crc_data.size());
            write_be32(file_data, crc);
        }

        // IDAT chunk with stored deflate blocks
        {
            std::vector<uint8_t> raw;

            // Build scanlines with filter type 0 (None)
            for (int y = 0; y < height; ++y) {
                raw.push_back(0);  // filter type
                for (int x = 0; x < width; ++x) {
                    raw.push_back(rgb8[(size_t)y * width * 3 + (size_t)x * 3]);
                    raw.push_back(rgb8[(size_t)y * width * 3 + (size_t)x * 3 + 1]);
                    raw.push_back(rgb8[(size_t)y * width * 3 + (size_t)x * 3 + 2]);
                }
            }

            // Zlib header
            std::vector<uint8_t> idat_data;
            idat_data.push_back(0x78);  // CMF: 8-bit window, default compression
            idat_data.push_back(0x9C);  // FLG: no preset dictionary, checksum

            // Store as deflate blocks (stored, no compression)
            size_t pos = 0;
            while (pos < raw.size()) {
                size_t block_size = std::min(size_t(65535), raw.size() - pos);
                bool final_block = (pos + block_size == raw.size());

                uint8_t hdr = (final_block ? 1 : 0) | 0;  // BFINAL | BTYPE=0 (stored)
                idat_data.push_back(hdr);

                uint16_t len = (uint16_t)block_size;
                uint16_t nlen = (uint16_t)~len;
                idat_data.push_back((uint8_t)(len & 0xFF));
                idat_data.push_back((uint8_t)(len >> 8));
                idat_data.push_back((uint8_t)(nlen & 0xFF));
                idat_data.push_back((uint8_t)(nlen >> 8));

                idat_data.insert(idat_data.end(), raw.begin() + pos, raw.begin() + pos + block_size);
                pos += block_size;
            }

            // Adler32 checksum
            uint32_t adler = adler32(raw.data(), raw.size());
            write_be32(idat_data, adler);

            // Write IDAT chunk
            std::vector<uint8_t> chunk_type = {'I', 'D', 'A', 'T'};
            write_be32(file_data, (uint32_t)idat_data.size());
            file_data.insert(file_data.end(), chunk_type.begin(), chunk_type.end());
            file_data.insert(file_data.end(), idat_data.begin(), idat_data.end());

            std::vector<uint8_t> crc_data = chunk_type;
            crc_data.insert(crc_data.end(), idat_data.begin(), idat_data.end());
            uint32_t crc = crc32(crc_data.data(), crc_data.size());
            write_be32(file_data, crc);
        }

        // IEND chunk
        {
            std::vector<uint8_t> chunk_type = {'I', 'E', 'N', 'D'};
            write_be32(file_data, 0);  // length
            file_data.insert(file_data.end(), chunk_type.begin(), chunk_type.end());
            uint32_t crc = crc32(chunk_type.data(), chunk_type.size());
            write_be32(file_data, crc);
        }

        // Write to file
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("write_png: cannot open file for writing");
        }
        f.write((const char*)file_data.data(), file_data.size());
        if (!f) {
            throw std::runtime_error("write_png: write failed");
        }
    } catch (const std::exception&) {
        throw;
    }
}

void write_pfm(const std::string& path, int width, int height, const std::vector<float>& rgb) {
    if ((int)rgb.size() != 3 * width * height) {
        throw std::runtime_error("write_pfm: rgb size mismatch");
    }

    try {
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("write_pfm: cannot open file for writing");
        }

        // PFM header: "PF\n<width> <height>\n-1.0\n"
        f << "PF\n";
        f << width << " " << height << "\n";
        f << "-1.0\n";  // little-endian scale

        // Write pixels bottom-to-top, little-endian floats
        for (int y = height - 1; y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                float values[3] = {
                    rgb[((size_t)y * width + x) * 3],
                    rgb[((size_t)y * width + x) * 3 + 1],
                    rgb[((size_t)y * width + x) * 3 + 2],
                };
                f.write((const char*)values, sizeof(values));
            }
        }

        if (!f) {
            throw std::runtime_error("write_pfm: write failed");
        }
    } catch (const std::exception&) {
        throw;
    }
}

std::vector<float> read_pfm(const std::string& path, int* width, int* height) {
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("read_pfm: cannot open file");
        }

        std::string magic;
        f >> magic;
        if (magic != "PF" && magic != "Pf") {
            throw std::runtime_error("read_pfm: not a PFM file");
        }

        int w, h;
        f >> w >> h;
        if (w <= 0 || h <= 0) {
            throw std::runtime_error("read_pfm: invalid dimensions");
        }

        float scale;
        f >> scale;
        bool is_big_endian = scale > 0.0f;

        // Skip to binary data (exactly one whitespace after scale)
        char c;
        f.get(c);
        if (!f || (c != '\n' && c != '\r' && c != ' ' && c != '\t')) {
            throw std::runtime_error("read_pfm: bad header");
        }

        std::vector<float> rgb((size_t)3 * w * h);

        // Read pixels bottom-to-top (PFM format)
        for (int y = h - 1; y >= 0; --y) {
            for (int x = 0; x < w; ++x) {
                uint8_t bytes[12];
                f.read((char*)bytes, sizeof(bytes));
                if (!f) {
                    throw std::runtime_error("read_pfm: truncated file");
                }

                for (int col = 0; col < 3; ++col) {
                    float val;
                    if (is_big_endian) {
                        // Reverse bytes for big-endian to little-endian conversion
                        uint8_t reversed[4] = {bytes[4*col+3], bytes[4*col+2], bytes[4*col+1], bytes[4*col]};
                        std::memcpy(&val, reversed, sizeof(float));
                    } else {
                        // Little-endian: read directly
                        std::memcpy(&val, bytes + 4*col, sizeof(float));
                    }
                    rgb[((size_t)y * w + x) * 3 + col] = val;
                }
            }
        }

        *width = w;
        *height = h;
        return rgb;
    } catch (const std::exception&) {
        throw;
    }
}

uint64_t fnv1a64(const void* data, size_t bytes) {
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

std::string hex64(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)v);
    return std::string(buf);
}

}  // namespace wf
