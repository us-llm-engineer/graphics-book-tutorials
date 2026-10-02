// Minimal image I/O: PNG writer (stored deflate), PFM read/write, FNV-1a hashing.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace wf {

// 8-bit RGB PNG (color type 2), zlib stored blocks, CRC32 and Adler32 computed here.
// Throws std::runtime_error if the file cannot be written or sizes mismatch.
// Scanline filter type 0 (None). A failed write leaves no file behind.
void write_png(const std::string& path, int width, int height, const std::vector<uint8_t>& rgb8);

// Little-endian PFM ("PF", scale -1.0), rows stored bottom-to-top as the format requires.
void write_pfm(const std::string& path, int width, int height, const std::vector<float>& rgb);
// Reads PFM written by write_pfm (and generic PF with either endianness). Throws on error.
std::vector<float> read_pfm(const std::string& path, int* width, int* height);

uint64_t fnv1a64(const void* data, size_t bytes);
std::string hex64(uint64_t v);  // 16 lowercase hex digits

}  // namespace wf
