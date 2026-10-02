// Counter-based RNG: every draw is a pure function of (seed, pixel, sample, depth, dim).
// Draw order never matters, so megakernel and wavefront consume identical random numbers.
#pragma once
#include "wf/math.hpp"

namespace wf {

WF_HD uint32_t hash_u32(uint32_t x) {  // lowbias32 integer hash
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

struct PathKey {
    uint32_t k0, k1;  // derived from (seed, pixel, sample)
};

// Two independent hash chains over (seed, pixel, sample): k0 and k1 do not determine each other,
// so a key carries about 64 bits and distinct paths virtually never share a random stream.
WF_HD PathKey make_path_key(uint32_t seed, uint32_t pixel, uint32_t sample) {
    uint32_t a = hash_u32(seed ^ 0x9e3779b9U);
    uint32_t b = hash_u32(seed + 0x85ebca6bU);
    uint32_t k0 = hash_u32(hash_u32(a + pixel * 0xc2b2ae35U) ^ (sample * 0x27d4eb2fU + 0x165667b1U));
    uint32_t k1 = hash_u32(hash_u32(b ^ (pixel * 0x9e3779b1U)) + (sample * 0x85ebca77U + 0xc2b2ae3dU));
    return PathKey{k0, k1};
}

// Dimension slots used by the estimator (kept < 16 per depth).
enum RngDim : uint32_t {
    kDimPixelX = 0, kDimPixelY = 1,
    kDimLightPick = 2, kDimLightU = 3, kDimLightV = 4,
    kDimBsdfU = 5, kDimBsdfV = 6, kDimRoulette = 7, kDimDielectric = 8,
    kDimCount = 16
};

// Uniform float in [0,1) with 24 bits of resolution.
WF_HD float rng(PathKey key, uint32_t depth, uint32_t dim) {
    uint32_t ctr = depth * kDimCount + dim;
    uint32_t h = hash_u32(key.k0 ^ hash_u32(ctr * 0x9e3779b9U + key.k1));
    return (h >> 8) * (1.0f / 16777216.0f);
}

}  // namespace wf
