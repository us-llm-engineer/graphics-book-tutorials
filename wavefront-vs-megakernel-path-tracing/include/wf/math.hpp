// Vector math usable on host and device. No external dependencies.
#pragma once
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define WF_HD __host__ __device__ __forceinline__
#else
#define WF_HD inline
#endif

namespace wf {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvPi = 0.31830988618379067154f;
constexpr float kInf = 3.402823466e+38f;

struct Vec3 {
    float x, y, z;
};

WF_HD Vec3 make_vec3(float x, float y, float z) { return Vec3{x, y, z}; }
WF_HD Vec3 operator+(Vec3 a, Vec3 b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
WF_HD Vec3 operator-(Vec3 a, Vec3 b) { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
WF_HD Vec3 operator-(Vec3 a) { return Vec3{-a.x, -a.y, -a.z}; }
WF_HD Vec3 operator*(Vec3 a, Vec3 b) { return Vec3{a.x * b.x, a.y * b.y, a.z * b.z}; }
WF_HD Vec3 operator*(Vec3 a, float s) { return Vec3{a.x * s, a.y * s, a.z * s}; }
WF_HD Vec3 operator*(float s, Vec3 a) { return Vec3{a.x * s, a.y * s, a.z * s}; }
WF_HD Vec3 operator/(Vec3 a, float s) { float r = 1.0f / s; return Vec3{a.x * r, a.y * r, a.z * r}; }
WF_HD Vec3& operator+=(Vec3& a, Vec3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
WF_HD Vec3& operator*=(Vec3& a, Vec3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
WF_HD Vec3& operator*=(Vec3& a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }

WF_HD float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
WF_HD Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
WF_HD float length2(Vec3 a) { return dot(a, a); }
WF_HD float length(Vec3 a) { return sqrtf(dot(a, a)); }
WF_HD Vec3 normalize(Vec3 a) { float l = length(a); return l > 0.0f ? a / l : a; }
WF_HD float fminf3(float a, float b) { return a < b ? a : b; }
WF_HD float fmaxf3(float a, float b) { return a > b ? a : b; }
WF_HD float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
WF_HD float max_component(Vec3 a) { return fmaxf3(a.x, fmaxf3(a.y, a.z)); }
WF_HD Vec3 vmin(Vec3 a, Vec3 b) { return Vec3{fminf3(a.x, b.x), fminf3(a.y, b.y), fminf3(a.z, b.z)}; }
WF_HD Vec3 vmax(Vec3 a, Vec3 b) { return Vec3{fmaxf3(a.x, b.x), fmaxf3(a.y, b.y), fmaxf3(a.z, b.z)}; }
WF_HD Vec3 reflect(Vec3 d, Vec3 n) { return d - n * (2.0f * dot(d, n)); }
WF_HD float luminance(Vec3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }
// NaN and +-inf both fail v - v == 0 (portable on host and device, no fast-math allowed).
WF_HD bool is_finite1(float v) { return v - v == 0.0f; }
WF_HD bool is_finite3(Vec3 a) { return is_finite1(a.x) && is_finite1(a.y) && is_finite1(a.z); }

// Orthonormal basis around unit n (Frisvad-style, branch on sign of z).
WF_HD void make_basis(Vec3 n, Vec3* t, Vec3* b) {
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sign + n.z);
    float c = n.x * n.y * a;
    *t = Vec3{1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x};
    *b = Vec3{c, sign + n.y * n.y * a, -n.y};
}

struct Ray {
    Vec3 o;
    float tmin;
    Vec3 d;
    float tmax;
};

}  // namespace wf
