// Device-side queue and scan primitives, written from scratch (no CUB, no Thrust).
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace wf::gpu {

// Warp-aggregated queue push: lanes with pred=true reserve consecutive slots with ONE atomicAdd per
// warp (ballot + popc), then write `value`. The whole warp must call this (no divergent early
// exit). If the reservation would pass `capacity`, the push is dropped and *overflow is
// incremented (atomically) once per dropped lane. Returns the slot index or 0xFFFFFFFFu.
__device__ __forceinline__ uint32_t queue_push(uint32_t* items, uint32_t* counter,
                                               uint32_t capacity, uint32_t value, bool pred,
                                               uint32_t* overflow) {
    unsigned mask = __activemask();
    unsigned ballot = __ballot_sync(mask, pred);
    int lane = threadIdx.x & 31;
    int leader = __ffs(ballot) - 1;
    uint32_t base = 0;
    if (ballot != 0 && lane == leader) base = atomicAdd(counter, (uint32_t)__popc(ballot));
    base = __shfl_sync(mask, base, leader);
    if (!pred) return 0xFFFFFFFFu;
    uint32_t slot = base + (uint32_t)__popc(ballot & ((1u << lane) - 1u));
    if (slot >= capacity) { atomicAdd(overflow, 1u); return 0xFFFFFFFFu; }
    items[slot] = value;
    return slot;
}

// Scratch space for scan/compaction; sized for up to `capacity` elements.
struct ScanWorkspace {
    uint32_t* block_sums = nullptr;   // device, ceil(capacity / 1024) entries (+1)
    int capacity = 0;
};
cudaError_t scan_workspace_create(ScanWorkspace* ws, int capacity);
void scan_workspace_destroy(ScanWorkspace* ws);

// Exclusive prefix sum of n uint32 values: out[i] = sum(in[0..i-1]). in may equal out. Writes the
// grand total to *d_total (device pointer, may be null). Block scan + scan of block sums + add
// offsets, all custom kernels. n <= ws.capacity. Stream-ordered, no host sync.
cudaError_t scan_exclusive_u32(const uint32_t* d_in, uint32_t* d_out, int n, uint32_t* d_total,
                               ScanWorkspace& ws, cudaStream_t stream);

// Stable stream compaction: out[k] = values[i] for the k-th i with flags[i] != 0 (original order
// preserved). *d_count receives the number selected. flags and values are unchanged. Needs a
// scratch uint32 array `d_scratch` of n entries.
cudaError_t compact_u32(const uint32_t* d_values, const uint32_t* d_flags, uint32_t* d_out,
                        uint32_t* d_count, uint32_t* d_scratch, int n, ScanWorkspace& ws,
                        cudaStream_t stream);

// Test hook: runs a kernel in which `n` lanes push their index when flag[i] != 0 through
// queue_push with the given capacity; fills out_items (capacity entries), *d_counter and
// *d_overflow (both must be zeroed by the caller). Order inside the queue is unspecified.
cudaError_t test_queue_push(const uint32_t* d_flags, int n, uint32_t* d_items, uint32_t capacity,
                            uint32_t* d_counter, uint32_t* d_overflow, cudaStream_t stream);

}  // namespace wf::gpu
