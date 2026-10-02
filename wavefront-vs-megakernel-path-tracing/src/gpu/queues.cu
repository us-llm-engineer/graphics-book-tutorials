// GPU queue primitives: exclusive scan, stable compaction, warp-aggregated queue_push.
// Written from scratch, no CUB or Thrust.
#include <cuda_runtime.h>
#include "queues.cuh"

namespace wf::gpu {

constexpr int kBlockSize = 1024;   // one block scans a 1024-element tile (32 warps)
constexpr unsigned kFullMask = 0xffffffffu;

// Warp-level inclusive scan. Every lane of the warp must call it (full-mask shuffles).
__device__ __forceinline__ uint32_t warp_scan_inclusive(uint32_t val) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        uint32_t other = __shfl_up_sync(kFullMask, val, offset);
        if (lane >= offset) val += other;
    }
    return val;
}

// Block-wide scan of one 1024-element tile; every thread of the block calls it, no early exit.
// Returns the thread's exclusive prefix and sets *tile_total (visible to all threads) to the
// inclusive sum of the whole tile. `warp_totals` is a 32-entry shared array.
__device__ __forceinline__ uint32_t block_scan_exclusive(uint32_t val, uint32_t* warp_totals,
                                                         uint32_t* tile_total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    uint32_t inclusive = warp_scan_inclusive(val);
    if (lane == 31) warp_totals[warp] = inclusive;
    __syncthreads();
    if (warp == 0) warp_totals[lane] = warp_scan_inclusive(warp_totals[lane]);  // prefix of warp sums
    __syncthreads();
    uint32_t before = (warp > 0) ? warp_totals[warp - 1] : 0u;
    *tile_total = warp_totals[31];
    return before + inclusive - val;
}

// ============================================================================
// Workspace management
// ============================================================================

cudaError_t scan_workspace_create(ScanWorkspace* ws, int capacity) {
    if (capacity <= 0) return cudaErrorInvalidValue;

    int n_blocks = (capacity + kBlockSize - 1) / kBlockSize;
    ws->capacity = capacity;

    cudaError_t e = cudaMalloc(&ws->block_sums, (size_t)(n_blocks + 1) * sizeof(uint32_t));
    if (e != cudaSuccess) {
        ws->block_sums = nullptr;
        ws->capacity = 0;
    }
    return e;
}

void scan_workspace_destroy(ScanWorkspace* ws) {
    if (ws->block_sums) {
        cudaFree(ws->block_sums);
        ws->block_sums = nullptr;
    }
    ws->capacity = 0;
}

// ============================================================================
// Pass 1: exclusive scan inside each tile, tile totals to block_sums
// ============================================================================

__global__ void scan_pass1_kernel(const uint32_t* d_in, uint32_t* d_out, int n, uint32_t* d_block_sums) {
    __shared__ uint32_t warp_totals[32];
    const int idx = blockIdx.x * kBlockSize + threadIdx.x;

    // The input element is read into a register before any write, so d_in == d_out is legal.
    uint32_t val = (idx < n) ? d_in[idx] : 0u;
    uint32_t tile_total;
    uint32_t excl = block_scan_exclusive(val, warp_totals, &tile_total);

    if (idx < n) d_out[idx] = excl;
    if (threadIdx.x == kBlockSize - 1) d_block_sums[blockIdx.x] = tile_total;
}

// ============================================================================
// Pass 2: one block exclusive-scans the tile totals (looping over 1024-entry tiles), grand total
// ============================================================================

__global__ void scan_pass2_kernel(uint32_t* d_block_sums, int n_blocks, uint32_t* d_total) {
    __shared__ uint32_t warp_totals[32];
    __shared__ uint32_t carry;
    if (threadIdx.x == 0) carry = 0u;
    __syncthreads();

    for (int base = 0; base < n_blocks; base += kBlockSize) {
        const int i = base + threadIdx.x;
        uint32_t val = (i < n_blocks) ? d_block_sums[i] : 0u;
        uint32_t tile_total;
        uint32_t excl = block_scan_exclusive(val, warp_totals, &tile_total);
        const uint32_t offset = carry;
        if (i < n_blocks) d_block_sums[i] = offset + excl;
        __syncthreads();                                  // everyone has read carry and warp_totals
        if (threadIdx.x == 0) carry = offset + tile_total;
        __syncthreads();                                  // carry is visible to the next tile
    }
    if (threadIdx.x == 0 && d_total) *d_total = carry;
}

// ============================================================================
// Pass 3: add each tile's offset
// ============================================================================

__global__ void scan_pass3_kernel(uint32_t* d_out, int n, const uint32_t* d_block_sums) {
    const int idx = blockIdx.x * kBlockSize + threadIdx.x;
    if (idx < n) d_out[idx] += d_block_sums[blockIdx.x];
}

// ============================================================================
// Main scan function
// ============================================================================

cudaError_t scan_exclusive_u32(const uint32_t* d_in, uint32_t* d_out, int n, uint32_t* d_total,
                               ScanWorkspace& ws, cudaStream_t stream) {
    if (n < 0 || n > ws.capacity) return cudaErrorInvalidValue;

    if (n == 0) {
        if (d_total) return cudaMemsetAsync(d_total, 0, sizeof(uint32_t), stream);
        return cudaSuccess;
    }

    const int n_blocks = (n + kBlockSize - 1) / kBlockSize;

    scan_pass1_kernel<<<n_blocks, kBlockSize, 0, stream>>>(d_in, d_out, n, ws.block_sums);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    scan_pass2_kernel<<<1, kBlockSize, 0, stream>>>(ws.block_sums, n_blocks, d_total);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    scan_pass3_kernel<<<n_blocks, kBlockSize, 0, stream>>>(d_out, n, ws.block_sums);
    return cudaGetLastError();
}

// ============================================================================
// Stable compaction
// ============================================================================

// Indicator: 1 if flag != 0, else 0 (the scan runs on the indicator, never on raw flag values).
__global__ void make_indicator_kernel(const uint32_t* d_flags, uint32_t* d_indicator, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_indicator[idx] = (d_flags[idx] != 0u) ? 1u : 0u;
}

__global__ void compact_kernel(const uint32_t* d_values, const uint32_t* d_flags,
                               uint32_t* d_out, const uint32_t* d_indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && d_flags[idx] != 0u) d_out[d_indices[idx]] = d_values[idx];
}

cudaError_t compact_u32(const uint32_t* d_values, const uint32_t* d_flags, uint32_t* d_out,
                        uint32_t* d_count, uint32_t* d_scratch, int n, ScanWorkspace& ws,
                        cudaStream_t stream) {
    if (n < 0 || n > ws.capacity) return cudaErrorInvalidValue;

    if (n == 0) {
        if (d_count) return cudaMemsetAsync(d_count, 0, sizeof(uint32_t), stream);
        return cudaSuccess;
    }

    const int n_blocks = (n + 255) / 256;
    make_indicator_kernel<<<n_blocks, 256, 0, stream>>>(d_flags, d_scratch, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    e = scan_exclusive_u32(d_scratch, d_scratch, n, d_count, ws, stream);
    if (e != cudaSuccess) return e;

    compact_kernel<<<n_blocks, 256, 0, stream>>>(d_values, d_flags, d_out, d_scratch, n);
    return cudaGetLastError();
}

// ============================================================================
// Test queue_push
// ============================================================================

__global__ void test_queue_push_kernel(const uint32_t* d_flags, int n, uint32_t* d_items,
                                        uint32_t capacity, uint32_t* d_counter, uint32_t* d_overflow) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool pred = (idx < n && d_flags[idx] != 0u);
    queue_push(d_items, d_counter, capacity, (uint32_t)idx, pred, d_overflow);
}

cudaError_t test_queue_push(const uint32_t* d_flags, int n, uint32_t* d_items, uint32_t capacity,
                            uint32_t* d_counter, uint32_t* d_overflow, cudaStream_t stream) {
    int n_blocks = (n + 255) / 256;
    test_queue_push_kernel<<<n_blocks, 256, 0, stream>>>(d_flags, n, d_items, capacity, d_counter, d_overflow);
    return cudaGetLastError();
}

}  // namespace wf::gpu
