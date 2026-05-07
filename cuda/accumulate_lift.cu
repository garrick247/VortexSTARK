// GPU kernel for AccumulationOps::lift_and_accumulate.
//
// Implements one iteration of the lift-and-accumulate loop (stwo CPU impl):
//
//   log_ratio = col.len().ilog2() - curr.len().ilog2()
//   for i in 0..col.len():
//       src_idx = (i >> (log_ratio + 1) << 1) | (i & 1)
//       col[i] += curr[src_idx]
//
// SoA layout: SecureColumnByCoords has 4 SEPARATE Col<B, BaseField> channels,
// each `col_n` M31 values long. The Rust caller invokes this kernel 4 times,
// once per channel, with that channel's buffer.  So this kernel does plain
// M31 add on a single channel's buffer of `col_n` u32 elements (NOT 4*col_n).

#include "include/qm31.cuh"

__global__ void accumulate_lift_kernel(
    uint32_t* __restrict__ col,         // in/out: col_n M31 values (one channel)
    const uint32_t* __restrict__ curr,  // read-only: curr_n M31 values (one channel)
    uint32_t col_n,
    uint32_t log_ratio                  // = log2(col_n) - log2(curr_n)
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= col_n) return;

    // src_idx = (i >> (log_ratio + 1) << 1) | (i & 1)
    uint32_t shift = log_ratio + 1;
    uint32_t src_idx = ((i >> shift) << 1) | (i & 1u);

    col[i] = m31_add(col[i], curr[src_idx]);
}

extern "C" {

void cuda_accumulate_lift(
    uint32_t* col,
    const uint32_t* curr,
    uint32_t col_n,
    uint32_t log_ratio
) {
    uint32_t blocks = (col_n + 255) / 256;
    accumulate_lift_kernel<<<blocks, 256>>>(col, curr, col_n, log_ratio);
}

} // extern "C"