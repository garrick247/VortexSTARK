// Per-u32-word M31 reduction (in place).
//
// Matches the CPU reference `M31::reduce` in src/field/m31.rs:
//
//     fn reduce(val: u64) -> M31 {
//         let lo = (val & P as u64) as u32;
//         let hi = (val >> 31) as u32;
//         let r = lo + hi;
//         M31(if r >= P { r - P } else { r })
//     }
//
// For a u32 input (top bit 0 or 1), this folds `[0, 2^32)` → `[0, P-1]`
// canonically — `P` and `2P` both map to 0. This produces the
// `HashValue<QM31>`-safe form of a standard Blake2s digest word.
//
// Used by the `shinobi-hash` feature to post-process Merkle layer
// outputs so every word fits Shinobi's deserializer's `val < P` check.

#include <cstdint>

static constexpr uint32_t M31_P = 0x7FFFFFFFu;

__global__ void reduce_words_to_m31_kernel(
    uint32_t* __restrict__ data,
    uint32_t n_words
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_words) return;
    uint32_t v = data[idx];
    // Standard M31 reduction: lo = v & P, hi = v >> 31, r = lo + hi,
    // canonicalize with a branchless subtract.
    uint32_t lo = v & M31_P;
    uint32_t hi = v >> 31;
    uint32_t r  = lo + hi;
    data[idx]   = (r >= M31_P) ? (r - M31_P) : r;
}

extern "C" void cuda_reduce_words_to_m31(
    uint32_t* data,
    uint32_t n_words
) {
    if (n_words == 0) return;
    uint32_t threads = 256;
    uint32_t blocks  = (n_words + threads - 1) / threads;
    reduce_words_to_m31_kernel<<<blocks, threads>>>(data, n_words);
}
