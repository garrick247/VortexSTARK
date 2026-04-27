// GPU permutation kernels for circle-domain index reorderings.
//
// The Cairo prover emits quotient/OODS-quotient evaluations in "hc-natural"
// order (half-coset natural) but commits them in "canonic-BRT" order (bit-
// reversed canonic coset). The conversion is:
//
//   for k in 0..n:
//       cn = if k & 1 { n - 1 - k/2 } else { k/2 }   // hc -> canonic
//       j  = bit_reverse(cn) in log_n bits            // canonic -> BRT
//       dst[j] = src[k]
//
// This is a simple one-pass scatter; one thread per source index.

#include <cstdint>

// FORGE permute wire-in: when `forge-permute` cargo feature is on,
// build.rs defines FORGE_PERMUTE=1 and both extern "C" entry points
// route to the proof-checked FORGE host shims in
// cuda/permute_forge.cu (source forge/analysis/vortex_ntt/permute.fg,
// 9 proof obligations discharged).
#ifdef FORGE_PERMUTE
extern "C" {
    void cuda_permute_hc_to_canonic_brt_forge(
        const uint32_t* src, uint32_t* dst, uint32_t n, uint32_t log_n);
    void cuda_permute_canonic_brt_to_hc_natural_forge(
        const uint32_t* src, uint32_t* dst, uint32_t n, uint32_t log_n);
}
#endif

__global__ void permute_hc_to_canonic_brt_kernel(
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    uint32_t n,
    uint32_t log_n
) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint32_t cn = (k & 1u) ? (n - 1u - (k >> 1)) : (k >> 1);
    uint32_t j = __brev(cn) >> (32u - log_n);
    dst[j] = src[k];
}

// Inverse of the above. Same index mapping, but we read src[j] and write dst[k]
// so that dst is hc-natural ordered from a canonic-BRT source. Used to keep the
// constraint-kernel input GPU-resident after Merkle commit (which needs BRT-canonic).
__global__ void permute_canonic_brt_to_hc_natural_kernel(
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    uint32_t n,
    uint32_t log_n
) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    uint32_t cn = (k & 1u) ? (n - 1u - (k >> 1)) : (k >> 1);
    uint32_t j = __brev(cn) >> (32u - log_n);
    dst[k] = src[j];
}

extern "C" {

void cuda_permute_hc_to_canonic_brt(
    const uint32_t* src,
    uint32_t* dst,
    uint32_t n,
    uint32_t log_n
) {
#ifdef FORGE_PERMUTE
    cuda_permute_hc_to_canonic_brt_forge(src, dst, n, log_n);
#else
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1u) / threads;
    permute_hc_to_canonic_brt_kernel<<<blocks, threads>>>(src, dst, n, log_n);
#endif
}

void cuda_permute_canonic_brt_to_hc_natural(
    const uint32_t* src,
    uint32_t* dst,
    uint32_t n,
    uint32_t log_n
) {
#ifdef FORGE_PERMUTE
    cuda_permute_canonic_brt_to_hc_natural_forge(src, dst, n, log_n);
#else
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1u) / threads;
    permute_canonic_brt_to_hc_natural_kernel<<<blocks, threads>>>(src, dst, n, log_n);
#endif
}

} // extern "C"
