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

extern "C" {

void cuda_permute_hc_to_canonic_brt(
    const uint32_t* src,
    uint32_t* dst,
    uint32_t n,
    uint32_t log_n
) {
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1u) / threads;
    permute_hc_to_canonic_brt_kernel<<<blocks, threads>>>(src, dst, n, log_n);
}

} // extern "C"
