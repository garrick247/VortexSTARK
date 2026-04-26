// FORGE-emitted M31 batch-inverse kernel + host-launch shim.
// Source: forge/analysis/vortex_ntt/batch_inverse.fg
// 160 proof obligations: m31_inv termination + result < M31_P, per-chunk
// loop invariants on prefix products, dependent bounds on stack array.
//
// Drop-in for cuda/batch_inverse.cu (same chunk size 64, same Montgomery
// trick). Used by gpu_forge_batch_inverse_matches_handwritten.

#include "forge/batch_inverse_forge.cu"

extern "C" void cuda_batch_inverse_m31_forge(
    const uint32_t* input,
    uint32_t* output,
    uint32_t n
) {
    if (n == 0) return;
    uint32_t chunk = 64u;
    uint32_t n_chunks = (n + chunk - 1u) / chunk;
    uint32_t threads = 256u;
    uint32_t blocks  = (n_chunks + threads - 1u) / threads;
    forge_span_u32_t s_in  = { const_cast<uint32_t*>(input), (uintptr_t)n };
    forge_span_u32_t s_out = { output, (uintptr_t)n };
    batch_inverse_m31<<<blocks, threads>>>(s_in, s_out, (uint64_t)n);
}
