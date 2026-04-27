// FORGE-emitted barycentric_eval kernel + host shim.
// Source: forge/analysis/vortex_ntt/barycentric_eval.fg
// 197 proof obligations including bit-twiddle bounds, M31 invariants
// through the grid-stride accumulator, warp-shuffle re-bound via
// `% M31_P` post-cast, and the inter-warp shared-memory reduction.

#include "forge/barycentric_eval_forge.cu"

extern "C" void cuda_barycentric_eval_forge(
    const uint32_t* evals,
    const uint32_t* weights,
    uint32_t n,
    uint32_t* out,
    uint32_t n_blocks
) {
    if (n == 0u || n_blocks == 0u) return;
    uint32_t threads = 256u;
    forge_span_u32_t s_ev = { const_cast<uint32_t*>(evals), (uintptr_t)n };
    forge_span_u32_t s_we = { const_cast<uint32_t*>(weights), (uintptr_t)n * 4u };
    forge_span_u32_t s_ou = { out, (uintptr_t)n_blocks * 4u };
    barycentric_eval<<<n_blocks, threads>>>(s_ev, s_we, s_ou, (uint64_t)n);
    cudaDeviceSynchronize();
}
