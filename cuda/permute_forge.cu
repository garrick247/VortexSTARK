// FORGE-emitted permute kernels + host shims.
// Source: forge/analysis/vortex_ntt/permute.fg (9 proof obligations).
//
// Wraps the FORGE-emitted hc-natural ↔ canonic-BRT scatter kernels and
// presents the same public ABI as cuda/permute.cu's
// `cuda_permute_hc_to_canonic_brt` / `_to_hc_natural`. Toggled via the
// `forge-permute` cargo feature; the macro dispatch lives in
// cuda/permute.cu.

#include "forge/permute_forge.cu"

extern "C" void cuda_permute_hc_to_canonic_brt_forge(
    const uint32_t* src,
    uint32_t* dst,
    uint32_t n,
    uint32_t log_n
) {
    if (n == 0u) return;
    uint32_t threads = 256u;
    uint32_t blocks  = (n + threads - 1u) / threads;
    forge_span_u32_t s_src = { const_cast<uint32_t*>(src), (uintptr_t)n };
    forge_span_u32_t s_dst = { dst, (uintptr_t)n };
    permute_hc_to_canonic_brt<<<blocks, threads>>>(s_src, s_dst, (uint64_t)n, log_n);
}

extern "C" void cuda_permute_canonic_brt_to_hc_natural_forge(
    const uint32_t* src,
    uint32_t* dst,
    uint32_t n,
    uint32_t log_n
) {
    if (n == 0u) return;
    uint32_t threads = 256u;
    uint32_t blocks  = (n + threads - 1u) / threads;
    forge_span_u32_t s_src = { const_cast<uint32_t*>(src), (uintptr_t)n };
    forge_span_u32_t s_dst = { dst, (uintptr_t)n };
    permute_canonic_brt_to_hc_natural<<<blocks, threads>>>(s_src, s_dst, (uint64_t)n, log_n);
}
