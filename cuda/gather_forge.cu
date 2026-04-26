// FORGE-emitted gather kernels (u32 + u256 = 8-word) + host-launch
// shims. Kernel bodies emitted from
// `forge/analysis/vortex_ntt/gather.fg` via `forge build-lib`.
// 20 proof obligations discharged — including the dependent bounds
// check on `src[idx[i]]` (the load index is data, so FORGE proves
// safety only when the runtime guard `idx[i] < src.len` is present).
//
// Drop-in replacement for `cuda/gather.cu`'s public ABI; FORGE versions
// take an explicit `src_len` parameter that the host already knows.
// Used by the `gpu_forge_gather_*_matches_handwritten` tests.

#include "forge/gather_forge.cu"

extern "C" void cuda_gather_u32_forge(
    const uint32_t* src,
    const uint32_t* idx,
    uint32_t* dst,
    uint32_t n,
    uint32_t src_len
) {
    if (n == 0) return;
    uint32_t threads = 256;
    uint32_t blocks  = (n + threads - 1) / threads;
    forge_span_u32_t s_src = { const_cast<uint32_t*>(src), (uintptr_t)src_len };
    forge_span_u32_t s_idx = { const_cast<uint32_t*>(idx), (uintptr_t)n };
    forge_span_u32_t s_dst = { dst, (uintptr_t)n };
    gather_u32<<<blocks, threads>>>(s_src, s_idx, s_dst,
                                    (uint64_t)n, (uint64_t)src_len);
}

extern "C" void cuda_gather_u256_forge(
    const uint32_t* src,
    const uint32_t* idx,
    uint32_t* dst,
    uint32_t n,
    uint32_t src_len
) {
    if (n == 0) return;
    uint32_t threads = 256;
    uint32_t blocks  = (n + threads - 1) / threads;
    forge_span_u32_t s_src = { const_cast<uint32_t*>(src), (uintptr_t)src_len };
    forge_span_u32_t s_idx = { const_cast<uint32_t*>(idx), (uintptr_t)n };
    forge_span_u32_t s_dst = { dst, (uintptr_t)n * 8u };
    gather_u256<<<blocks, threads>>>(s_src, s_idx, s_dst,
                                     (uint64_t)n, (uint64_t)src_len);
}
