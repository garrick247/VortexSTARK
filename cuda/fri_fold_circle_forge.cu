// FORGE-emitted FRI fold_circle_into_line kernel + host shim.
// Source: forge/analysis/vortex_ntt/fri_fold_circle.fg
// 237 proof obligations.

#include "forge/fri_fold_circle_forge.cu"

extern "C" void cuda_fold_circle_into_line_soa_forge(
    uint32_t* dst0, uint32_t* dst1,
    uint32_t* dst2, uint32_t* dst3,
    const uint32_t* src0, const uint32_t* src1,
    const uint32_t* src2, const uint32_t* src3,
    const uint32_t* twiddles,
    const uint32_t* alpha,     // [4] on host
    const uint32_t* alpha_sq,  // [4] on host
    uint32_t half_n
) {
    if (half_n == 0) return;
    uint32_t threads = 256;
    uint32_t blocks  = (half_n + threads - 1) / threads;
    forge_span_u32_t s_dst0 = { dst0, (uintptr_t)half_n };
    forge_span_u32_t s_dst1 = { dst1, (uintptr_t)half_n };
    forge_span_u32_t s_dst2 = { dst2, (uintptr_t)half_n };
    forge_span_u32_t s_dst3 = { dst3, (uintptr_t)half_n };
    forge_span_u32_t s_src0 = { const_cast<uint32_t*>(src0), (uintptr_t)half_n * 2u };
    forge_span_u32_t s_src1 = { const_cast<uint32_t*>(src1), (uintptr_t)half_n * 2u };
    forge_span_u32_t s_src2 = { const_cast<uint32_t*>(src2), (uintptr_t)half_n * 2u };
    forge_span_u32_t s_src3 = { const_cast<uint32_t*>(src3), (uintptr_t)half_n * 2u };
    forge_span_u32_t s_tw   = { const_cast<uint32_t*>(twiddles), (uintptr_t)half_n };
    fold_circle_into_line_soa<<<blocks, threads>>>(
        s_dst0, s_dst1, s_dst2, s_dst3,
        s_src0, s_src1, s_src2, s_src3, s_tw,
        alpha[0], alpha[1], alpha[2], alpha[3],
        alpha_sq[0], alpha_sq[1], alpha_sq[2], alpha_sq[3],
        (uint64_t)half_n
    );
}
