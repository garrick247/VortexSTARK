// FORGE-emitted FRI fold_line kernel (SoA layout, QM31) + host shim.
// Source: forge/analysis/vortex_ntt/fri_fold_line.fg
// 193 proof obligations: per-component < M31_P preconditions, span
// bounds re-establishment chain, and the canonicalize→helper-call
// flow that makes the stdlib's m31/qm31 contracts discharge.

#include "forge/fri_fold_line_forge.cu"

extern "C" void cuda_fold_line_soa_forge(
    const uint32_t* in0, const uint32_t* in1,
    const uint32_t* in2, const uint32_t* in3,
    const uint32_t* twiddles,
    uint32_t* out0, uint32_t* out1,
    uint32_t* out2, uint32_t* out3,
    const uint32_t* alpha,    // [4] on host
    uint32_t half_n
) {
    if (half_n == 0) return;
    uint32_t threads = 256;
    uint32_t blocks  = (half_n + threads - 1) / threads;
    forge_span_u32_t s_in0  = { const_cast<uint32_t*>(in0),  (uintptr_t)half_n * 2u };
    forge_span_u32_t s_in1  = { const_cast<uint32_t*>(in1),  (uintptr_t)half_n * 2u };
    forge_span_u32_t s_in2  = { const_cast<uint32_t*>(in2),  (uintptr_t)half_n * 2u };
    forge_span_u32_t s_in3  = { const_cast<uint32_t*>(in3),  (uintptr_t)half_n * 2u };
    forge_span_u32_t s_tw   = { const_cast<uint32_t*>(twiddles), (uintptr_t)half_n };
    forge_span_u32_t s_out0 = { out0, (uintptr_t)half_n };
    forge_span_u32_t s_out1 = { out1, (uintptr_t)half_n };
    forge_span_u32_t s_out2 = { out2, (uintptr_t)half_n };
    forge_span_u32_t s_out3 = { out3, (uintptr_t)half_n };
    fold_line_soa<<<blocks, threads>>>(
        s_in0, s_in1, s_in2, s_in3, s_tw,
        s_out0, s_out1, s_out2, s_out3,
        alpha[0], alpha[1], alpha[2], alpha[3],
        (uint64_t)half_n
    );
}
