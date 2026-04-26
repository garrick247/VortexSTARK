// FORGE-emitted QM31 bit-reverse kernel + host-launch shim.
// Source: forge/analysis/vortex_ntt/bit_reverse_qm31.fg
// 12 proof obligations: bit_reverse_u32 termination/invariant + per-thread
// bounds checks on the 4-u32 chunk read/write.
//
// Replaces the kernel body in cuda/bit_reverse_wide.cu under
// `forge-blake2s` style opt-in (separate FFI symbol so callers can
// choose deterministically).

#include "forge/bit_reverse_qm31_forge.cu"

extern "C" void cuda_bit_reverse_qm31_forge(
    const uint32_t* in,
    uint32_t* out,
    uint32_t n,
    uint32_t log_n
) {
    if (n == 0) return;
    uint32_t threads = 256;
    uint32_t blocks  = (n + threads - 1) / threads;
    forge_span_u32_t s_in  = { const_cast<uint32_t*>(in),  (uintptr_t)n * 4u };
    forge_span_u32_t s_out = { out, (uintptr_t)n * 4u };
    bit_reverse_qm31<<<blocks, threads>>>(s_in, s_out, (uint64_t)n, log_n);
}
