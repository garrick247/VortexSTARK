// FORGE-emitted Circle NTT layer kernel + host shim.
// Source: forge/analysis/vortex_ntt/circle_ntt_layer.fg
// 138 proof obligations including bounds-check on the bit-math
// derived (idx0, idx1) pair and `< M31_P` invariants threaded
// through `m31_sub` → `m31_mul` for the inverse butterfly.

#include "forge/circle_ntt_layer_forge.cu"

extern "C" void cuda_circle_ntt_layer_forge(
    uint32_t* data,
    const uint32_t* twiddles,
    uint32_t layer_idx,
    uint32_t n,
    int forward
) {
    if (n < 2) return;
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks  = (half_n + threads - 1) / threads;
    forge_span_u32_t s_data = { data, (uintptr_t)n };
    // Twiddles span: largest h = (half_n - 1) >> layer_idx; bound by half_n.
    forge_span_u32_t s_tw   = { const_cast<uint32_t*>(twiddles), (uintptr_t)half_n };
    if (forward) {
        circle_ntt_layer_forward<<<blocks, threads>>>(
            s_data, s_tw, layer_idx, (uint64_t)half_n);
    } else {
        circle_ntt_layer_inverse<<<blocks, threads>>>(
            s_data, s_tw, layer_idx, (uint64_t)half_n);
    }
}
