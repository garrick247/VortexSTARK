// M31 reduction kernel — FORGE-generated body + host-launch shim.
//
// The kernel body in `forge/reduce_m31_forge.cu` is emitted by the FORGE
// compiler from `forge/analysis/vortex_ntt/reduce_m31.fg` via
//   `forge build-lib <path>`
// (lib-mode emit, added in forge commit after a7ad1b8 — produces .cu
// with zero `int main()` bodies so it integrates cleanly as a linked
// translation unit in this binary). 118 proof obligations discharged:
//   - `reduce_word(v)` postcondition: result < M31_P (canonical form)
//   - kernel bounds checks on `data[tid]` reads/writes

#include "forge/reduce_m31_forge.cu"

// Host-launch shim matching VortexSTARK's existing ABI. Signature is
// preserved byte-for-byte so Rust-side callers (blake2s_m31::
// reduce_device_buffer) don't need to change.
extern "C" void cuda_reduce_words_to_m31(uint32_t* data, uint32_t n_words) {
    if (n_words == 0) return;
    uint32_t threads = 256;
    uint32_t blocks  = (n_words + threads - 1) / threads;
    // FORGE-emitted kernel takes a span (fat pointer). Build one inline.
    forge_span_u32_t span = { data, (uintptr_t)n_words };
    reduce_words_to_m31<<<blocks, threads>>>(span, (uint64_t)n_words);
}
