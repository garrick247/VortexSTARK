// M31 reduction kernel — FORGE-generated body + host-launch shim.
//
// The kernel body in `forge/reduce_m31_forge.cu` is emitted by the FORGE
// compiler (analysis/vortex_ntt/reduce_m31.fg) with 118 proof obligations
// discharged by Z3 — including:
//   - `reduce_word(v)` postcondition: result < M31_P (canonical form)
//   - kernel bounds checks on `data[tid]` reads/writes
//
// FORGE emit hand-patches applied (tracked as upstream-bug work):
//   1. `__device__` added to `warp_reduce_*` helpers; FORGE's current
//      emit defaults them to `__host__`, which then fails to call the
//      `__device__`-only `__shfl_xor_sync`. Fix belongs in FORGE's
//      codegen_c.ml emitter.
//   2. All `int main()` stubs stripped. FORGE currently emits a `main()`
//      per included module (user + std::gpu + std::m31 = 3 copies) which
//      collides when linked into a larger project. FORGE should gain a
//      `--no-main` / library-mode emit.
//
// These patches move to FORGE itself in a follow-up; this shim is the
// pilot showing the FORGE CUDA backend can produce kernels VortexSTARK
// ships with.

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
