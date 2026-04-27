// FORGE-emitted Blake2s PoW grind kernel + host shim.
// Source: forge/analysis/vortex_ntt/grind.fg
// 22 proof obligations discharged + 0 user assumptions — Blake2s round
// structure copy-paste-equivalent to merkle_hash_leaves.fg's compression,
// trailing-zero count via 32-branch unrolled if-else chain.

#include "forge/grind_forge.cu"

extern "C" void cuda_grind_pow_forge(
    const uint32_t* prefixed_digest,
    uint64_t* result,
    uint32_t pow_bits,
    uint64_t batch_offset,
    uint32_t n_threads
) {
    if (n_threads == 0u) return;
    uint32_t threads_per_block = 256u;
    uint32_t blocks = (n_threads + threads_per_block - 1u) / threads_per_block;
    forge_span_u32_t s_pd = { const_cast<uint32_t*>(prefixed_digest), (uintptr_t)8u };
    grind_pow<<<blocks, threads_per_block>>>(
        s_pd, result, pow_bits, batch_offset, (uint64_t)n_threads
    );
}
