// FORGE-emitted Blake2s leaf-hash kernel (single-column variant) +
// host-launch shim. The kernel body in
// `forge/merkle_hash_leaves_forge.cu` is emitted by the FORGE compiler
// from `forge/analysis/vortex_ntt/merkle_hash_leaves.fg` via
//   `forge build-lib <path>`
// 133 proof obligations discharged:
//   - reduce_word postcondition (output < M31_P)
//   - kernel-level preconditions (n_leaves <= column.len,
//     n_leaves * 8 <= hashes.len)
//   - 8 output bounds checks gated by an explicit `if base + 7 < hashes.len`
//     guard (SMT can't follow non-linear `leaf*8 < n_leaves*8`)
//
// This sits alongside the hand-written `cuda_merkle_hash_leaves`
// (cuda/blake2s.cu) — same Blake2s, same M31 reduction, but generated
// from a verified source. Used by the `gpu_forge_leaf_hash_matches_cpu`
// test (src/blake2s_m31.rs) for n_cols = 1 to validate parity.

#include "forge/merkle_hash_leaves_forge.cu"

extern "C" void cuda_merkle_hash_leaves_forge_single(
    const uint32_t* column,
    uint32_t* hashes,
    uint32_t n_leaves
) {
    if (n_leaves == 0) return;
    uint32_t threads = 256;
    uint32_t blocks  = (n_leaves + threads - 1) / threads;
    forge_span_u32_t col  = { const_cast<uint32_t*>(column),
                              (uintptr_t)n_leaves };
    forge_span_u32_t hash = { hashes, (uintptr_t)n_leaves * 8u };
    merkle_hash_leaves_single<<<blocks, threads>>>(col, hash,
                                                   (uint64_t)n_leaves);
}

extern "C" void cuda_merkle_hash_leaves_forge_quad(
    const uint32_t* c0, const uint32_t* c1,
    const uint32_t* c2, const uint32_t* c3,
    uint32_t* hashes,
    uint32_t n_leaves
) {
    if (n_leaves == 0) return;
    uint32_t threads = 256;
    uint32_t blocks  = (n_leaves + threads - 1) / threads;
    forge_span_u32_t s0 = { const_cast<uint32_t*>(c0), (uintptr_t)n_leaves };
    forge_span_u32_t s1 = { const_cast<uint32_t*>(c1), (uintptr_t)n_leaves };
    forge_span_u32_t s2 = { const_cast<uint32_t*>(c2), (uintptr_t)n_leaves };
    forge_span_u32_t s3 = { const_cast<uint32_t*>(c3), (uintptr_t)n_leaves };
    forge_span_u32_t hash = { hashes, (uintptr_t)n_leaves * 8u };
    merkle_hash_leaves_quad<<<blocks, threads>>>(s0, s1, s2, s3, hash,
                                                  (uint64_t)n_leaves);
}
