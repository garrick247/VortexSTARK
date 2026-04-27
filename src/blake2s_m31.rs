//! Blake2s-M31 — Starknet Shinobi's hash variant.
//!
//! Standard Blake2s-256 compression, followed by a per-word Mersenne-31
//! reduction on the 8-u32 digest. This is the hash that produces values
//! safe for `HashValue<QM31>` — every output u32 is in `[0, P-1]` where
//! `P = 2^31 - 1`, so the deserializer's `val < P` check never trips.
//!
//! Source: `starkware-libs/stwo-circuits/crates/circuits/src/blake.rs`
//! (`blake_qm31`) + `stwo v2.2.0/src/core/vcs/blake2_hash.rs`
//! (`reduce_to_m31`). Rev `2591775` of stwo-circuits, fetched 2026-04-24.
//!
//! ## What upstream does (verbatim spec)
//!
//! ```text
//! fn blake_qm31(input: &[QM31], n_bytes: usize) -> HashValue<QM31> {
//!   let bytes = concat(QM31::to_bytes(x) for x in input)[..n_bytes];
//!   let digest = Blake2s256(bytes);              // standard Blake2s-256
//!   let reduced = reduce_to_m31(digest);          // per-u32-word Mersenne reduce
//!   HashValue(qm31_from_bytes(reduced[0..16]),
//!             qm31_from_bytes(reduced[16..32]))
//! }
//! ```
//!
//! `reduce_to_m31` reads the 32-byte digest as 8 little-endian u32s,
//! applies `M31::reduce` to each, and writes back as 8 LE u32s.
//! VortexSTARK's `M31::reduce` produces canonical values in `[0, P-1]`
//! (upstream produces the same range via a different bit formula — the
//! end result is identical: `P` and `2P` map to `0`).

use crate::circuit_serialize::HashValueQm31;
use crate::field::m31::M31;
#[cfg(test)]
use crate::field::m31::P as M31_P;
use crate::field::qm31::QM31;

/// Hash an arbitrary byte slice with Blake2s-M31.
///
/// Returns 8 M31-reduced output words. Every returned u32 satisfies
/// `w < 2^31 - 1` and is safe to place in an `M31` or feed to
/// `CircuitSerialize`.
pub fn blake2s_m31_bytes(input: &[u8]) -> [u32; 8] {
    use blake2::{Blake2s256, Digest as _};
    let digest_bytes: [u8; 32] = {
        let mut h = Blake2s256::new();
        h.update(input);
        h.finalize().into()
    };
    reduce_digest_to_m31(digest_bytes)
}

/// Read 8 LE u32s from a 32-byte digest and apply `M31::reduce` per word.
#[inline]
pub fn reduce_digest_to_m31(digest: [u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..8 {
        let word = u32::from_le_bytes(digest[i * 4..(i + 1) * 4].try_into().unwrap());
        // M31::reduce maps [0, 2^32) → [0, P-1]. The `word as u64` widening
        // is important — reduce() is defined over u64 so values above P
        // are folded without overflow.
        out[i] = M31::reduce(word as u64).0;
    }
    out
}

/// Upstream's `blake_qm31(input, n_bytes)` — the wire-spec hash.
///
/// `input.len() == n_bytes.div_ceil(16)` is required (upstream asserts
/// this). Trailing bytes past `n_bytes` within the last QM31 must be
/// zero; the caller is responsible for zeroing them.
pub fn blake2s_m31_qm31(input: &[QM31], n_bytes: usize) -> HashValueQm31 {
    assert_eq!(
        input.len(),
        n_bytes.div_ceil(16),
        "blake2s_m31_qm31: input QM31 count must cover n_bytes (ceil-div by 16)",
    );

    // Each QM31 serializes as 16 little-endian bytes (4 × M31 × 4 bytes).
    // We only feed `n_bytes` to the hasher — matches upstream slicing.
    let mut buf: Vec<u8> = Vec::with_capacity(input.len() * 16);
    for q in input {
        for m in q.to_m31_array() {
            buf.extend_from_slice(&m.0.to_le_bytes());
        }
    }
    let words = blake2s_m31_bytes(&buf[..n_bytes]);

    // qm31_from_bytes: 4 LE u32s → CM31(w0,w1) + CM31(w2,w3)*u.
    // Matches our `QM31::from_u32_array([a.a, a.b, b.a, b.b])` convention.
    let a = QM31::from_u32_array([words[0], words[1], words[2], words[3]]);
    let b = QM31::from_u32_array([words[4], words[5], words[6], words[7]]);
    HashValueQm31(a, b)
}

/// Convenience: take a raw standard Blake2s 8-word digest (e.g. the
/// output of our existing Merkle tree) and return its M31-reduced form.
/// Useful for cheap migration — same input, Shinobi-compatible output.
pub fn reduce_words_to_m31(digest_words: [u32; 8]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..8 {
        out[i] = M31::reduce(digest_words[i] as u64).0;
    }
    out
}

/// Reduce every u32 in a `DeviceBuffer<u32>` to the M31 range via the
/// `cuda_reduce_words_to_m31` GPU kernel. In place.
///
/// Used by the `shinobi-hash` feature to post-process each Merkle
/// layer's hash output so downstream parent-hash calls see child bytes
/// that already sit in `[0, P-1]`. Matches the CPU reduction byte-for-
/// byte (the CUDA kernel in `cuda/reduce_m31.cu` uses the identical
/// `lo = v & P; hi = v >> 31; r = lo + hi; canon(r)` formula).
#[cfg(feature = "shinobi-hash")]
pub fn reduce_device_buffer(buf: &mut crate::device::DeviceBuffer<u32>, n_words: usize) {
    use crate::cuda::ffi;
    if n_words == 0 {
        return;
    }
    unsafe {
        ffi::cuda_reduce_words_to_m31(buf.as_mut_ptr(), n_words as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_output_is_in_m31_range() {
        let out = blake2s_m31_bytes(&[]);
        for &w in &out {
            assert!(w < M31_P, "reduced word {w:#x} must be < P");
        }
    }

    #[test]
    fn reduce_maps_known_values() {
        // M31::reduce agrees with upstream reduce for endpoints:
        //   0 → 0, 1 → 1, P → 0, 2P → 0, 2^32-1 → 1.
        assert_eq!(M31::reduce(0).0, 0);
        assert_eq!(M31::reduce(1).0, 1);
        assert_eq!(M31::reduce(M31_P as u64).0, 0, "P must fold to canonical 0");
        assert_eq!(M31::reduce((M31_P as u64) * 2).0, 0);
        assert_eq!(M31::reduce(0xFFFF_FFFFu64).0, 1, "(2^32 - 1) mod P = 1");
    }

    #[test]
    fn reduce_digest_to_m31_preserves_canonical_words() {
        // If every word is already < P, reduce is a no-op.
        let mut digest = [0u8; 32];
        for i in 0..8 {
            digest[i * 4..i * 4 + 4].copy_from_slice(&(i as u32).to_le_bytes());
        }
        let out = reduce_digest_to_m31(digest);
        assert_eq!(out, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn reduce_digest_to_m31_folds_non_canonical_words() {
        // Raw Blake2s-like output with bit 31 set.
        let mut digest = [0u8; 32];
        digest[0..4].copy_from_slice(&0x8000_0001u32.to_le_bytes()); // 2^31 + 1
        digest[4..8].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // 2^32 - 1
        digest[8..12].copy_from_slice(&0x7FFF_FFFFu32.to_le_bytes()); // P
        let out = reduce_digest_to_m31(digest);
        // 2^31+1 mod P = 2
        assert_eq!(out[0], 2);
        // 2^32-1 mod P = 1
        assert_eq!(out[1], 1);
        // P mod P = 0 (canonical)
        assert_eq!(out[2], 0);
        for &w in &out {
            assert!(w < M31_P, "every output word must be < P");
        }
    }

    #[test]
    fn reduce_words_to_m31_matches_digest_version() {
        let words = [0x8000_0001u32, 0xFFFF_FFFFu32, 0x7FFF_FFFFu32, 0, 1, 2, 3, 4];
        let mut digest = [0u8; 32];
        for (i, &w) in words.iter().enumerate() {
            digest[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        assert_eq!(reduce_words_to_m31(words), reduce_digest_to_m31(digest));
    }

    #[test]
    fn blake2s_m31_qm31_output_is_wire_safe() {
        // Every component of the HashValueQm31 must be in [0, P-1] so
        // CircuitSerialize's debug_assert won't trip.
        let input: Vec<QM31> = (0..4).map(|i| {
            QM31::from_u32_array([i, i + 1, i + 2, i + 3])
        }).collect();
        let h = blake2s_m31_qm31(&input, 4 * 16);

        for m in h.0.to_m31_array().iter().chain(h.1.to_m31_array().iter()) {
            assert!(m.0 < M31_P, "output M31 {:#x} must be < P", m.0);
        }
    }

    #[test]
    fn blake2s_m31_qm31_matches_manual_pipeline() {
        // `blake2s_m31_qm31` is sugar over `blake2s_m31_bytes` + QM31 packing.
        // Verify the composition lines up.
        let input: Vec<QM31> = vec![
            QM31::from_u32_array([1, 2, 3, 4]),
            QM31::from_u32_array([5, 6, 7, 8]),
        ];
        let n_bytes = 2 * 16;

        let manual_bytes: Vec<u8> = input.iter()
            .flat_map(|q| q.to_m31_array().iter().flat_map(|m| m.0.to_le_bytes()).collect::<Vec<_>>())
            .collect();
        let manual_words = blake2s_m31_bytes(&manual_bytes[..n_bytes]);
        let manual_a = QM31::from_u32_array([manual_words[0], manual_words[1], manual_words[2], manual_words[3]]);
        let manual_b = QM31::from_u32_array([manual_words[4], manual_words[5], manual_words[6], manual_words[7]]);

        let via_api = blake2s_m31_qm31(&input, n_bytes);
        assert_eq!(via_api.0, manual_a);
        assert_eq!(via_api.1, manual_b);
    }

    #[test]
    fn blake2s_m31_qm31_honors_n_bytes_slicing() {
        // With n_bytes = 20 and 2 QM31 inputs, only the first 20 bytes of
        // the 32-byte concat feed the hasher. That should differ from
        // hashing all 32.
        let input: Vec<QM31> = vec![
            QM31::from_u32_array([1, 2, 3, 4]),
            QM31::from_u32_array([5, 6, 7, 8]),
        ];
        let h20 = blake2s_m31_qm31(&input, 20);
        let h32 = blake2s_m31_qm31(&input, 32);
        assert_ne!(h20, h32, "different n_bytes must give different digests");
    }

    #[test]
    #[should_panic(expected = "blake2s_m31_qm31: input QM31 count must cover n_bytes")]
    fn blake2s_m31_qm31_requires_matching_input_length() {
        // Upstream asserts input.len() == n_bytes.div_ceil(16).
        let input: Vec<QM31> = vec![QM31::from_u32_array([1, 2, 3, 4])];
        let _ = blake2s_m31_qm31(&input, 48); // needs 3 QM31s, has 1 → panic
    }

    /// GPU kernel (cuda/reduce_m31.cu) must produce the same output as
    /// `reduce_words_to_m31` on the CPU. This is the contract that
    /// future Merkle-tree wiring (task #43) will rely on — if the
    /// GPU reduction ever diverges from the CPU one, roots won't match.
    #[test]
    #[cfg(feature = "shinobi-hash")]
    fn gpu_reduce_matches_cpu() {
        use crate::device::DeviceBuffer;

        // Mix of canonical (< P), boundary (P, 2P), and non-canonical
        // (bit 31 set) inputs.
        let input: Vec<u32> = vec![
            0, 1, 42, M31_P - 1, M31_P, M31_P + 1, 2 * M31_P, 2 * M31_P + 1,
            0x8000_0000, 0x8000_0001, 0xFFFF_FFFE, 0xFFFF_FFFF,
            // Pad to a non-multiple of the 256-thread block to exercise
            // the tail branch.
            0xDEAD_BEEF, 0xCAFE_BABE, 0x0BAD_F00D, 0x5A5A_5A5A,
            0x1234_5678, 0x8765_4321, 0xF00D_CAFE, 0xABCDu32,
        ];
        let expected: Vec<u32> = input.iter().map(|&w| M31::reduce(w as u64).0).collect();

        let mut d = DeviceBuffer::from_host(&input);
        reduce_device_buffer(&mut d, input.len());
        let got = d.to_host();

        assert_eq!(got, expected, "GPU reduce_words_to_m31 diverges from CPU");

        // Every output word must be canonical.
        for &w in &got {
            assert!(w < M31_P, "GPU output {w:#x} must be < P");
        }
    }

    /// Bug-2 regression: re-run the leaf-hash parity check WITHOUT the
    /// shinobi-hash feature so it surfaces under any cargo-feature combo
    /// that has forge-blake2s on (e.g., forge-blake2s + forge-permute,
    /// the failing combo in test_prove_lean_matches_prove). If forge merkle
    /// produces different output than the hand-written kernel for the
    /// same input, the divergence is in the forge kernel itself rather
    /// than upstream NTT/eval data — which localizes the cross-TU bug.
    #[test]
    #[cfg(feature = "forge-blake2s")]
    fn gpu_forge_leaf_hash_parity_for_bug2() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let n_leaves: u32 = 1000;
        let column: Vec<u32> = (0..n_leaves)
            .map(|i| (i.wrapping_mul(0x9E37_79B9)) ^ 0xDEAD_BEEF)
            .collect();

        let d_col = DeviceBuffer::from_host(&column);
        let col_ptrs: Vec<*const u32> = vec![d_col.as_ptr()];
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

        let mut d_ref = DeviceBuffer::<u32>::alloc((n_leaves as usize) * 8);
        unsafe {
            ffi::cuda_merkle_hash_leaves(
                d_col_ptrs.as_ptr() as *const *const u32,
                d_ref.as_mut_ptr(),
                1,
                n_leaves,
            );
            assert_eq!(ffi::cudaDeviceSynchronize(), 0);
        }
        let ref_hashes = d_ref.to_host();

        let mut d_forge = DeviceBuffer::<u32>::alloc((n_leaves as usize) * 8);
        unsafe {
            ffi::cuda_merkle_hash_leaves_forge_single(
                d_col.as_ptr(),
                d_forge.as_mut_ptr(),
                n_leaves,
            );
            assert_eq!(ffi::cudaDeviceSynchronize(), 0);
        }
        let forge_hashes = d_forge.to_host();

        if forge_hashes != ref_hashes {
            for i in 0..n_leaves as usize {
                let lo = i * 8;
                let hi = lo + 8;
                if forge_hashes[lo..hi] != ref_hashes[lo..hi] {
                    eprintln!("[bug2] leaf {i}: forge={:08x?} ref={:08x?}",
                              &forge_hashes[lo..hi], &ref_hashes[lo..hi]);
                    if i > 4 { break; }
                }
            }
        }
        assert_eq!(forge_hashes, ref_hashes,
                   "FORGE leaf-hash diverges from hand-written under this feature combo");
    }

    /// FORGE-emitted single-column leaf-hash kernel
    /// (cuda/merkle_hash_leaves_forge.cu, generated from
    /// forge/analysis/vortex_ntt/merkle_hash_leaves.fg with 133 proof
    /// obligations discharged) must produce byte-identical output to
    /// the hand-written `cuda_merkle_hash_leaves` for n_cols = 1.
    /// This is the parity check that lets us swap one for the other.
    #[test]
    #[cfg(feature = "shinobi-hash")]
    fn gpu_forge_leaf_hash_matches_handwritten() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;
        use std::ffi::c_void;

        // Pick a leaf count that's not a multiple of the 256-thread block
        // (exercises the `leaf < n_leaves` guard inside the FORGE kernel).
        let n_leaves: u32 = 1000;
        let column: Vec<u32> = (0..n_leaves)
            .map(|i| (i.wrapping_mul(0x9E37_79B9)) ^ 0xDEAD_BEEF)
            .collect();

        // Reference: hand-written kernel via cuda_merkle_hash_leaves
        // with n_cols = 1.
        let d_col = DeviceBuffer::from_host(&column);
        let col_ptrs: Vec<*const u32> = vec![d_col.as_ptr()];
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
        let mut d_ref = DeviceBuffer::<u32>::alloc((n_leaves as usize) * 8);
        unsafe {
            ffi::cuda_merkle_hash_leaves(
                d_col_ptrs.as_ptr() as *const *const u32,
                d_ref.as_mut_ptr(),
                1,
                n_leaves,
            );
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "ref cuda_merkle_hash_leaves: cuda error {err}");
        }
        let ref_hashes = d_ref.to_host();

        // Candidate: FORGE-emitted kernel.
        let mut d_forge = DeviceBuffer::<u32>::alloc((n_leaves as usize) * 8);
        unsafe {
            ffi::cuda_merkle_hash_leaves_forge_single(
                d_col.as_ptr(),
                d_forge.as_mut_ptr(),
                n_leaves,
            );
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "forge kernel: cuda error {err}");
        }
        let forge_hashes = d_forge.to_host();

        let _ = std::ptr::null::<c_void>(); // silence unused import on some builds

        assert_eq!(
            forge_hashes, ref_hashes,
            "FORGE leaf-hash diverges from hand-written kernel"
        );
        for &w in &forge_hashes {
            assert!(w < M31_P, "FORGE output {w:#x} must be < P (M31-reduced)");
        }
    }

    /// Bug-2 quad-variant regression: same parity check as
    /// gpu_forge_leaf_hash_parity_for_bug2 but for n_cols = 4 (the
    /// shape used by quotient_commit in prove()). Catches drift
    /// without needing shinobi-hash to be on.
    #[test]
    #[cfg(feature = "forge-blake2s")]
    fn gpu_forge_leaf_hash_quad_parity_for_bug2() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let n_leaves: u32 = 1000;
        let cols: [Vec<u32>; 4] = [
            (0..n_leaves).map(|i| i.wrapping_mul(0x9E37_79B9) ^ 0xDEAD_BEEF).collect(),
            (0..n_leaves).map(|i| i.wrapping_mul(0x6A09_E667) ^ 0xCAFE_F00D).collect(),
            (0..n_leaves).map(|i| i.wrapping_mul(0xBB67_AE85) ^ 0x1234_5678).collect(),
            (0..n_leaves).map(|i| i.wrapping_mul(0x3C6E_F372) ^ 0x0BAD_F00D).collect(),
        ];
        let d_cols: Vec<DeviceBuffer<u32>> =
            cols.iter().map(|c| DeviceBuffer::from_host(c)).collect();
        let col_ptrs: Vec<*const u32> = d_cols.iter().map(|d| d.as_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

        let mut d_ref = DeviceBuffer::<u32>::alloc((n_leaves as usize) * 8);
        unsafe {
            ffi::cuda_merkle_hash_leaves(
                d_col_ptrs.as_ptr() as *const *const u32,
                d_ref.as_mut_ptr(),
                4,
                n_leaves,
            );
            assert_eq!(ffi::cudaDeviceSynchronize(), 0);
        }
        let ref_hashes = d_ref.to_host();

        let mut d_forge = DeviceBuffer::<u32>::alloc((n_leaves as usize) * 8);
        unsafe {
            ffi::cuda_merkle_hash_leaves_forge_quad(
                d_cols[0].as_ptr(), d_cols[1].as_ptr(),
                d_cols[2].as_ptr(), d_cols[3].as_ptr(),
                d_forge.as_mut_ptr(),
                n_leaves,
            );
            assert_eq!(ffi::cudaDeviceSynchronize(), 0);
        }
        let forge_hashes = d_forge.to_host();
        assert_eq!(forge_hashes, ref_hashes,
                   "FORGE quad leaf-hash diverges from hand-written under this feature combo");
    }

    /// FORGE-emitted 4-column SoA leaf-hash kernel. Same parity check
    /// as the single-column variant but for n_cols = 4 — the actual
    /// shape used by the prover for trace + composition columns.
    #[test]
    #[cfg(feature = "shinobi-hash")]
    fn gpu_forge_leaf_hash_quad_matches_handwritten() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let n_leaves: u32 = 1000;
        let cols: [Vec<u32>; 4] = [
            (0..n_leaves).map(|i| i.wrapping_mul(0x9E37_79B9) ^ 0xDEAD_BEEF).collect(),
            (0..n_leaves).map(|i| i.wrapping_mul(0x6A09_E667) ^ 0xCAFE_F00D).collect(),
            (0..n_leaves).map(|i| i.wrapping_mul(0xBB67_AE85) ^ 0x1234_5678).collect(),
            (0..n_leaves).map(|i| i.wrapping_mul(0x3C6E_F372) ^ 0x0BAD_F00D).collect(),
        ];

        let d_cols: Vec<DeviceBuffer<u32>> =
            cols.iter().map(|c| DeviceBuffer::from_host(c)).collect();
        let col_ptrs: Vec<*const u32> = d_cols.iter().map(|d| d.as_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

        // Reference: hand-written kernel with n_cols = 4
        let mut d_ref = DeviceBuffer::<u32>::alloc((n_leaves as usize) * 8);
        unsafe {
            ffi::cuda_merkle_hash_leaves(
                d_col_ptrs.as_ptr() as *const *const u32,
                d_ref.as_mut_ptr(),
                4,
                n_leaves,
            );
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "ref cuda_merkle_hash_leaves: cuda error {err}");
        }
        let ref_hashes = d_ref.to_host();

        // Candidate: FORGE quad kernel
        let mut d_forge = DeviceBuffer::<u32>::alloc((n_leaves as usize) * 8);
        unsafe {
            ffi::cuda_merkle_hash_leaves_forge_quad(
                d_cols[0].as_ptr(),
                d_cols[1].as_ptr(),
                d_cols[2].as_ptr(),
                d_cols[3].as_ptr(),
                d_forge.as_mut_ptr(),
                n_leaves,
            );
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "forge quad kernel: cuda error {err}");
        }
        let forge_hashes = d_forge.to_host();

        assert_eq!(
            forge_hashes, ref_hashes,
            "FORGE quad leaf-hash diverges from hand-written kernel"
        );
        for &w in &forge_hashes {
            assert!(w < M31_P, "FORGE quad output {w:#x} must be < P (M31-reduced)");
        }
    }

    /// FORGE-emitted gather_u32 kernel parity vs hand-written
    /// (cuda/gather.cu). FORGE kernel adds an explicit `idx[i] < src_len`
    /// guard that the hand-written kernel implicitly relies on. For
    /// in-bounds indices, output is byte-identical.
    #[test]
    fn gpu_forge_gather_u32_matches_handwritten() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let src_len: u32 = 4096;
        let n: u32 = 1000;
        let src: Vec<u32> = (0..src_len)
            .map(|i| i.wrapping_mul(0x9E37_79B9) ^ 0x1234_5678)
            .collect();
        let idx: Vec<u32> = (0..n).map(|i| (i * 17 + 3) % src_len).collect();

        let d_src = DeviceBuffer::from_host(&src);
        let d_idx = DeviceBuffer::from_host(&idx);
        let mut d_ref = DeviceBuffer::<u32>::alloc(n as usize);
        let mut d_forge = DeviceBuffer::<u32>::alloc(n as usize);

        unsafe {
            ffi::cuda_gather_u32(d_src.as_ptr(), d_idx.as_ptr(),
                                 d_ref.as_mut_ptr(), n);
            ffi::cuda_gather_u32_forge(d_src.as_ptr(), d_idx.as_ptr(),
                                       d_forge.as_mut_ptr(), n, src_len);
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "cuda error {err}");
        }
        assert_eq!(d_forge.to_host(), d_ref.to_host(),
                   "FORGE gather_u32 diverges from hand-written");
    }

    /// FORGE-emitted gather_u256 (8-word elements) kernel parity vs
    /// hand-written (cuda/gather.cu). FORGE adds explicit bounds
    /// guards on both src offset and dst offset.
    #[test]
    fn gpu_forge_gather_u256_matches_handwritten() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        // src_len is in u32 units. Each "element" is 8 u32s. Use 256
        // elements = 2048 u32s.
        let n_elements: u32 = 256;
        let src_len: u32 = n_elements * 8;
        let n: u32 = 100;
        let src: Vec<u32> = (0..src_len)
            .map(|i| i.wrapping_mul(0x6A09_E667) ^ 0xCAFE_F00D)
            .collect();
        let idx: Vec<u32> = (0..n).map(|i| (i * 7 + 1) % n_elements).collect();

        let d_src = DeviceBuffer::from_host(&src);
        let d_idx = DeviceBuffer::from_host(&idx);
        let mut d_ref = DeviceBuffer::<u32>::alloc((n as usize) * 8);
        let mut d_forge = DeviceBuffer::<u32>::alloc((n as usize) * 8);

        unsafe {
            ffi::cuda_gather_u256(d_src.as_ptr(), d_idx.as_ptr(),
                                  d_ref.as_mut_ptr(), n);
            ffi::cuda_gather_u256_forge(d_src.as_ptr(), d_idx.as_ptr(),
                                        d_forge.as_mut_ptr(), n, src_len);
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "cuda error {err}");
        }
        assert_eq!(d_forge.to_host(), d_ref.to_host(),
                   "FORGE gather_u256 diverges from hand-written");
    }

    /// FORGE bit-reverse parity vs cuda/bit_reverse_wide.cu.
    /// Verifies the 4-u32 chunk permutation matches for log_n=10
    /// (1024 QM31 elements = 4096 u32s).
    #[test]
    fn gpu_forge_bit_reverse_qm31_matches_handwritten() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let log_n: u32 = 10;
        let n: u32 = 1u32 << log_n;
        let input: Vec<u32> = (0..(n as usize) * 4)
            .map(|i| (i as u32).wrapping_mul(0x9E37_79B9) ^ 0xDEAD_BEEF)
            .collect();

        let d_in = DeviceBuffer::from_host(&input);
        let mut d_ref = DeviceBuffer::<u32>::alloc((n as usize) * 4);
        let mut d_forge = DeviceBuffer::<u32>::alloc((n as usize) * 4);

        unsafe {
            ffi::cuda_bit_reverse_qm31(d_in.as_ptr(), d_ref.as_mut_ptr(),
                                        n, log_n);
            ffi::cuda_bit_reverse_qm31_forge(d_in.as_ptr(),
                                              d_forge.as_mut_ptr(),
                                              n, log_n);
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "cuda error {err}");
        }
        assert_eq!(d_forge.to_host(), d_ref.to_host(),
                   "FORGE bit_reverse_qm31 diverges from hand-written");
    }

    /// FORGE batch-inverse parity vs cuda/batch_inverse.cu. Mixes
    /// canonical, zero, and edge-case inputs across two full chunks.
    #[test]
    fn gpu_forge_batch_inverse_m31_matches_handwritten() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        // 2 chunks of 64 each + a 23-tail to exercise the partial-chunk
        // path. Mix of 0, 1, P-1, and pseudo-random canonical values.
        let mut input: Vec<u32> = Vec::with_capacity(151);
        for i in 0..151u32 {
            let v = match i % 11 {
                0 => 0u32,                                     // zero
                1 => 1u32,                                     // identity
                2 => M31_P - 1,                                // boundary
                _ => ((i.wrapping_mul(0x9E37_79B9)) ^ 0xCAFEu32) % M31_P,
            };
            input.push(v);
        }
        let n: u32 = input.len() as u32;

        let d_in = DeviceBuffer::from_host(&input);
        let mut d_ref = DeviceBuffer::<u32>::alloc(n as usize);
        let mut d_forge = DeviceBuffer::<u32>::alloc(n as usize);

        unsafe {
            ffi::cuda_batch_inverse_m31(d_in.as_ptr(), d_ref.as_mut_ptr(), n);
            ffi::cuda_batch_inverse_m31_forge(d_in.as_ptr(),
                                                d_forge.as_mut_ptr(), n);
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "cuda error {err}");
        }
        assert_eq!(d_forge.to_host(), d_ref.to_host(),
                   "FORGE batch_inverse_m31 diverges from hand-written");
    }

    /// FORGE FRI fold_line_soa parity vs cuda/fri.cu. The QM31 SoA
    /// arithmetic is the actual hot-path FRI kernel — a real prover
    /// replacement, not a sliver.
    #[test]
    fn gpu_forge_fold_line_soa_matches_handwritten() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let half_n: u32 = 512;
        let n: u32 = half_n * 2;
        let mk = |seed: u32| -> Vec<u32> {
            (0..n).map(|i| (i.wrapping_mul(seed) ^ 0x1234_5678) % M31_P).collect()
        };
        let in0 = mk(0x9E37_79B9);
        let in1 = mk(0x6A09_E667);
        let in2 = mk(0xBB67_AE85);
        let in3 = mk(0x3C6E_F372);
        let twiddles: Vec<u32> = (0..half_n)
            .map(|i| (i.wrapping_mul(0xA54F_F53A) ^ 0xCAFEu32) % M31_P)
            .collect();
        let alpha: [u32; 4] = [123u32 % M31_P, 4567u32 % M31_P,
                                89u32 % M31_P, 0xDEADu32 % M31_P];

        let d_in = [DeviceBuffer::from_host(&in0), DeviceBuffer::from_host(&in1),
                    DeviceBuffer::from_host(&in2), DeviceBuffer::from_host(&in3)];
        let d_tw = DeviceBuffer::from_host(&twiddles);
        // alpha is passed as a host pointer (read on the host side
        // before kernel launch), so DON'T make it a DeviceBuffer.

        let mut d_ref0 = DeviceBuffer::<u32>::alloc(half_n as usize);
        let mut d_ref1 = DeviceBuffer::<u32>::alloc(half_n as usize);
        let mut d_ref2 = DeviceBuffer::<u32>::alloc(half_n as usize);
        let mut d_ref3 = DeviceBuffer::<u32>::alloc(half_n as usize);
        let mut d_f0 = DeviceBuffer::<u32>::alloc(half_n as usize);
        let mut d_f1 = DeviceBuffer::<u32>::alloc(half_n as usize);
        let mut d_f2 = DeviceBuffer::<u32>::alloc(half_n as usize);
        let mut d_f3 = DeviceBuffer::<u32>::alloc(half_n as usize);

        unsafe {
            ffi::cuda_fold_line_soa(
                d_in[0].as_ptr(), d_in[1].as_ptr(), d_in[2].as_ptr(), d_in[3].as_ptr(),
                d_tw.as_ptr(),
                d_ref0.as_mut_ptr(), d_ref1.as_mut_ptr(),
                d_ref2.as_mut_ptr(), d_ref3.as_mut_ptr(),
                alpha.as_ptr(), half_n,
            );
            ffi::cuda_fold_line_soa_forge(
                d_in[0].as_ptr(), d_in[1].as_ptr(), d_in[2].as_ptr(), d_in[3].as_ptr(),
                d_tw.as_ptr(),
                d_f0.as_mut_ptr(), d_f1.as_mut_ptr(),
                d_f2.as_mut_ptr(), d_f3.as_mut_ptr(),
                alpha.as_ptr(), half_n,
            );
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "cuda error {err}");
        }
        assert_eq!(d_f0.to_host(), d_ref0.to_host(), "FRI fold_line out0 diverges");
        assert_eq!(d_f1.to_host(), d_ref1.to_host(), "FRI fold_line out1 diverges");
        assert_eq!(d_f2.to_host(), d_ref2.to_host(), "FRI fold_line out2 diverges");
        assert_eq!(d_f3.to_host(), d_ref3.to_host(), "FRI fold_line out3 diverges");
    }

    /// FORGE FRI fold_circle_into_line_soa parity vs cuda/fri.cu.
    #[test]
    fn gpu_forge_fold_circle_into_line_soa_matches_handwritten() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let half_n: u32 = 512;
        let n: u32 = half_n * 2;
        let mk = |seed: u32, len: u32| -> Vec<u32> {
            (0..len).map(|i| (i.wrapping_mul(seed) ^ 0x1234_5678) % M31_P).collect()
        };
        let src = [mk(0x9E37_79B9, n), mk(0x6A09_E667, n),
                   mk(0xBB67_AE85, n), mk(0x3C6E_F372, n)];
        // Initial dst: a fresh accumulator full of small canonical values.
        let dst_init = [mk(0x510E_527F, half_n), mk(0x9B05_688C, half_n),
                        mk(0x1F83_D9AB, half_n), mk(0x5BE0_CD19, half_n)];
        let twiddles: Vec<u32> = (0..half_n)
            .map(|i| (i.wrapping_mul(0xA54F_F53A) ^ 0xCAFEu32) % M31_P)
            .collect();
        let alpha: [u32; 4] = [123u32 % M31_P, 4567u32 % M31_P,
                                89u32 % M31_P, 0xDEADu32 % M31_P];
        let alpha_sq: [u32; 4] = [97u32 % M31_P, 0xBEEFu32 % M31_P,
                                   42u32 % M31_P, 0xCAFEu32 % M31_P];

        let d_src: Vec<DeviceBuffer<u32>> = src.iter()
            .map(|s| DeviceBuffer::from_host(s)).collect();
        let d_tw = DeviceBuffer::from_host(&twiddles);
        // alpha and alpha_sq are passed as host pointers — leave them
        // as `[u32; 4]` arrays.

        // Fresh dst buffers per implementation so each gets an
        // independent in-place mutation.
        let mut d_ref: Vec<DeviceBuffer<u32>> = dst_init.iter()
            .map(|s| DeviceBuffer::from_host(s)).collect();
        let mut d_forge: Vec<DeviceBuffer<u32>> = dst_init.iter()
            .map(|s| DeviceBuffer::from_host(s)).collect();

        unsafe {
            ffi::cuda_fold_circle_into_line_soa(
                d_ref[0].as_mut_ptr(), d_ref[1].as_mut_ptr(),
                d_ref[2].as_mut_ptr(), d_ref[3].as_mut_ptr(),
                d_src[0].as_ptr(), d_src[1].as_ptr(),
                d_src[2].as_ptr(), d_src[3].as_ptr(),
                d_tw.as_ptr(), alpha.as_ptr(), alpha_sq.as_ptr(),
                half_n,
            );
            ffi::cuda_fold_circle_into_line_soa_forge(
                d_forge[0].as_mut_ptr(), d_forge[1].as_mut_ptr(),
                d_forge[2].as_mut_ptr(), d_forge[3].as_mut_ptr(),
                d_src[0].as_ptr(), d_src[1].as_ptr(),
                d_src[2].as_ptr(), d_src[3].as_ptr(),
                d_tw.as_ptr(), alpha.as_ptr(), alpha_sq.as_ptr(),
                half_n,
            );
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "cuda error {err}");
        }
        for k in 0..4 {
            assert_eq!(d_forge[k].to_host(), d_ref[k].to_host(),
                       "FRI fold_circle dst{k} diverges");
        }
    }

    /// Phase3 slab-kernel parity vs full kernel.
    ///
    /// Validates that `cuda_cairo_quotient_slab` (chunk_n = eval_size,
    /// chunk_offset = 0, slab_mode = 1) produces the same output as
    /// `cuda_cairo_quotient` (slab_mode = 0). The slab path uses
    /// per-col extended buffers with the trailing `blowup_step` rows
    /// being a copy of the FIRST `blowup_step` rows of the col (the
    /// wrap rule the kernel doc specifies for the last chunk).
    ///
    /// This test doesn't exercise multi-chunk explicitly, but it
    /// covers the slab_mode code path (`g_idx = chunk_offset + i`,
    /// `next_i = i + blowup_step`) end-to-end, including the wrap.
    /// Multi-chunk parity would only add the extra step of splitting
    /// the slab into N pieces, which doesn't add new kernel paths.
    ///
    /// Inputs are synthetic random u32s in M31 range — the kernel's
    /// output is meaningless as a Cairo quotient, but the bit-equality
    /// of full vs slab mode is the property we're testing.
    #[test]
    fn gpu_cairo_quotient_slab_matches_full() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let eval_size: u32 = 256;
        let blowup_step: u32 = 1 << crate::prover::BLOWUP_BITS;
        let mut rng_state: u32 = 0xDEAD_BEEF;
        let mut next_u32 = || -> u32 {
            // xorshift32, masked to M31
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state & M31_P
        };

        // ── Shared input buffers (random M31s) ──
        let trace_full: Vec<u32> = (0..eval_size).map(|_| next_u32()).collect();
        let side_full:  Vec<u32> = (0..eval_size).map(|_| next_u32()).collect();
        let alpha:    [u32; 4] = [next_u32(), next_u32(), next_u32(), next_u32()];
        let challenges_flat: Vec<u32> = (0..24).map(|_| next_u32()).collect();
        let vh_inv:        Vec<u32> = (0..eval_size).map(|_| next_u32()).collect();
        let trans_factor:  Vec<u32> = (0..eval_size).map(|_| next_u32()).collect();

        let d_trace_full = DeviceBuffer::from_host(&trace_full);
        let d_side_full  = DeviceBuffer::from_host(&side_full);
        let d_vh_inv     = DeviceBuffer::from_host(&vh_inv);
        let d_trans_factor = DeviceBuffer::from_host(&trans_factor);
        let d_alpha    = DeviceBuffer::from_host(&alpha);
        let d_challenges = DeviceBuffer::from_host(&challenges_flat);

        // ── Slab buffers: full + first `blowup_step` rows (wrap) ──
        let trace_slab: Vec<u32> = trace_full.iter().copied()
            .chain(trace_full.iter().take(blowup_step as usize).copied())
            .collect();
        let side_slab: Vec<u32> = side_full.iter().copied()
            .chain(side_full.iter().take(blowup_step as usize).copied())
            .collect();
        let d_trace_slab = DeviceBuffer::from_host(&trace_slab);
        let d_side_slab  = DeviceBuffer::from_host(&side_slab);

        // ── 34 trace col pointers (all share the same buffer) ──
        let trace_full_ptrs: Vec<*const u32> = (0..34).map(|_| d_trace_full.as_ptr()).collect();
        let trace_slab_ptrs: Vec<*const u32> = (0..34).map(|_| d_trace_slab.as_ptr()).collect();
        let d_trace_full_ptrs = DeviceBuffer::from_host(&trace_full_ptrs);
        let d_trace_slab_ptrs = DeviceBuffer::from_host(&trace_slab_ptrs);

        // Outputs
        let mut q_ref:  [DeviceBuffer<u32>; 4] = std::array::from_fn(
            |_| DeviceBuffer::<u32>::alloc(eval_size as usize));
        let mut q_slab: [DeviceBuffer<u32>; 4] = std::array::from_fn(
            |_| DeviceBuffer::<u32>::alloc(eval_size as usize));

        // Convenience aliases — all side cols share `d_side_full`/`d_side_slab`.
        let f = d_side_full.as_ptr();
        let s = d_side_slab.as_ptr();

        unsafe {
            ffi::cuda_cairo_quotient(
                d_trace_full_ptrs.as_ptr() as *const *const u32,
                f, f, f, f, // s_logup
                f, f, f, f, f, f, f, f, f, f, f, f, // t1l, t2l, t3l
                f, f, f, f, // s_rc
                f, f, f, f, f, f, f, f, // u1r, u2r
                f, f, f, f, // s_dict
                q_ref[0].as_mut_ptr(), q_ref[1].as_mut_ptr(),
                q_ref[2].as_mut_ptr(), q_ref[3].as_mut_ptr(),
                d_alpha.as_ptr(), d_vh_inv.as_ptr(),
                d_trans_factor.as_ptr(), d_challenges.as_ptr(),
                eval_size, blowup_step,
            );
            ffi::cuda_cairo_quotient_slab(
                d_trace_slab_ptrs.as_ptr() as *const *const u32,
                s, s, s, s,
                s, s, s, s, s, s, s, s, s, s, s, s,
                s, s, s, s,
                s, s, s, s, s, s, s, s,
                s, s, s, s,
                q_slab[0].as_mut_ptr(), q_slab[1].as_mut_ptr(),
                q_slab[2].as_mut_ptr(), q_slab[3].as_mut_ptr(),
                d_alpha.as_ptr(), d_vh_inv.as_ptr(),
                d_trans_factor.as_ptr(), d_challenges.as_ptr(),
                /*chunk_n=*/eval_size, blowup_step, /*chunk_offset=*/0,
            );
            let err = ffi::cudaDeviceSynchronize();
            assert_eq!(err, 0, "cuda error {err}");
        }

        for k in 0..4 {
            assert_eq!(q_slab[k].to_host(), q_ref[k].to_host(),
                       "slab kernel q{k} diverges from full kernel");
        }
    }

    /// Phase3 multi-chunk parity. Splits the work into 4 chunks and
    /// verifies the assembled output matches the full single-shot
    /// kernel byte-for-byte. This is the test that says "row-chunking
    /// at log_n=26 will produce a correct quotient" — the slab kernel
    /// + dispatcher discharge it without needing actual VRAM streaming
    /// (which is a separate prover refactor).
    #[test]
    fn gpu_cairo_quotient_multi_chunk_matches_full() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let eval_size: u32 = 256;
        let n_chunks: u32 = 4;
        let chunk_n: u32 = eval_size / n_chunks;
        let blowup_step: u32 = 1 << crate::prover::BLOWUP_BITS;
        assert!(blowup_step <= chunk_n, "blowup_step must fit in chunk_n");

        let mut rng_state: u32 = 0xCAFE_F00D;
        let mut next_u32 = || -> u32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            rng_state & M31_P
        };

        let trace_full: Vec<u32> = (0..eval_size).map(|_| next_u32()).collect();
        let side_full:  Vec<u32> = (0..eval_size).map(|_| next_u32()).collect();
        let alpha:    [u32; 4] = [next_u32(), next_u32(), next_u32(), next_u32()];
        let challenges_flat: Vec<u32> = (0..24).map(|_| next_u32()).collect();
        let vh_inv:        Vec<u32> = (0..eval_size).map(|_| next_u32()).collect();
        let trans_factor:  Vec<u32> = (0..eval_size).map(|_| next_u32()).collect();

        let d_trace_full = DeviceBuffer::from_host(&trace_full);
        let d_side_full  = DeviceBuffer::from_host(&side_full);
        let d_vh_inv     = DeviceBuffer::from_host(&vh_inv);
        let d_trans_factor = DeviceBuffer::from_host(&trans_factor);
        let d_alpha    = DeviceBuffer::from_host(&alpha);
        let d_challenges = DeviceBuffer::from_host(&challenges_flat);

        // ── Reference: full kernel ──
        let trace_full_ptrs: Vec<*const u32> = (0..34).map(|_| d_trace_full.as_ptr()).collect();
        let d_trace_full_ptrs = DeviceBuffer::from_host(&trace_full_ptrs);
        let mut q_ref:  [DeviceBuffer<u32>; 4] = std::array::from_fn(
            |_| DeviceBuffer::<u32>::alloc(eval_size as usize));
        let f = d_side_full.as_ptr();
        unsafe {
            ffi::cuda_cairo_quotient(
                d_trace_full_ptrs.as_ptr() as *const *const u32,
                f, f, f, f,
                f, f, f, f, f, f, f, f, f, f, f, f,
                f, f, f, f,
                f, f, f, f, f, f, f, f,
                f, f, f, f,
                q_ref[0].as_mut_ptr(), q_ref[1].as_mut_ptr(),
                q_ref[2].as_mut_ptr(), q_ref[3].as_mut_ptr(),
                d_alpha.as_ptr(), d_vh_inv.as_ptr(),
                d_trans_factor.as_ptr(), d_challenges.as_ptr(),
                eval_size, blowup_step,
            );
            ffi::cudaDeviceSynchronize();
        }

        // ── Chunked: N_CHUNKS calls to slab kernel ──
        // For chunks k in [0, N-1), slab pointer = col + chunk_offset
        //   (the +chunk_n + blowup_step rows fit within eval_size).
        // For the last chunk, build a wrap-extended buffer:
        //   slab[0..chunk_n] = col[(N-1)*chunk_n .. eval_size]
        //   slab[chunk_n..chunk_n+blowup_step] = col[0..blowup_step]
        let mut q_chunked: [DeviceBuffer<u32>; 4] = std::array::from_fn(
            |_| DeviceBuffer::<u32>::alloc(eval_size as usize));

        let last_start = (n_chunks - 1) * chunk_n;
        let mut wrap_trace = trace_full[last_start as usize..].to_vec();
        wrap_trace.extend_from_slice(&trace_full[..blowup_step as usize]);
        let mut wrap_side = side_full[last_start as usize..].to_vec();
        wrap_side.extend_from_slice(&side_full[..blowup_step as usize]);
        let d_wrap_trace = DeviceBuffer::from_host(&wrap_trace);
        let d_wrap_side  = DeviceBuffer::from_host(&wrap_side);

        for k in 0..n_chunks {
            let chunk_offset = k * chunk_n;
            let trace_slab_ptr = if k == n_chunks - 1 {
                d_wrap_trace.as_ptr()
            } else {
                unsafe { d_trace_full.as_ptr().add(chunk_offset as usize) }
            };
            let side_slab_ptr = if k == n_chunks - 1 {
                d_wrap_side.as_ptr()
            } else {
                unsafe { d_side_full.as_ptr().add(chunk_offset as usize) }
            };
            let trace_slab_ptrs: Vec<*const u32> = (0..34).map(|_| trace_slab_ptr).collect();
            let d_trace_slab_ptrs = DeviceBuffer::from_host(&trace_slab_ptrs);
            let s = side_slab_ptr;

            unsafe {
                ffi::cuda_cairo_quotient_slab(
                    d_trace_slab_ptrs.as_ptr() as *const *const u32,
                    s, s, s, s,
                    s, s, s, s, s, s, s, s, s, s, s, s,
                    s, s, s, s,
                    s, s, s, s, s, s, s, s,
                    s, s, s, s,
                    q_chunked[0].as_mut_ptr(), q_chunked[1].as_mut_ptr(),
                    q_chunked[2].as_mut_ptr(), q_chunked[3].as_mut_ptr(),
                    d_alpha.as_ptr(), d_vh_inv.as_ptr(),
                    d_trans_factor.as_ptr(), d_challenges.as_ptr(),
                    chunk_n, blowup_step, chunk_offset,
                );
            }
        }
        unsafe { ffi::cudaDeviceSynchronize(); }

        for k in 0..4 {
            assert_eq!(q_chunked[k].to_host(), q_ref[k].to_host(),
                       "multi-chunk slab kernel q{k} diverges from full kernel");
        }
    }

    /// FORGE Circle NTT butterfly layer parity vs cuda/circle_ntt.cu.
    /// Tests both forward and inverse on the same input. Single layer
    /// (layer_idx=3) on 2^10 points.
    #[test]
    fn gpu_forge_circle_ntt_layer_matches_handwritten() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;

        let log_n: u32 = 10;
        let n: u32 = 1 << log_n;
        let half_n: u32 = n / 2;
        let layer_idx: u32 = 3;
        let data: Vec<u32> = (0..n)
            .map(|i| (i.wrapping_mul(0x9E37_79B9) ^ 0x1234_5678) % M31_P)
            .collect();
        let twiddles: Vec<u32> = (0..half_n)
            .map(|i| (i.wrapping_mul(0x6A09_E667) ^ 0xCAFE) % M31_P)
            .collect();

        let d_tw = DeviceBuffer::from_host(&twiddles);

        for forward in [1, 0] {
            let mut d_ref = DeviceBuffer::from_host(&data);
            let mut d_forge = DeviceBuffer::from_host(&data);
            unsafe {
                ffi::cuda_circle_ntt_layer(
                    d_ref.as_mut_ptr(), d_tw.as_ptr(), layer_idx, n, forward);
                ffi::cuda_circle_ntt_layer_forge(
                    d_forge.as_mut_ptr(), d_tw.as_ptr(), layer_idx, n, forward);
                let err = ffi::cudaDeviceSynchronize();
                assert_eq!(err, 0, "cuda error {err}");
            }
            assert_eq!(
                d_forge.to_host(), d_ref.to_host(),
                "Circle NTT layer (forward={forward}) diverges"
            );
        }
    }

    /// Microbenchmark: FORGE-emitted vs hand-written kernels.
    /// Run with `cargo test --release ... gpu_forge_microbench
    /// --features forge-blake2s -- --ignored --nocapture`.
    /// Compares wall-clock time for 100 iterations on a representative
    /// problem size, reports a ratio. The cuda runtime serializes
    /// kernels on the default stream so this isn't a perfect
    /// throughput benchmark — it's a launch-overhead + per-thread
    /// instruction-count comparison.
    #[test]
    #[ignore]
    fn gpu_forge_microbench() {
        use crate::cuda::ffi;
        use crate::device::DeviceBuffer;
        use std::time::Instant;

        let iters: u32 = 100;

        // ── 1. M31 batch inverse ──
        {
            let n: u32 = 1 << 20; // 1M elements
            let input: Vec<u32> = (0..n).map(|i| (i + 1) % M31_P).collect();
            let d_in = DeviceBuffer::from_host(&input);
            let mut d_out_ref = DeviceBuffer::<u32>::alloc(n as usize);
            let mut d_out_forge = DeviceBuffer::<u32>::alloc(n as usize);

            // warmup
            unsafe {
                ffi::cuda_batch_inverse_m31(d_in.as_ptr(), d_out_ref.as_mut_ptr(), n);
                ffi::cuda_batch_inverse_m31_forge(d_in.as_ptr(), d_out_forge.as_mut_ptr(), n);
                ffi::cudaDeviceSynchronize();
            }

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_batch_inverse_m31(d_in.as_ptr(), d_out_ref.as_mut_ptr(), n);
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_ref = t0.elapsed();

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_batch_inverse_m31_forge(d_in.as_ptr(), d_out_forge.as_mut_ptr(), n);
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_forge = t0.elapsed();

            let ratio = t_forge.as_secs_f64() / t_ref.as_secs_f64();
            println!("batch_inverse_m31  n=2^20  ref={:?}  forge={:?}  ratio={:.3}x",
                     t_ref / iters, t_forge / iters, ratio);
        }

        // ── 2. QM31 bit_reverse ──
        {
            let log_n: u32 = 18;
            let n: u32 = 1 << log_n;
            let input: Vec<u32> = (0..(n * 4)).collect();
            let d_in = DeviceBuffer::from_host(&input);
            let mut d_ref = DeviceBuffer::<u32>::alloc((n * 4) as usize);
            let mut d_forge = DeviceBuffer::<u32>::alloc((n * 4) as usize);

            unsafe {
                ffi::cuda_bit_reverse_qm31(d_in.as_ptr(), d_ref.as_mut_ptr(), n, log_n);
                ffi::cuda_bit_reverse_qm31_forge(d_in.as_ptr(), d_forge.as_mut_ptr(), n, log_n);
                ffi::cudaDeviceSynchronize();
            }

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_bit_reverse_qm31(d_in.as_ptr(), d_ref.as_mut_ptr(), n, log_n);
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_ref = t0.elapsed();

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_bit_reverse_qm31_forge(d_in.as_ptr(), d_forge.as_mut_ptr(), n, log_n);
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_forge = t0.elapsed();

            let ratio = t_forge.as_secs_f64() / t_ref.as_secs_f64();
            println!("bit_reverse_qm31   n=2^{}  ref={:?}  forge={:?}  ratio={:.3}x",
                     log_n, t_ref / iters, t_forge / iters, ratio);
        }

        // ── 3. FRI fold_line ──
        {
            let half_n: u32 = 1 << 19;
            let n: u32 = half_n * 2;
            let mk = |seed: u32, len: u32| -> Vec<u32> {
                (0..len).map(|i| (i.wrapping_mul(seed) ^ 0x12345678) % M31_P).collect()
            };
            let cols = [mk(1, n), mk(2, n), mk(3, n), mk(4, n)];
            let twiddles = mk(5, half_n);
            let alpha: [u32; 4] = [123, 4567, 89, 0xDEADu32 % M31_P];

            let d_cols: Vec<DeviceBuffer<u32>> =
                cols.iter().map(|c| DeviceBuffer::from_host(c)).collect();
            let d_tw = DeviceBuffer::from_host(&twiddles);
            let mut d_out: Vec<DeviceBuffer<u32>> =
                (0..4).map(|_| DeviceBuffer::<u32>::alloc(half_n as usize)).collect();
            let mut d_out_forge: Vec<DeviceBuffer<u32>> =
                (0..4).map(|_| DeviceBuffer::<u32>::alloc(half_n as usize)).collect();

            unsafe {
                ffi::cuda_fold_line_soa(
                    d_cols[0].as_ptr(), d_cols[1].as_ptr(),
                    d_cols[2].as_ptr(), d_cols[3].as_ptr(),
                    d_tw.as_ptr(),
                    d_out[0].as_mut_ptr(), d_out[1].as_mut_ptr(),
                    d_out[2].as_mut_ptr(), d_out[3].as_mut_ptr(),
                    alpha.as_ptr(), half_n,
                );
                ffi::cuda_fold_line_soa_forge(
                    d_cols[0].as_ptr(), d_cols[1].as_ptr(),
                    d_cols[2].as_ptr(), d_cols[3].as_ptr(),
                    d_tw.as_ptr(),
                    d_out_forge[0].as_mut_ptr(), d_out_forge[1].as_mut_ptr(),
                    d_out_forge[2].as_mut_ptr(), d_out_forge[3].as_mut_ptr(),
                    alpha.as_ptr(), half_n,
                );
                ffi::cudaDeviceSynchronize();
            }

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_fold_line_soa(
                        d_cols[0].as_ptr(), d_cols[1].as_ptr(),
                        d_cols[2].as_ptr(), d_cols[3].as_ptr(),
                        d_tw.as_ptr(),
                        d_out[0].as_mut_ptr(), d_out[1].as_mut_ptr(),
                        d_out[2].as_mut_ptr(), d_out[3].as_mut_ptr(),
                        alpha.as_ptr(), half_n,
                    );
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_ref = t0.elapsed();

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_fold_line_soa_forge(
                        d_cols[0].as_ptr(), d_cols[1].as_ptr(),
                        d_cols[2].as_ptr(), d_cols[3].as_ptr(),
                        d_tw.as_ptr(),
                        d_out_forge[0].as_mut_ptr(), d_out_forge[1].as_mut_ptr(),
                        d_out_forge[2].as_mut_ptr(), d_out_forge[3].as_mut_ptr(),
                        alpha.as_ptr(), half_n,
                    );
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_forge = t0.elapsed();

            let ratio = t_forge.as_secs_f64() / t_ref.as_secs_f64();
            println!("fri_fold_line_soa  half_n=2^19  ref={:?}  forge={:?}  ratio={:.3}x",
                     t_ref / iters, t_forge / iters, ratio);
        }

        // ── 4. NTT butterfly layer (forward) ──
        {
            let log_n: u32 = 20;
            let n: u32 = 1 << log_n;
            let half_n: u32 = n / 2;
            let layer_idx: u32 = 5;
            let data: Vec<u32> = (0..n)
                .map(|i| (i.wrapping_mul(0x9E37_79B9) ^ 0x1234_5678) % M31_P)
                .collect();
            let twiddles: Vec<u32> = (0..half_n)
                .map(|i| (i.wrapping_mul(0x6A09_E667) ^ 0xCAFE) % M31_P)
                .collect();
            let d_tw = DeviceBuffer::from_host(&twiddles);
            let mut d_ref = DeviceBuffer::from_host(&data);
            let mut d_forge = DeviceBuffer::from_host(&data);

            unsafe {
                ffi::cuda_circle_ntt_layer(d_ref.as_mut_ptr(), d_tw.as_ptr(),
                                            layer_idx, n, 1);
                ffi::cuda_circle_ntt_layer_forge(d_forge.as_mut_ptr(),
                                                  d_tw.as_ptr(),
                                                  layer_idx, n, 1);
                ffi::cudaDeviceSynchronize();
            }

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_circle_ntt_layer(d_ref.as_mut_ptr(),
                                                d_tw.as_ptr(),
                                                layer_idx, n, 1);
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_ref = t0.elapsed();

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_circle_ntt_layer_forge(d_forge.as_mut_ptr(),
                                                      d_tw.as_ptr(),
                                                      layer_idx, n, 1);
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_forge = t0.elapsed();

            let ratio = t_forge.as_secs_f64() / t_ref.as_secs_f64();
            println!("circle_ntt_layer   n=2^{}  layer={}  ref={:?}  forge={:?}  ratio={:.3}x",
                     log_n, layer_idx, t_ref / iters, t_forge / iters, ratio);
        }

        // ── 5. Blake2s leaf hash, 4-col ──
        {
            let n_leaves: u32 = 1 << 20;
            let mk = |seed: u32| -> Vec<u32> {
                (0..n_leaves).map(|i| i.wrapping_mul(seed) ^ 0x12345678).collect()
            };
            let cols = [mk(1), mk(2), mk(3), mk(4)];
            let d_cols: Vec<DeviceBuffer<u32>> =
                cols.iter().map(|c| DeviceBuffer::from_host(c)).collect();
            let col_ptrs: Vec<*const u32> = d_cols.iter().map(|d| d.as_ptr()).collect();
            let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);
            let mut d_ref = DeviceBuffer::<u32>::alloc((n_leaves * 8) as usize);
            let mut d_forge = DeviceBuffer::<u32>::alloc((n_leaves * 8) as usize);

            unsafe {
                ffi::cuda_merkle_hash_leaves(
                    d_col_ptrs.as_ptr() as *const *const u32,
                    d_ref.as_mut_ptr(), 4, n_leaves);
                ffi::cuda_merkle_hash_leaves_forge_quad(
                    d_cols[0].as_ptr(), d_cols[1].as_ptr(),
                    d_cols[2].as_ptr(), d_cols[3].as_ptr(),
                    d_forge.as_mut_ptr(), n_leaves);
                ffi::cudaDeviceSynchronize();
            }

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_merkle_hash_leaves(
                        d_col_ptrs.as_ptr() as *const *const u32,
                        d_ref.as_mut_ptr(), 4, n_leaves);
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_ref = t0.elapsed();

            let t0 = Instant::now();
            unsafe {
                for _ in 0..iters {
                    ffi::cuda_merkle_hash_leaves_forge_quad(
                        d_cols[0].as_ptr(), d_cols[1].as_ptr(),
                        d_cols[2].as_ptr(), d_cols[3].as_ptr(),
                        d_forge.as_mut_ptr(), n_leaves);
                }
                ffi::cudaDeviceSynchronize();
            }
            let t_forge = t0.elapsed();

            let ratio = t_forge.as_secs_f64() / t_ref.as_secs_f64();
            println!("merkle_quad        n=2^20  ref={:?}  forge={:?}  ratio={:.3}x",
                     t_ref / iters, t_forge / iters, ratio);
        }
    }
}
