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
use crate::field::m31::{M31, P as M31_P};
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
}
