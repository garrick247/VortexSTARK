//! Starknet Shinobi wire format — `circuit_serialize`.
//!
//! Faithful re-implementation of
//! `starkware-libs/stwo-circuits/crates/circuit_serialize/src/serialize.rs`
//! (rev `2591775`, main @ 2026-04). The format is a hand-rolled,
//! length-prefix-free, zero-tag binary layout consumed by
//! `privacy_circuit_verify::verify_cairo`. The deserializer reconstructs
//! lengths from a separate `ProofConfig`, so every byte we emit must be
//! exactly in the order and size that the verifier's config expects.
//!
//! Reference (upstream `serialize.rs` lines 52–106):
//!
//! ```text
//! M31            → u32 LE (4 bytes)
//! QM31           → [M31; 4]     (16 bytes, order: a.a, a.b, b.a, b.b)
//! HashValue<QM31>→ (QM31, QM31) (32 bytes)
//! [T]/[T;N]/Vec  → concatenation, no length prefix
//! Claim          → packed log sizes (1 byte each, padded to u32 multiple)
//!                  then claimed_sums as Vec<QM31>
//! InteractionAtOods → at_oods, then at_prev iff Some (no tag)
//! ```
//!
//! ## What this module DOES give you
//!
//! Exact-match primitive encoders for the wire leaves — M31, QM31,
//! HashValue, slice/array/Vec — plus the outer `zstd::encode_all` framing
//! and the `verify_cairo` public-claim prefix.
//!
//! ## What this module does NOT give you
//!
//! A byte-for-byte encode of a VortexSTARK `CairoProof` that the Shinobi
//! verifier will accept. The upstream top-level `Proof<QM31>` has a
//! different *shape* than our `CairoProof` — different commitment trees,
//! different OODS layout, different FRI config constants — and matching
//! it requires the AIR itself to match, not just the serializer. See
//! `src/snip36.rs` for the full caveat.
//!
//! ## zstd level
//!
//! The sequencer's `privacy_prove::compress_proof` calls
//! `zstd::encode_all(proof_bytes, 3)`. Anything else will decompress but
//! will not round-trip through the exact-match tests the Cairo
//! verifier's test vectors use, so we pin level 3 explicitly.

use crate::field::m31::{M31, P as M31_P};
use crate::field::qm31::QM31;

/// Trait matching upstream `circuit_serialize::CircuitSerialize`.
///
/// Implementations MUST match upstream byte-for-byte.
pub trait CircuitSerialize {
    fn serialize(&self, output: &mut Vec<u8>);
}

impl CircuitSerialize for M31 {
    fn serialize(&self, output: &mut Vec<u8>) {
        // Upstream: `output.extend_from_slice(&self.0.to_le_bytes());`
        // Invariant: `self.0 < P = 2^31 - 1`. Debug-assert it so tests
        // catch any caller that stuffs a raw digest word into M31.
        debug_assert!(self.0 < M31_P, "M31 out of range: {}", self.0);
        output.extend_from_slice(&self.0.to_le_bytes());
    }
}

impl CircuitSerialize for QM31 {
    fn serialize(&self, output: &mut Vec<u8>) {
        // Upstream: iterate `self.to_m31_array()` and serialize each.
        for m in self.to_m31_array() {
            m.serialize(output);
        }
    }
}

impl<T: CircuitSerialize> CircuitSerialize for [T] {
    fn serialize(&self, output: &mut Vec<u8>) {
        for v in self {
            v.serialize(output);
        }
    }
}

impl<T: CircuitSerialize, const N: usize> CircuitSerialize for [T; N] {
    fn serialize(&self, output: &mut Vec<u8>) {
        self.as_slice().serialize(output);
    }
}

impl<T: CircuitSerialize> CircuitSerialize for Vec<T> {
    fn serialize(&self, output: &mut Vec<u8>) {
        self.as_slice().serialize(output);
    }
}

/// Wire twin of upstream `circuits::blake::HashValue<QM31>`.
/// On the wire: two QM31s back-to-back = 32 bytes.
///
/// NOTE: bit-31 of every constituent u32 MUST be 0 (M31 range). Raw
/// Blake2s outputs routinely have bit 31 set, so feeding a 32-byte
/// digest directly here will panic under debug. Upstream uses
/// `circuits::blake::Blake2sM31` which clamps each output word to 31
/// bits at the hash level. VortexSTARK's Merkle tree ships with
/// standard Blake2s today — aligning to Blake2sM31 is part of the
/// AIR-alignment work tracked in SHINOBI_COMPAT_PLAN (future).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashValueQm31(pub QM31, pub QM31);

impl HashValueQm31 {
    /// Build a HashValueQm31 from 8 u32 words (e.g. an M31-clamped
    /// Blake2 digest). Each word is debug-checked against `M31::P`.
    pub fn from_words(w: [u32; 8]) -> Self {
        let a = QM31::from_u32_array([w[0], w[1], w[2], w[3]]);
        let b = QM31::from_u32_array([w[4], w[5], w[6], w[7]]);
        Self(a, b)
    }
}

impl CircuitSerialize for HashValueQm31 {
    fn serialize(&self, output: &mut Vec<u8>) {
        // Upstream: `a.serialize(output); b.serialize(output);`.
        self.0.serialize(output);
        self.1.serialize(output);
    }
}

/// Serialize a `Claim` — packed per-component log sizes followed by
/// `claimed_sums: Vec<QM31>`.
///
/// Upstream layout:
///   for each `QM31` in `packed_component_log_sizes`:
///     for each of its 4 `M31`s:
///       output.push((m31.0 & 0xFF) as u8)
///   claimed_sums.serialize(output)
///
/// Callers pass log sizes as raw bytes (one per component), padded to a
/// multiple of 4 — matching the upstream unpack step
/// `n_components.next_multiple_of(4)`.
pub fn serialize_claim(
    packed_log_size_bytes: &[u8],
    claimed_sums: &[QM31],
    output: &mut Vec<u8>,
) {
    assert!(
        packed_log_size_bytes.len() % 4 == 0,
        "log-size pack must be padded to a multiple of 4 bytes"
    );
    output.extend_from_slice(packed_log_size_bytes);
    claimed_sums.serialize(output);
}

/// Pack a slice of per-component log sizes (each `<= u8::MAX`) into the
/// `(n_components.next_multiple_of(4))`-byte layout Shinobi expects.
pub fn pack_component_log_sizes(log_sizes: &[u32]) -> Vec<u8> {
    assert!(
        log_sizes.iter().all(|&v| v <= u8::MAX as u32),
        "component log sizes must fit in a byte — upstream asserts LOG_SIZE_BITS <= 8",
    );
    let padded = log_sizes.len().next_multiple_of(4);
    let mut out = Vec::with_capacity(padded);
    out.extend(log_sizes.iter().map(|&v| v as u8));
    out.resize(padded, 0);
    out
}

/// Wire twin of upstream `InteractionAtOods<QM31>`: `at_oods` always,
/// followed by `at_prev` only if `Some` (no tag byte — reader knows
/// from `ProofConfig.cumulative_sum_columns[i]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteractionAtOods {
    pub at_oods: QM31,
    pub at_prev: Option<QM31>,
}

impl CircuitSerialize for InteractionAtOods {
    fn serialize(&self, output: &mut Vec<u8>) {
        self.at_oods.serialize(output);
        if let Some(ap) = self.at_prev {
            ap.serialize(output);
        }
    }
}

// ── Outer envelope ────────────────────────────────────────────────────────────

/// Shinobi-fixed constants. Match `privacy_circuit_verify/src/consts.rs`
/// and `stwo-circuits/crates/cairo_air/src/statement.rs` at rev
/// `2591775`.
///
/// `PUBLIC_DATA_LEN = 2*STATE_LEN + 2*PUB_MEMORY_VALUE_M31_LEN*N_SEGMENTS
///                  + N_SAFE_CALL_IDS`
/// where `STATE_LEN = 3`, `PUB_MEMORY_VALUE_M31_LEN = 2`, `N_SEGMENTS = 11`,
/// `N_SAFE_CALL_IDS = 2` ⇒ `6 + 44 + 2 = 52`.
pub const PUBLIC_DATA_LEN: usize = 52;

/// Sequencer constant (`privacy_circuit_verify::consts::NUM_OUTPUTS`).
pub const NUM_OUTPUTS: usize = 1;

/// zstd level used by `privacy_prove::compress_proof`. Pinned at 3.
pub const ZSTD_LEVEL: i32 = 3;

/// Build the `(PUBLIC_DATA_LEN + NUM_OUTPUTS + program_len) * 4`-byte
/// u32-LE prefix that precedes the serialized proof in the `verify_cairo`
/// (non-recursive) path. The recursive path (the one Invoke V3 uses)
/// has no prefix — use [`frame_with_zstd`] on the proof bytes directly.
pub fn public_claim_prefix_bytes(public_data: &[u32], outputs: &[u32], program: &[u32]) -> Vec<u8> {
    assert_eq!(
        public_data.len(),
        PUBLIC_DATA_LEN,
        "public_data must have exactly PUBLIC_DATA_LEN ({PUBLIC_DATA_LEN}) elements",
    );
    assert_eq!(
        outputs.len(),
        NUM_OUTPUTS,
        "outputs must have exactly NUM_OUTPUTS ({NUM_OUTPUTS}) elements",
    );
    let total_words = public_data.len() + outputs.len() + program.len();
    let mut out = Vec::with_capacity(total_words * 4);
    for &w in public_data {
        out.extend_from_slice(&w.to_le_bytes());
    }
    for &w in outputs {
        out.extend_from_slice(&w.to_le_bytes());
    }
    for &w in program {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Compress a byte stream with zstd level 3 — matches
/// `privacy_prove::compress_proof`.
///
/// Available only under `--features shinobi-compat`, which pulls the
/// `zstd` crate into the vendor set. The sequencer's decoder in
/// `privacy_circuit_verify` calls `zstd::bulk::decompress(compressed,
/// max_bytes)` with `max_bytes = 2 * CAIRO_PROOF_UNCOMPRESSED_BYTES`
/// — our output round-trips through it.
#[cfg(feature = "shinobi-compat")]
pub fn frame_with_zstd(bytes: &[u8]) -> Vec<u8> {
    zstd::encode_all(bytes, ZSTD_LEVEL).expect("zstd encode")
}

#[cfg(not(feature = "shinobi-compat"))]
pub fn frame_with_zstd(_bytes: &[u8]) -> Vec<u8> {
    panic!(
        "frame_with_zstd requires --features shinobi-compat (which enables the zstd dep)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m31(v: u32) -> M31 {
        M31::new(v)
    }

    fn qm31(a: u32, b: u32, c: u32, d: u32) -> QM31 {
        QM31::from_u32_array([a, b, c, d])
    }

    #[test]
    fn m31_is_u32_le() {
        let mut out = Vec::new();
        m31(0x0102_0304).serialize(&mut out);
        assert_eq!(out, vec![0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn qm31_is_four_le_m31s() {
        let mut out = Vec::new();
        qm31(1, 2, 3, 4).serialize(&mut out);
        assert_eq!(out.len(), 16);
        assert_eq!(&out[0..4], &1u32.to_le_bytes());
        assert_eq!(&out[4..8], &2u32.to_le_bytes());
        assert_eq!(&out[8..12], &3u32.to_le_bytes());
        assert_eq!(&out[12..16], &4u32.to_le_bytes());
    }

    #[test]
    fn hash_value_is_two_qm31s() {
        let h = HashValueQm31(qm31(1, 2, 3, 4), qm31(5, 6, 7, 8));
        let mut out = Vec::new();
        h.serialize(&mut out);
        assert_eq!(out.len(), 32);
        for (i, expected) in [1u32, 2, 3, 4, 5, 6, 7, 8].iter().enumerate() {
            assert_eq!(&out[i * 4..(i + 1) * 4], &expected.to_le_bytes());
        }
    }

    #[test]
    fn vec_is_concatenation_no_prefix() {
        let v = vec![m31(1), m31(2), m31(3)];
        let mut out = Vec::new();
        v.serialize(&mut out);
        // 3 M31s = 12 bytes, no 4-byte length prefix.
        assert_eq!(out.len(), 12);
        assert_eq!(&out[0..4], &1u32.to_le_bytes());
    }

    #[test]
    fn interaction_at_oods_omits_none_at_prev() {
        let with = InteractionAtOods {
            at_oods: qm31(1, 2, 3, 4),
            at_prev: Some(qm31(5, 6, 7, 8)),
        };
        let without = InteractionAtOods {
            at_oods: qm31(1, 2, 3, 4),
            at_prev: None,
        };
        let mut a = Vec::new();
        let mut b = Vec::new();
        with.serialize(&mut a);
        without.serialize(&mut b);
        assert_eq!(a.len(), 32);
        assert_eq!(b.len(), 16, "None at_prev contributes zero bytes (no tag)");
    }

    #[test]
    fn claim_log_sizes_are_packed_and_padded() {
        // 3 components → pad to 4 bytes.
        let packed = pack_component_log_sizes(&[20, 18, 15]);
        assert_eq!(packed, vec![20, 18, 15, 0]);

        // 5 components → pad to 8 bytes.
        let packed = pack_component_log_sizes(&[1, 2, 3, 4, 5]);
        assert_eq!(packed, vec![1, 2, 3, 4, 5, 0, 0, 0]);

        // Exactly a multiple of 4 — no extra padding.
        let packed = pack_component_log_sizes(&[10, 11, 12, 13]);
        assert_eq!(packed, vec![10, 11, 12, 13]);
    }

    #[test]
    fn claim_layout_matches_upstream() {
        // packed_log_size_bytes then claimed_sums
        let claimed = vec![qm31(1, 2, 3, 4), qm31(5, 6, 7, 8)];
        let mut out = Vec::new();
        serialize_claim(&pack_component_log_sizes(&[20, 18]), &claimed, &mut out);
        // 4 bytes padded log sizes + 2 × 16 bytes QM31 = 36 bytes
        assert_eq!(out.len(), 4 + 32);
        assert_eq!(&out[0..4], &[20u8, 18, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "log-size pack must be padded")]
    fn claim_rejects_unpadded_log_sizes() {
        let mut out = Vec::new();
        // 3 bytes of log sizes — upstream always pads to multiple of 4.
        serialize_claim(&[20, 18, 15], &[], &mut out);
    }

    #[test]
    fn public_claim_prefix_is_u32_le_concat() {
        let pd = vec![1u32; PUBLIC_DATA_LEN];
        let outs = vec![2u32; NUM_OUTPUTS];
        let prog = vec![3u32; 4];
        let prefix = public_claim_prefix_bytes(&pd, &outs, &prog);
        assert_eq!(prefix.len(), (PUBLIC_DATA_LEN + NUM_OUTPUTS + 4) * 4);
        assert_eq!(&prefix[0..4], &1u32.to_le_bytes());
        // First element of outputs starts at PUBLIC_DATA_LEN * 4.
        let off = PUBLIC_DATA_LEN * 4;
        assert_eq!(&prefix[off..off + 4], &2u32.to_le_bytes());
        // First element of program starts at (PUBLIC_DATA_LEN+1)*4.
        let off2 = (PUBLIC_DATA_LEN + NUM_OUTPUTS) * 4;
        assert_eq!(&prefix[off2..off2 + 4], &3u32.to_le_bytes());
    }

    /// Under `shinobi-compat`, the zstd level-3 encoder matches the
    /// sequencer's `privacy_prove::compress_proof`. Round-trip a mixed
    /// payload through `frame_with_zstd` + `zstd::decode_all` to
    /// verify the framing is non-empty, well-formed, and lossless.
    #[test]
    #[cfg(feature = "shinobi-compat")]
    fn zstd_framing_roundtrips_at_level_3() {
        let mut payload = Vec::new();
        // Mix of high-entropy (random-looking) and low-entropy (zeros)
        // so the compressor has something meaningful to do.
        for i in 0u32..4096 {
            payload.extend_from_slice(&i.wrapping_mul(0x9E37_79B9).to_le_bytes());
        }
        payload.extend(std::iter::repeat(0u8).take(4096));

        let framed = frame_with_zstd(&payload);
        assert!(!framed.is_empty(), "compressed output is non-empty");
        // Zstd frame magic: 0x28B52FFD LE.
        assert_eq!(&framed[..4], &[0x28, 0xB5, 0x2F, 0xFD],
            "output starts with zstd frame magic");

        let decoded = zstd::decode_all(&framed[..]).expect("zstd decode");
        assert_eq!(decoded, payload, "zstd framing must round-trip losslessly");
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "M31 out of range")]
    fn m31_rejects_out_of_range_digest_word() {
        // Simulate a raw Blake2s digest word with bit 31 set. If we
        // allowed it, the upstream deserializer would also reject.
        // The guard is a `debug_assert!` (hot-path-free in release),
        // so this contract is only checkable in debug builds.
        let mut out = Vec::new();
        let bad = M31(0x8000_0001);
        bad.serialize(&mut out);
    }
}
