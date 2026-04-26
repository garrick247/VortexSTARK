//! Typed `Felt252` facade for Starknet field elements.
//!
//! VortexSTARK's internal field stack is M31 / CM31 / QM31 (stwo's Circle
//! STARK field). But values **carried through the VM** — storage keys,
//! contract addresses, class hashes, dict keys and dict values — are
//! semantically 252-bit Stark field elements (`felt252`). Losing that
//! distinction led to the u64-truncation boundary at
//! `HintContext::dict_accesses` and at RPC class-hash resolution.
//!
//! This module is **Phase 1 of `FELT252_DESIGN.md`**: name and expose a
//! `Felt252` type plus the conversions needed by later phases (typed
//! `SyscallState`, dict side-table, RPC resolver felt256 widening). No
//! existing callers are rewired yet — that happens in Phase 2.
//!
//! `Felt252` is a thin semantic wrapper around
//! [`crate::cairo_air::stark252_field::Fp`], so anywhere that already
//! consumes `Fp` can accept `Felt252` without conversion (they are the
//! same representation).

use crate::cairo_air::stark252_field::Fp;

/// A Starknet field element (252-bit, modulo the Stark prime).
///
/// Representation is 4 × `u64` little-endian limbs, identical to
/// [`crate::cairo_air::stark252_field::Fp`]. Arithmetic ops (`add`, `sub`,
/// `mul`, `inverse`) are provided by the underlying `Fp`.
///
/// Use [`Felt252::from_u64`] for small literals, [`Felt252::from_hex`]
/// for RPC-format hex strings (`"0x..."`), and [`Felt252::from_le_bytes`]
/// for 32-byte wire payloads.
pub type Felt252 = Fp;

impl FeltExt for Felt252 {}

/// Extension helpers on `Felt252` that don't exist on the underlying `Fp`.
/// Implemented as a trait so the inherent `impl` in `stark252_field` stays
/// owned by that module.
pub trait FeltExt {
    /// True if the felt is zero.
    fn is_zero(&self) -> bool
    where
        Self: AsFelt,
    {
        let v = self.as_felt().v;
        v[0] == 0 && v[1] == 0 && v[2] == 0 && v[3] == 0
    }

    /// Low 64 bits (limb 0). Lossy — use [`FeltExt::try_to_u64`] for a
    /// checked conversion that returns `None` when the upper 188 bits
    /// are non-zero.
    fn low_u64(&self) -> u64
    where
        Self: AsFelt,
    {
        self.as_felt().v[0]
    }

    /// Return `Some(x)` if the felt fits in a `u64`, else `None`.
    /// Useful at boundaries that currently carry only `u64` (syscall
    /// buffers, class-hash resolver), so callers can distinguish "short
    /// testnet hash" from "real 252-bit value that would be truncated".
    fn try_to_u64(&self) -> Option<u64>
    where
        Self: AsFelt,
    {
        let v = self.as_felt().v;
        if v[1] == 0 && v[2] == 0 && v[3] == 0 {
            Some(v[0])
        } else {
            None
        }
    }

    /// Serialize to 32 little-endian bytes (limb 0 is lowest).
    fn to_le_bytes(&self) -> [u8; 32]
    where
        Self: AsFelt,
    {
        let v = self.as_felt().v;
        let mut out = [0u8; 32];
        for (i, &limb) in v.iter().enumerate() {
            out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        out
    }

    /// `0x...`-prefixed big-endian hex, no leading zeros (canonical
    /// Starknet JSON-RPC format).
    fn to_hex_0x(&self) -> String
    where
        Self: AsFelt,
    {
        let v = self.as_felt().v;
        // Build big-endian hex, strip leading zeros, re-prefix with 0x.
        let mut raw = String::with_capacity(66);
        raw.push_str("0x");
        let mut started = false;
        for i in (0..4).rev() {
            let limb = v[i];
            if !started && limb == 0 {
                continue;
            }
            if started {
                raw.push_str(&format!("{limb:016x}"));
            } else {
                raw.push_str(&format!("{limb:x}"));
                started = true;
            }
        }
        if !started {
            raw.push('0');
        }
        raw
    }

    /// Decompose the 252 bits into 9 × M31 limbs (each ≤ 2^28 = one
    /// 28-bit chunk). Layout: `out[0]` is the lowest 28 bits, `out[8]`
    /// the highest 28 (bits 224..252). Used by the eventual dict side
    /// table: each felt row takes 9 M31 columns.
    ///
    /// `9 × 28 = 252`, so no bits are dropped.
    fn to_m31_limbs_9(&self) -> [u32; 9]
    where
        Self: AsFelt,
    {
        let bytes = self.to_le_bytes();
        let mut acc: u64 = 0;
        let mut bits_in_acc: u32 = 0;
        let mut out = [0u32; 9];
        let mut byte_idx = 0;
        for limb in out.iter_mut() {
            while bits_in_acc < 28 && byte_idx < 32 {
                acc |= (bytes[byte_idx] as u64) << bits_in_acc;
                bits_in_acc += 8;
                byte_idx += 1;
            }
            *limb = (acc & 0x0fff_ffff) as u32;
            acc >>= 28;
            bits_in_acc = bits_in_acc.saturating_sub(28);
        }
        out
    }
}

/// Internal glue to let `FeltExt` reach the underlying `Fp` without
/// taking a private field dependency. Same struct, just unwrap.
pub trait AsFelt {
    fn as_felt(&self) -> &Fp;
}

impl AsFelt for Fp {
    fn as_felt(&self) -> &Fp {
        self
    }
}

/// Deserialize 32 little-endian bytes to a `Felt252`. Input > prime is
/// reduced mod p (same contract as [`Fp::new`]).
pub fn from_le_bytes(bytes: &[u8; 32]) -> Felt252 {
    let mut v = [0u64; 4];
    for i in 0..4 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        v[i] = u64::from_le_bytes(buf);
    }
    Felt252::new(v)
}

/// Inverse of [`FeltExt::to_m31_limbs_9`]: reassemble 9 × 28-bit M31
/// limbs back into a `Felt252`. Limbs with the high 4 bits set (i.e.
/// value ≥ 2^28) are rejected via debug_assert — in release, those
/// bits are silently dropped by the `& 0x0fff_ffff` mask.
pub fn from_m31_limbs_9(limbs: &[u32; 9]) -> Felt252 {
    for (i, &l) in limbs.iter().enumerate() {
        debug_assert!(l < (1u32 << 28), "limb[{i}] = 0x{l:x} >= 2^28");
    }
    let mut acc: u128 = 0;
    let mut bits: u32 = 0;
    let mut bytes = [0u8; 32];
    let mut byte_idx = 0;
    for &l in limbs {
        acc |= ((l & 0x0fff_ffff) as u128) << bits;
        bits += 28;
        while bits >= 8 && byte_idx < 32 {
            bytes[byte_idx] = (acc & 0xff) as u8;
            acc >>= 8;
            bits -= 8;
            byte_idx += 1;
        }
    }
    // Flush any remaining bits into the final byte.
    if bits > 0 && byte_idx < 32 {
        bytes[byte_idx] = (acc & 0xff) as u8;
    }
    from_le_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_u64_then_low_u64_roundtrip() {
        let x = Felt252::from_u64(0xdeadbeef_cafebabe);
        assert_eq!(x.low_u64(), 0xdeadbeef_cafebabe);
        assert_eq!(x.try_to_u64(), Some(0xdeadbeef_cafebabe));
    }

    #[test]
    fn from_hex_canonical_parse() {
        let x = Felt252::from_hex("0x1");
        assert_eq!(x.low_u64(), 1);
        let y = Felt252::from_hex("0xdeadbeefcafebabe");
        assert_eq!(y.low_u64(), 0xdeadbeef_cafebabe);
    }

    #[test]
    fn zero_check() {
        assert!(Felt252::ZERO.is_zero());
        assert!(!Felt252::from_u64(1).is_zero());
    }

    #[test]
    fn try_to_u64_rejects_wide_value() {
        // A real 252-bit value doesn't fit in u64.
        let _wide = Felt252::from_hex("0x1234567890abcdef_fedcba0987654321");
        // from_hex handles the underscore strip? No — it just trims whitespace.
        // Use a pure hex string instead:
        let wide = Felt252::from_hex("0x1234567890abcdeffedcba0987654321");
        // Upper limb non-zero → try_to_u64 returns None.
        assert_eq!(wide.try_to_u64(), None);
    }

    #[test]
    fn le_bytes_roundtrip() {
        let x = Felt252::from_u64(0xcafe_babe_dead_beef);
        let bytes = x.to_le_bytes();
        let y = from_le_bytes(&bytes);
        assert_eq!(x, y);
    }

    #[test]
    fn to_m31_limbs_9_preserves_value() {
        // Build a felt from specific bytes and verify the 28-bit chunks
        // decompose correctly.
        let mut bytes = [0u8; 32];
        // Byte 0 low 7 bits = 0x7f; bits 0..8 of limb[0] are these.
        bytes[0] = 0x7f;
        bytes[1] = 0x00;
        bytes[2] = 0x00;
        bytes[3] = 0x10; // bit 28 set → goes into limb[1] low
        let x = from_le_bytes(&bytes);
        let limbs = x.to_m31_limbs_9();
        // Bits 0..28 = 0x0000007f (since bit 28 just starts limb[1]).
        assert_eq!(limbs[0], 0x0000_007f);
        // Bit 28 → limb[1] bit 0.
        assert_eq!(limbs[1], 1);
        // No bits set above that in our input.
        for (i, &l) in limbs.iter().enumerate().skip(2) {
            assert_eq!(l, 0, "limb[{i}] should be 0, got 0x{l:x}");
        }
    }

    #[test]
    fn to_hex_0x_canonical() {
        assert_eq!(Felt252::ZERO.to_hex_0x(), "0x0");
        assert_eq!(Felt252::from_u64(0xbeef).to_hex_0x(), "0xbeef");
        assert_eq!(Felt252::from_u64(1).to_hex_0x(), "0x1");
    }

    #[test]
    fn m31_limbs_9_roundtrip_to_from() {
        // Use the public pair: decompose then reassemble must be identity.
        let cases = [
            Felt252::ZERO,
            Felt252::from_u64(1),
            Felt252::from_u64(0xdeadbeefcafebabe),
            Felt252::from_hex("0x07abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567"),
        ];
        for x in cases {
            let limbs = x.to_m31_limbs_9();
            for (i, &l) in limbs.iter().enumerate() {
                assert!(l < (1u32 << 28), "limb[{i}] = 0x{l:x} exceeds 28-bit cap");
            }
            let y = from_m31_limbs_9(&limbs);
            assert_eq!(x, y, "to_m31_limbs_9 / from_m31_limbs_9 roundtrip broken");
        }
    }
}
