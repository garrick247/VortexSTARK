//! Dict consistency argument for Cairo felt252 dictionaries.
//!
//! # What this module provides
//!
//! **CPU-side chain verification** (`verify_chain`): checks that the ordered access log
//! produced by hint execution is internally consistent — for each key, each access's
//! `prev_value` matches the preceding access's `new_value`. Detects hint execution bugs
//! before they propagate into a proof.
//!
//! **LogUp helper functions** (`dict_logup_exec_sum`, `dict_logup_table_sum`): compute
//! the LogUp interaction sums over dict access triples. These are the building blocks for
//! a future STARK-level dict consistency argument.
//!
//! # Current soundness status
//!
//! The CPU chain check runs at prove time and will reject an inconsistent dict access log
//! with `ProveError::DictConsistencyViolation`. This protects against honest prover bugs.
//!
//! A *malicious* prover could still bypass this check (it runs outside the STARK). Full
//! STARK-level dict consistency requires dedicated trace columns (`dict_key`, `dict_prev`,
//! `dict_new`) committed before the Fiat-Shamir challenges, plus a LogUp argument wiring
//! those columns into the transcript. That is planned future work.
//!
//! # Access log format
//!
//! Each entry is `(key, prev_value, new_value)` in execution order as logged by
//! `Felt252DictEntryUpdate`. All values are u64 (M31-truncated felt252).

use std::collections::HashMap;
use crate::field::{M31, QM31};
use super::logup::qm31_from_m31;

/// Error type for dict consistency violations.
#[derive(Debug, Clone, PartialEq)]
pub enum DictConsistencyError {
    /// An access's prev_value does not match the preceding new_value for the same key.
    ChainViolation {
        key: u64,
        expected_prev: u64,
        actual_prev: u64,
    },
}

impl std::fmt::Display for DictConsistencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictConsistencyError::ChainViolation { key, expected_prev, actual_prev } =>
                write!(f, "dict chain violation at key {key:#x}: \
                           expected prev_value={expected_prev:#x}, got {actual_prev:#x}"),
        }
    }
}

/// Verify that the dict access log forms a valid chain.
///
/// For each key, accesses must satisfy: `prev_value[i+1] == new_value[i]`.
/// The first access to any key must have `prev_value == 0` (dict default).
///
/// This is a CPU-side check only — it does not constrain anything in the STARK.
pub fn verify_chain(accesses: &[(usize, u64, u64, u64)]) -> Result<(), DictConsistencyError> {
    // last_new[key] = new_value of the most recent access to that key.
    // Absence = key never accessed before (expected prev = 0).
    let mut last_new: HashMap<u64, u64> = HashMap::new();

    for &(_step, key, prev_value, new_value) in accesses {
        let expected_prev = last_new.get(&key).copied().unwrap_or(0);
        if prev_value != expected_prev {
            return Err(DictConsistencyError::ChainViolation {
                key,
                expected_prev,
                actual_prev: prev_value,
            });
        }
        last_new.insert(key, new_value);
    }
    Ok(())
}

/// Compute the execution-side LogUp sum over the dict access log.
///
/// Each access triple (key, prev, new) contributes `+1 / (z - entry)` where
///   `entry = key + alpha * prev + alpha^2 * new`
///
/// The execution sum and table sum cancel iff the access log is a valid permutation
/// of the table entries. This is the building block for a STARK-level dict argument.
pub fn dict_logup_exec_sum(
    accesses: &[(usize, u64, u64, u64)],
    z: QM31,
    alpha: QM31,
) -> QM31 {
    let alpha_sq = alpha * alpha;
    let mut sum = QM31::ZERO;
    for &(_step, key, prev, new_val) in accesses {
        let entry = qm31_from_m31(M31(key as u32))
            + alpha * qm31_from_m31(M31(prev as u32))
            + alpha_sq * qm31_from_m31(M31(new_val as u32));
        let denom = z - entry;
        debug_assert!(denom != QM31::ZERO, "dict LogUp denominator zero — Fiat-Shamir collision");
        sum = sum + denom.inverse();
    }
    sum
}

/// Compute the table-side LogUp sum (negated, with multiplicities).
///
/// The table is the multiset of unique (key, prev, new) triples from the access log,
/// each weighted by its multiplicity. When added to the execution sum, the total is zero.
pub fn dict_logup_table_sum(
    accesses: &[(usize, u64, u64, u64)],
    z: QM31,
    alpha: QM31,
) -> QM31 {
    use std::collections::BTreeMap;
    let alpha_sq = alpha * alpha;

    // Count multiplicities of each unique (key, prev, new) triple (step is irrelevant for table).
    let mut counts: BTreeMap<(u64, u64, u64), u32> = BTreeMap::new();
    for &(_step, key, prev, new_val) in accesses {
        *counts.entry((key, prev, new_val)).or_insert(0) += 1;
    }

    let mut sum = QM31::ZERO;
    for ((key, prev, new_val), mult) in counts {
        let entry = qm31_from_m31(M31(key as u32))
            + alpha * qm31_from_m31(M31(prev as u32))
            + alpha_sq * qm31_from_m31(M31(new_val as u32));
        let denom = z - entry;
        debug_assert!(denom != QM31::ZERO, "dict table LogUp denominator zero — Fiat-Shamir collision");
        let mult_q = qm31_from_m31(M31(mult));
        sum = sum - mult_q * denom.inverse();
    }
    sum
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 (FELT252_DESIGN.md Option B): Side-table types and Felt252 chain check.
// The main trace retains M31 dict columns (as pointers). The parallel side
// table below holds the full 252-bit key/prev/new values that the pointers
// reference. When the rest of Phase 2 lands, the side table will be committed
// via a Merkle root and linked to the main trace via a LogUp bus; the
// `verify_chain_felt` below is already usable as the CPU-side cross-check.
// ─────────────────────────────────────────────────────────────────────────────

use crate::felt252::Felt252;

/// One entry in the dict side table. `pointer` is the M31-valued position in
/// the main trace that references this row; `key` / `prev_value` /
/// `new_value` are the full Felt252 values.
///
/// Parallel to a `(step, u64, u64, u64)` entry in `HintContext::dict_accesses`
/// — the u64 fields there become small pointer values (< 2^31) once the
/// Phase 2 wiring is done; the true felt values live here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DictSideTableEntry {
    pub pointer: u32,
    pub key: Felt252,
    pub prev_value: Felt252,
    pub new_value: Felt252,
}

/// Felt252-typed error variant, parallel to [`DictConsistencyError::ChainViolation`].
/// Separate variant so the u64 and felt paths can live side by side while
/// the rest of Phase 2 is being wired.
#[derive(Debug, Clone, PartialEq)]
pub enum DictConsistencyErrorFelt {
    ChainViolation {
        key: Felt252,
        expected_prev: Felt252,
        actual_prev: Felt252,
    },
}

impl std::fmt::Display for DictConsistencyErrorFelt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::felt252::FeltExt;
        match self {
            DictConsistencyErrorFelt::ChainViolation { key, expected_prev, actual_prev } =>
                write!(
                    f,
                    "dict chain violation at key {}: expected prev_value={}, got {}",
                    key.to_hex_0x(), expected_prev.to_hex_0x(), actual_prev.to_hex_0x(),
                ),
        }
    }
}

/// Verify the ordered access log forms a valid chain, with full Felt252
/// values. For each key, `prev_value[i+1] == new_value[i]`; the first
/// access to any key must have `prev_value == 0`.
///
/// Equivalent to [`verify_chain`] but typed on `Felt252`. Once Phase 2
/// wiring lands, this replaces the u64 check; for now the two live
/// side-by-side so callers can migrate incrementally.
pub fn verify_chain_felt(
    accesses: &[(usize, Felt252, Felt252, Felt252)],
) -> Result<(), DictConsistencyErrorFelt> {
    // Felt252 (= Fp) doesn't derive Hash, so key the map on its raw limbs.
    let mut last_new: HashMap<[u64; 4], Felt252> = HashMap::new();
    for &(_step, key, prev_value, new_value) in accesses {
        let expected_prev = last_new.get(&key.v).copied().unwrap_or(Felt252::ZERO);
        if prev_value != expected_prev {
            return Err(DictConsistencyErrorFelt::ChainViolation {
                key,
                expected_prev,
                actual_prev: prev_value,
            });
        }
        last_new.insert(key.v, new_value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::cm31::CM31;

    fn q(a: u32, b: u32, c: u32, d: u32) -> QM31 {
        QM31 { a: CM31 { a: M31(a), b: M31(b) }, b: CM31 { a: M31(c), b: M31(d) } }
    }

    #[test]
    fn test_verify_chain_empty() {
        assert!(verify_chain(&[]).is_ok());
    }

    #[test]
    fn test_verify_chain_single_key() {
        // key=1: first access prev=0→new=42, second access prev=42→new=99
        let accesses = [(0usize, 1u64, 0u64, 42u64), (1, 1, 42, 99)];
        assert!(verify_chain(&accesses).is_ok());
    }

    #[test]
    fn test_verify_chain_multi_key() {
        let accesses = [
            (0usize, 1u64, 0u64, 10u64),  // key=1: 0→10
            (1, 2, 0, 20),                 // key=2: 0→20
            (2, 1, 10, 30),                // key=1: 10→30
            (3, 2, 20, 40),                // key=2: 20→40
        ];
        assert!(verify_chain(&accesses).is_ok());
    }

    #[test]
    fn test_verify_chain_violation() {
        // key=1: first access prev=0→new=42, but second access claims prev=99 instead of 42
        let accesses = [(0usize, 1u64, 0u64, 42u64), (1, 1, 99, 100)];
        let err = verify_chain(&accesses).unwrap_err();
        assert!(matches!(err, DictConsistencyError::ChainViolation { key: 1, expected_prev: 42, actual_prev: 99 }));
    }

    #[test]
    fn test_verify_chain_wrong_initial_prev() {
        // First access to key=5 must have prev=0, but claims prev=7
        let accesses = [(0usize, 5u64, 7u64, 10u64)];
        let err = verify_chain(&accesses).unwrap_err();
        assert!(matches!(err, DictConsistencyError::ChainViolation { key: 5, expected_prev: 0, actual_prev: 7 }));
    }

    #[test]
    fn test_logup_cancellation() {
        // exec_sum + table_sum should equal zero for any consistent access log
        let accesses = [(0usize, 1u64, 0u64, 42u64), (1, 1, 42, 99), (2, 2, 0, 7)];
        let z     = q(12345, 67890, 11111, 22222);
        let alpha = q(33333, 44444, 55555, 66666);
        let exec_sum  = dict_logup_exec_sum(&accesses, z, alpha);
        let table_sum = dict_logup_table_sum(&accesses, z, alpha);
        assert_eq!(exec_sum + table_sum, QM31::ZERO,
            "dict LogUp sums should cancel: exec={exec_sum:?} table={table_sum:?}");
    }

    #[test]
    fn test_logup_cancellation_with_repeated_triple() {
        let accesses = [(0usize, 1u64, 0u64, 5u64), (1, 2, 0, 5), (2, 3, 0, 5)];
        let z     = q(98765, 43210, 11111, 99999);
        let alpha = q(22222, 33333, 44444, 55555);
        let exec_sum  = dict_logup_exec_sum(&accesses, z, alpha);
        let table_sum = dict_logup_table_sum(&accesses, z, alpha);
        assert_eq!(exec_sum + table_sum, QM31::ZERO,
            "dict LogUp should cancel for distinct keys with same new_value");
    }

    // ── Phase 2 (Felt252 side-table) chain-check tests ────────────────────

    #[test]
    fn verify_chain_felt_empty() {
        assert!(verify_chain_felt(&[]).is_ok());
    }

    #[test]
    fn verify_chain_felt_single_key() {
        let k = Felt252::from_u64(1);
        let v0 = Felt252::ZERO;
        let v1 = Felt252::from_u64(42);
        let v2 = Felt252::from_u64(99);
        let accesses = [(0usize, k, v0, v1), (1, k, v1, v2)];
        assert!(verify_chain_felt(&accesses).is_ok());
    }

    #[test]
    fn verify_chain_felt_violation_reported_as_felt() {
        let k = Felt252::from_u64(1);
        let v0 = Felt252::ZERO;
        let v1 = Felt252::from_u64(42);
        let wrong = Felt252::from_u64(99);
        // Second access should have prev=42, but claims prev=99.
        let accesses = [(0usize, k, v0, v1), (1, k, wrong, Felt252::from_u64(100))];
        let err = verify_chain_felt(&accesses).unwrap_err();
        match err {
            DictConsistencyErrorFelt::ChainViolation { key, expected_prev, actual_prev } => {
                assert_eq!(key, k);
                assert_eq!(expected_prev, v1);
                assert_eq!(actual_prev, wrong);
            }
        }
    }

    #[test]
    fn verify_chain_felt_handles_wide_felt_keys() {
        // Use keys that would NOT fit in u64 — exactly what Phase 2 is
        // meant to unblock.
        let k_small = Felt252::from_u64(1);
        let k_wide  = Felt252::from_hex("0x0123456789abcdeffedcba9876543210deadbeefcafebabe0011223344556677");
        let v0 = Felt252::ZERO;
        let v_a = Felt252::from_hex("0xabcdef");
        let v_b = Felt252::from_hex("0x7777777777777777777777777777777777777777777777777777777777777777");
        let accesses = [
            (0usize, k_small, v0, v_a),
            (1,       k_wide,  v0, v_b),
            (2,       k_small, v_a, Felt252::from_u64(99)),
            (3,       k_wide,  v_b, Felt252::from_u64(0)),
        ];
        assert!(verify_chain_felt(&accesses).is_ok(),
            "wide-felt keys must validate as independent chains");
    }

    #[test]
    fn dict_side_table_entry_equality() {
        // Smoke: the struct equality works so a Merkle commit can later
        // treat entries as canonical.
        let a = DictSideTableEntry {
            pointer: 17,
            key: Felt252::from_u64(1),
            prev_value: Felt252::ZERO,
            new_value: Felt252::from_u64(42),
        };
        let b = DictSideTableEntry {
            pointer: 17,
            key: Felt252::from_u64(1),
            prev_value: Felt252::ZERO,
            new_value: Felt252::from_u64(42),
        };
        assert_eq!(a, b);
    }
}
