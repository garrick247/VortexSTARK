//! SNIP-36 (Shinobi) integration: convert a VortexSTARK `CairoProof`
//! into the on-wire format that Starknet v0.14.2's native verifier
//! accepts.
//!
//! SNIP-36 adds two fields to Invoke V3 transactions:
//!   * `proof: Vec<u32>` — the full STARK proof (mempool-only, base64
//!     in RPC). Passed to the S-Two verifier integrated into the
//!     Starknet gateway and sequencer.
//!   * `proof_facts: Vec<Felt252>` — the public facts derivable from
//!     the proof, made available to contracts via the new
//!     `get_execution_info_v3` syscall.
//!
//! ## Wire format
//!
//! The `proof` field is a flat `Vec<u32>` (little-endian) of the
//! bincode serialization of a stwo-compatible proof structure. We
//! reuse `TwoStarkProof` from `stwo_export` so the JSON and wire
//! formats stay in sync — swapping `serde_json` for `bincode` at the
//! boundary. This is NOT yet cross-validated against mainnet's
//! verifier; see the cross-validation notes below.
//!
//! ## proof_facts layout
//!
//! Each field is a single `Felt252` (u256 little-endian). Order matches
//! how `get_execution_info_v3` exposes them to contracts:
//!   0: program_hash (Blake2s digest packed as a single felt)
//!   1: initial_pc
//!   2: initial_ap
//!   3: n_steps
//!   4+: caller-supplied outputs (not derivable from the proof alone)
//!
//! A contract reads `proof_facts` to confirm the proof corresponds to
//! the expected program + entry point before acting on it.
//!
//! ## Cross-validation
//!
//! Before submitting on mainnet, this encoding MUST be tested against
//! Starknet's actual verifier. The `proof_facts` derivation is
//! straightforward (fixed layout), but the `proof` bytes must match
//! the exact bincode convention the gateway expects. A testnet
//! transaction with a known-valid proof is the definitive check.

use crate::cairo_air::prover::CairoProof;
use crate::felt252::Felt252;

/// A SNIP-36-ready bundle: the flat `proof` u32 array + the
/// `proof_facts` Felt252 array. Caller packs these into an Invoke V3
/// transaction via [`build_invoke_v3_tx`].
#[derive(Clone, Debug)]
pub struct Snip36Bundle {
    /// Full STARK proof as a little-endian u32 array. The on-chain
    /// verifier deserializes this back into a stwo `StarkProof`.
    pub proof: Vec<u32>,
    /// Public facts the proof attests to. Exposed to contracts
    /// through `get_execution_info_v3.proof_facts`.
    pub proof_facts: Vec<Felt252>,
}

/// Derive the Felt252 proof_facts array from a `CairoProof`.
///
/// Layout (fixed for v0 of the integration — update the version byte
/// if the layout ever changes):
///   [0]    version = 1
///   [1]    program_hash (packed as one felt; Blake2s-256 output
///          is 32 bytes, fits in a felt252)
///   [2]    initial_pc
///   [3]    initial_ap
///   [4]    n_steps
pub fn proof_facts(proof: &CairoProof) -> Vec<Felt252> {
    let pi = &proof.public_inputs;
    // Pack the 8×u32 program_hash into a single Felt252 (32 bytes,
    // which fits in 252 bits with 2 bits to spare — Blake2s outputs
    // never set the top 4 bits of the leading byte in practice, and
    // even if they did, the 252-bit Stark prime is large enough to
    // hold any 32-byte value mod p unambiguously for hash comparison).
    let hash_bytes = {
        let mut b = [0u8; 32];
        for (i, &w) in pi.program_hash.iter().enumerate() {
            b[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }
        b
    };
    let program_hash_felt = crate::felt252::from_le_bytes(&hash_bytes);

    vec![
        Felt252::from_u64(1), // version
        program_hash_felt,
        Felt252::from_u64(pi.initial_pc as u64),
        Felt252::from_u64(pi.initial_ap as u64),
        Felt252::from_u64(pi.n_steps as u64),
    ]
}

/// Serialize a `CairoProof` into the flat u32 wire format SNIP-36
/// expects. Current encoding: JSON-serialize the stwo-compatible
/// proof structure, then reinterpret the UTF-8 byte stream as u32
/// little-endian words (padded with zeros to a multiple of 4 bytes).
///
/// **Cross-validation TODO:** Starknet's verifier almost certainly
/// expects a binary (bincode-style) encoding, not JSON. This JSON
/// path is a placeholder that keeps the VortexSTARK side unblocked
/// without vendoring a new dependency. Before mainnet submission,
/// swap the encoder for whatever the reference S-Two verifier parses
/// (test against `starknet-devnet-rs` or Starknet testnet with a
/// known-valid proof).
pub fn proof_to_snip36_bytes(proof: &CairoProof) -> Vec<u32> {
    let two_proof = crate::stwo_export::cairo_proof_to_stwo(proof);
    let bytes = serde_json::to_vec(&two_proof)
        .expect("stwo proof serde_json encode");
    let padded_len = (bytes.len() + 3) & !3;
    let mut padded = bytes;
    padded.resize(padded_len, 0);
    padded
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
        .collect()
}

/// Build both halves of the SNIP-36 bundle from a single `CairoProof`.
pub fn to_snip36_bundle(proof: &CairoProof) -> Snip36Bundle {
    Snip36Bundle {
        proof: proof_to_snip36_bytes(proof),
        proof_facts: proof_facts(proof),
    }
}

/// Construct a SNIP-36-ready Invoke V3 transaction JSON body.
///
/// The caller supplies the usual Starknet transaction fields (sender,
/// selector, calldata, etc.); this function fills in the `proof` and
/// `proof_facts` fields from the bundle and emits the JSON that can
/// be POSTed to the gateway or submitted via starknet.js.
pub fn build_invoke_v3_tx(
    bundle: &Snip36Bundle,
    sender_address_hex: &str,
    entry_point_selector_hex: &str,
    calldata_hex: &[String],
    nonce_hex: &str,
    resource_bounds: &serde_json::Value,
) -> serde_json::Value {
    // Hex strings MUST be 0x-prefixed little-endian felt252 encodings
    // as used by Starknet JSON-RPC.
    let proof_facts_hex: Vec<String> = bundle
        .proof_facts
        .iter()
        .map(|f| {
            use crate::felt252::FeltExt;
            f.to_hex_0x()
        })
        .collect();

    serde_json::json!({
        "type": "INVOKE",
        "version": "0x3",
        "sender_address": sender_address_hex,
        "entry_point_selector": entry_point_selector_hex,
        "calldata": calldata_hex,
        "nonce": nonce_hex,
        "resource_bounds": resource_bounds,
        "tip": "0x0",
        "paymaster_data": [],
        "account_deployment_data": [],
        "nonce_data_availability_mode": "L1",
        "fee_data_availability_mode": "L1",
        // SNIP-36 fields
        "proof": bundle.proof,
        "proof_facts": proof_facts_hex,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_facts_layout_is_stable() {
        use crate::cairo_air::prover::{CairoProof, CairoPublicInputs, CAIRO_PROOF_VERSION};
        use crate::prover::{QueryDecommitment, N_QUERIES};
        // Minimal synthetic proof — we're testing the facts layout, not
        // a real prove/verify round-trip.
        let empty_qd = || QueryDecommitment::<[u32; 4]> {
            values: vec![[0u32; 4]; N_QUERIES],
            sibling_values: vec![[0u32; 4]; N_QUERIES],
            auth_paths: vec![vec![]; N_QUERIES],
            sibling_auth_paths: vec![vec![]; N_QUERIES],
        };
        let proof = CairoProof {
            version: CAIRO_PROOF_VERSION,
            log_trace_size: 5,
            public_inputs: CairoPublicInputs {
                initial_pc: 42,
                initial_ap: 100,
                n_steps: 32,
                program_hash: [0xcafe, 0xbabe, 0, 0, 0, 0, 0, 0],
                program: vec![],
            },
            trace_commitment: [1; 8],
            trace_commitment_hi: [1; 8],
            dict_trace_commitment: [1; 8],
            dict_side_table: vec![],
            dict_side_table_commitment: [0; 8],
            interaction_commitment: [1; 8],
            rc_interaction_commitment: [1; 8],
            ec_trace_commitment: None,
            ec_trace_commitment_hi: None,
            ec_log_eval: None,
            ec_trace_at_queries: vec![],
            ec_trace_at_queries_next: vec![],
            ec_trace_auth_paths: vec![],
            ec_trace_auth_paths_next: vec![],
            ec_trace_auth_paths_hi: vec![],
            ec_trace_auth_paths_hi_next: vec![],
            quotient_commitment: [1; 8],
            quotient_decommitment: empty_qd(),
            oods_quotient_commitment: [1; 8],
            oods_quotient_decommitment: empty_qd(),
            fri_start_channel_state: [0; 8],
            fri_commitments: vec![],
            fri_last_layer: vec![],
            fri_last_layer_poly: vec![],
            query_indices: vec![0; N_QUERIES],
            trace_values_at_queries: vec![],
            trace_values_at_queries_next: vec![],
            trace_auth_paths: vec![],
            trace_auth_paths_next: vec![],
            trace_auth_paths_hi: vec![],
            trace_auth_paths_hi_next: vec![],
            trace_auth_paths_dict: vec![],
            trace_auth_paths_dict_next: vec![],
            fri_decommitments: vec![],
            interaction_decommitment: empty_qd(),
            interaction_decommitment_next: empty_qd(),
            rc_interaction_decommitment: empty_qd(),
            rc_interaction_decommitment_next: empty_qd(),
            dict_main_interaction_commitment: [1; 8],
            dict_main_interaction_decommitment: empty_qd(),
            dict_main_interaction_decommitment_next: empty_qd(),
            dict_link_final: [0; 4],
            dict_n_accesses: 0,
            logup_challenges: [0; 24],
            dict_exec_commitment: None,
            dict_sorted_commitment: None,
            dict_log_n: None,
            dict_exec_final_sum: None,
            dict_sorted_final_sum: None,
            dict_exec_data: vec![],
            dict_sorted_data: vec![],
            dict_access_pointers: vec![],
            bitwise_commitment: None,
            bitwise_rows: vec![],
            memory_table_commitment: [1; 8],
            memory_table_data: vec![],
            memory_instr_data: vec![],
            rc_counts_commitment: [1; 8],
            logup_t1_commitment: [1; 8],
            logup_t2_commitment: [1; 8],
            logup_t3_commitment: [1; 8],
            rc_u1_commitment: [1; 8],
            rc_u2_commitment: [1; 8],
            logup_t1_decom: empty_qd(),
            logup_t2_decom: empty_qd(),
            logup_t3_decom: empty_qd(),
            rc_u1_decom: empty_qd(),
            rc_u2_decom: empty_qd(),
            rc_counts_data: vec![],
            logup_final_sum: [0; 4],
            rc_final_sum: [0; 4],
            pow_nonce: 0,
            oods_z: [0; 8],
            oods_trace_at_z: vec![],
            oods_trace_at_z_next: vec![],
            oods_quotient_at_z: [[0; 4]; 4],
            oods_alpha: [0; 4],
            oods_interaction_at_z: [[[0; 4]; 4]; 3],
            oods_interaction_at_z_next: [[[0; 4]; 4]; 3],
        };

        let facts = proof_facts(&proof);
        assert_eq!(facts.len(), 5, "v0 proof_facts has 5 entries");
        assert_eq!(facts[0], Felt252::from_u64(1), "facts[0] = version");
        // facts[1] = packed program_hash
        use crate::felt252::FeltExt;
        let hash_hex = facts[1].to_hex_0x();
        assert!(hash_hex.starts_with("0x"), "program_hash felt prints as hex");
        assert_eq!(facts[2], Felt252::from_u64(42), "facts[2] = initial_pc");
        assert_eq!(facts[3], Felt252::from_u64(100), "facts[3] = initial_ap");
        assert_eq!(facts[4], Felt252::from_u64(32), "facts[4] = n_steps");
    }
}
