//! SNIP-36 (Shinobi) integration: convert a VortexSTARK `CairoProof`
//! into the on-wire format that Starknet v0.14.2's native verifier
//! accepts.
//!
//! SNIP-36 adds two fields to Invoke V3 transactions:
//!   * `proof: Arc<Vec<u8>>` — raw proof bytes, serialized as base64
//!     in JSON-RPC (confirmed by reading
//!     `starkware-libs/sequencer/crates/starknet_api/src/transaction/fields.rs`
//!     as of 2026-04). Verified by the `privacy_circuit_verify` crate.
//!   * `proof_facts: Vec<Felt>` — the public facts the proof attests
//!     to, exposed to contracts via `get_execution_info_v3`.
//!
//! ## ⚠ Phase 1 only accepts SNOS (Starknet OS) proofs
//!
//! The sequencer's `starknet_proof_verifier/src/proof_verifier.rs`
//! rejects any proof whose `proof_facts[1] != VIRTUAL_SNOS`:
//!
//! ```text
//! VIRTUAL_SNOS                 = ASCII "VIRTUAL_SNOS"
//! Error: "Non-SNOS proofs are not currently supported"
//! ```
//!
//! VortexSTARK currently proves single-Cairo-program executions, NOT
//! Starknet OS (SNOS) runs. This means:
//! - VortexSTARK output wrapped as SNIP-36 will be **rejected** by
//!   mainnet Phase 1. Submitting it would error out, not produce a
//!   silent accept of a bad proof.
//! - To ship on mainnet under Shinobi, VortexSTARK would need to
//!   either prove an SNOS execution (a much larger Cairo program),
//!   or wait for a future SNIP that accepts arbitrary Cairo-AIR
//!   proofs (planned, not shipped).
//!
//! ## proof_facts layout (actual, from sequencer source)
//!
//! The real layout is:
//!   [0]    PROOF_VERSION = `0x50524f4f4630` (ASCII "PROOF0")
//!   [1]    variant_marker = VIRTUAL_SNOS (only SNOS accepted Phase 1)
//!   [2]    virtual_os_output_version = VIRTUAL_SNOS0
//!   [3]    program_hash
//!   [4..]  SNOS output felts
//!
//! Our `proof_facts_snos_placeholder` builds the first three markers
//! correctly so the CLI can emit syntactically-correct Invoke V3 JSON
//! for tooling/testing. The `[4..]` SNOS output must come from an
//! actual SNOS run, not from VortexSTARK's single-program proof.
//!
//! ## Wire format for `proof` bytes (confirmed from source 2026-04)
//!
//! Source: `starkware-libs/proving-utils/crates/privacy_circuit_verify/src/lib.rs`
//! at rev `0305dbe` (the rev pinned by the sequencer's Cargo.toml).
//!
//! **The bytes are zstd-compressed.** After `zstd::bulk::decompress`:
//!   - For `verify_cairo` (the non-recursive path): the uncompressed
//!     stream starts with a `public_claim: Vec<u32>` of fixed length
//!     `(PUBLIC_DATA_LEN + NUM_OUTPUTS + program_len) * 4` bytes,
//!     then a custom-serialized `StarkProof` parsed by
//!     `circuit_serialize::deserialize_proof_with_config`.
//!   - For `verify_recursive_circuit` (the path Invoke V3 uses):
//!     the uncompressed stream is just the serialized proof; the
//!     public claim is derived from `proof_facts` instead.
//!
//! `circuit_serialize` is a custom binary format — NOT bincode, NOT
//! serde_json. Matching it requires vendoring the crate (also in
//! proving-utils) or re-implementing its write side.
//!
//! ## Shinobi's hardcoded verifier parameters (Phase 1)
//!
//! The privacy Cairo verifier pins these:
//!   CAIRO_TRACE_LOG_SIZE      = 20    (1M rows, exact)
//!   CAIRO_LOG_BLOWUP_FACTOR   = 3     (vs VortexSTARK's 2)
//!   CAIRO_PCS_CONFIG.pow_bits = 27    (vs VortexSTARK's 26)
//!   CAIRO_FRI_CONFIG.n_queries = 23   (vs VortexSTARK's 80)
//!   CIRCUIT_TRACE_LOG_SIZE    = 21    (recursion circuit)
//!   CIRCUIT_LOG_BLOWUP_FACTOR = 2
//!
//! VortexSTARK's default build uses different parameters. A
//! Shinobi-compatible build would need a feature flag that swaps
//! `BLOWUP_BITS = 3`, `POW_BITS = 27`, `N_QUERIES = 23`, and locks
//! `log_n = 20` for the production Cairo path. See
//! `SHINOBI_COMPAT_PLAN.md` (future doc) for the full checklist.
//!
//! ## Current encoding status
//!
//! `proof_to_snip36_bytes` still emits a structural placeholder —
//! `serde_json::to_vec` of the stwo-shaped proof export — because the
//! top-level on-the-wire `Proof<QM31>` type has a different *shape*
//! than our `CairoProof` (different commitment trees, OODS layout,
//! FRI config constants). See `SHINOBI_COMPAT_PLAN` note in
//! `src/circuit_serialize.rs` for the full AIR-alignment caveat.
//!
//! The primitive encoders the wire format actually uses (M31, QM31,
//! HashValue, slice/Vec) are now implemented faithfully in
//! `crate::circuit_serialize` and exercised by unit tests — those
//! match the upstream crate
//! (`starkware-libs/stwo-circuits/crates/circuit_serialize`)
//! byte-for-byte. Once the AIR is aligned, replacing
//! `proof_to_snip36_bytes` with a struct-for-struct encode built out
//! of those primitives is mechanical.

use crate::cairo_air::prover::CairoProof;
use crate::felt252::Felt252;

/// A SNIP-36-ready bundle: raw `proof` bytes + the `proof_facts`
/// Felt252 array. Caller packs these into an Invoke V3 transaction
/// via [`build_invoke_v3_tx`], which base64-encodes the bytes for
/// JSON-RPC transport.
#[derive(Clone, Debug)]
pub struct Snip36Bundle {
    /// Full STARK proof as raw bytes. The sequencer's `Proof` type is
    /// `Arc<Vec<u8>>`, serialized as base64 in JSON.
    pub proof: Vec<u8>,
    /// Public facts the proof attests to. Exposed to contracts
    /// through `get_execution_info_v3.proof_facts`.
    pub proof_facts: Vec<Felt252>,
}

/// PROOF_VERSION marker — ASCII "PROOF0" as a felt252.
/// Matches `starkware-libs/sequencer/crates/starknet_api/src/transaction/fields.rs`.
pub const PROOF_VERSION_HEX: &str = "0x50524f4f4630";

/// VIRTUAL_SNOS variant marker — ASCII "VIRTUAL_SNOS" as a felt252.
/// The ONLY variant Phase 1 accepts.
pub const VIRTUAL_SNOS_HEX: &str = "0x5649525455414c5f534e4f53";

/// VIRTUAL_OS_OUTPUT_VERSION — ASCII "VIRTUAL_SNOS0" as a felt252.
pub const VIRTUAL_OS_OUTPUT_VERSION_HEX: &str = "0x5649525455414c5f534e4f5330";

/// Build proof_facts with the correct Phase 1 header markers.
///
/// WARNING: `proof_facts[4..]` must be the actual SNOS output for the
/// proof to verify. A VortexSTARK single-Cairo-program proof does
/// **not** produce SNOS output; submitting this to mainnet will be
/// rejected by the privacy-circuit verifier. This helper is provided
/// for tooling, testing, and future SNOS-prover work.
///
/// Layout (matches sequencer source, 2026-04):
///   [0] PROOF_VERSION
///   [1] VIRTUAL_SNOS
///   [2] VIRTUAL_OS_OUTPUT_VERSION
///   [3] program_hash (from the proof's public_inputs)
///   [4..] caller-supplied SNOS output felts (passed in as `snos_output`)
pub fn proof_facts(proof: &CairoProof, snos_output: &[Felt252]) -> Vec<Felt252> {
    let pi = &proof.public_inputs;
    // Pack the 8×u32 program_hash into a single Felt252 (32 bytes,
    // fits in 252 bits modulo the Stark prime).
    let hash_bytes = {
        let mut b = [0u8; 32];
        for (i, &w) in pi.program_hash.iter().enumerate() {
            b[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }
        b
    };
    let program_hash_felt = crate::felt252::from_le_bytes(&hash_bytes);

    let mut facts = Vec::with_capacity(4 + snos_output.len());
    facts.push(Felt252::from_hex(PROOF_VERSION_HEX));
    facts.push(Felt252::from_hex(VIRTUAL_SNOS_HEX));
    facts.push(Felt252::from_hex(VIRTUAL_OS_OUTPUT_VERSION_HEX));
    facts.push(program_hash_felt);
    facts.extend_from_slice(snos_output);
    facts
}

/// Serialize a `CairoProof` into raw bytes. The on-wire `Proof` field
/// is `Arc<Vec<u8>>` base64-encoded in RPC; this returns the inner
/// bytes, which [`build_invoke_v3_tx`] then base64-encodes.
///
/// **Cross-validation TODO:** the exact binary layout the
/// `privacy_circuit_verify` crate expects has not been cross-checked.
/// Current placeholder: serde_json serialization of the stwo proof
/// structure. Before mainnet submission, swap for whatever
/// `privacy_circuit_verify` actually parses and test against
/// starknet-devnet-rs with a known-valid (SNOS) proof.
pub fn proof_to_snip36_bytes(proof: &CairoProof) -> Vec<u8> {
    let two_proof = crate::stwo_export::cairo_proof_to_stwo(proof);
    serde_json::to_vec(&two_proof).expect("stwo proof serde_json encode")
}

/// Base64 encoding using the standard alphabet (RFC 4648) — matches
/// the sequencer's `base64::encode` call on the `Proof` type.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let b = [bytes[i], bytes[i + 1], bytes[i + 2]];
        out.push(ALPHABET[(b[0] >> 2) as usize] as char);
        out.push(ALPHABET[(((b[0] & 0x03) << 4) | (b[1] >> 4)) as usize] as char);
        out.push(ALPHABET[(((b[1] & 0x0f) << 2) | (b[2] >> 6)) as usize] as char);
        out.push(ALPHABET[(b[2] & 0x3f) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let b0 = bytes[i];
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[((b0 & 0x03) << 4) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let b0 = bytes[i];
        let b1 = bytes[i + 1];
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(ALPHABET[((b1 & 0x0f) << 2) as usize] as char);
        out.push('=');
    }
    out
}

/// Build both halves of the SNIP-36 bundle from a single `CairoProof`.
/// `snos_output` is the virtual-OS task output that follows the
/// PROOF_VERSION/VIRTUAL_SNOS/VIRTUAL_OS_OUTPUT_VERSION/program_hash
/// header. An empty slice produces a syntactically-valid Invoke V3
/// payload, but will fail verification because the SNOS task content
/// is empty.
pub fn to_snip36_bundle(proof: &CairoProof, snos_output: &[Felt252]) -> Snip36Bundle {
    Snip36Bundle {
        proof: proof_to_snip36_bytes(proof),
        proof_facts: proof_facts(proof, snos_output),
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

    // Proof is serialized as base64 per the sequencer's
    // `impl Serialize for Proof { serializer.serialize_str(base64::encode(...)) }`.
    let proof_b64 = base64_encode(&bundle.proof);

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
        "proof": proof_b64,
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

        // Pass empty SNOS output — just verifies the header layout.
        let facts = proof_facts(&proof, &[]);
        assert_eq!(facts.len(), 4, "header has 4 entries before SNOS output");
        assert_eq!(facts[0], Felt252::from_hex(PROOF_VERSION_HEX),
            "facts[0] = PROOF_VERSION ('PROOF0')");
        assert_eq!(facts[1], Felt252::from_hex(VIRTUAL_SNOS_HEX),
            "facts[1] = VIRTUAL_SNOS (Phase 1 only accepts this variant)");
        assert_eq!(facts[2], Felt252::from_hex(VIRTUAL_OS_OUTPUT_VERSION_HEX),
            "facts[2] = VIRTUAL_OS_OUTPUT_VERSION ('VIRTUAL_SNOS0')");
        // facts[3] = program_hash packed as felt
        use crate::felt252::FeltExt;
        let hash_hex = facts[3].to_hex_0x();
        assert!(hash_hex.starts_with("0x"), "program_hash felt prints as hex");

        // With a two-felt SNOS output, layout grows to 6.
        let snos_output = vec![Felt252::from_u64(999), Felt252::from_u64(1000)];
        let facts_with = proof_facts(&proof, &snos_output);
        assert_eq!(facts_with.len(), 6);
        assert_eq!(facts_with[4], Felt252::from_u64(999));
        assert_eq!(facts_with[5], Felt252::from_u64(1000));
    }

    #[test]
    fn base64_encode_roundtrip() {
        // Smoke test against known vectors.
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }
}
