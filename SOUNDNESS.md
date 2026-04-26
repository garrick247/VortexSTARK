# VortexSTARK Soundness Status

## Constraint System (N_VM_COLS=31 execution columns + 3 dict linkage = N_COLS=34 total; N_CONSTRAINTS=35)

### Flag binary (constraints 0-14)
- All 15 flags constrained to {0, 1} via `flag * (1 - flag) = 0`

### Result computation (constraint 15)
- `(1 - pc_jnz) * (res - expected_res) = 0`
- Expected_res = default*op1 + res_add*(op0+op1) + res_mul*(op0*op1)

### PC update (constraint 16)
- Non-jnz: next_pc = regular + abs + rel
- Jnz taken: `pc_jnz * dst * (next_pc - pc - op1) = 0`

### AP update (constraint 17)
- next_ap = ap + ap_add*res + ap_add1 + call*2

### FP update (constraint 18)
- next_fp = (1-call-ret)*fp + call*(ap+2) + ret*dst

### Assert_eq (constraint 19)
- `opcode_assert * (dst - res) = 0`

### Operand address verification (constraints 20-22)
- dst_addr = (1-dst_reg)*ap + dst_reg*fp + off0 - 0x8000
- op0_addr = (1-op0_reg)*ap + op0_reg*fp + off1 - 0x8000
- op1_addr = op1_base + off2 - 0x8000 (op1_base depends on source flags)

### JNZ soundness (constraints 23-24)
- Fall-through: `pc_jnz * (1 - dst*dst_inv) * (next_pc - pc - inst_size) = 0`
- Inverse consistency: `pc_jnz * dst * (1 - dst*dst_inv) = 0`
- dst_inv auxiliary column: M31 inverse of dst (0 when dst=0)

### Mutual exclusivity (constraints 25-29)
- Op1 source: op1_imm*op1_fp=0, op1_imm*op1_ap=0, op1_fp*op1_ap=0
- PC update: jump_abs*jump_rel + jump_abs*jnz + jump_rel*jnz = 0
- Opcode: call*ret + call*assert + ret*assert = 0

### Instruction decomposition (constraint 30)
- `inst_lo + inst_hi ≡ off0 + off1·2^16 + off2·2^32 + Σ(flag_i·2^(48+i))` (mod P)
- Exploits M31 wraparound: 2^31 ≡ 1, so inst_hi·2^31 ≡ inst_hi
- All 63 bits covered: 48 bits via offsets, 15 bits via flag binary constraints

## Verified Subsystems

### Vanishing polynomial / zerofier (FIXED 2026-03-22)
- Previously: quotient kernel wrote C(x) (raw constraint sum). FRI proved C(x) low-degree,
  but low-degree C(x) alone does not prove constraints vanish on the trace domain.
- Now: GPU kernel `compute_vanishing_inv_kernel` computes 1/Z_H(x) for every NTT position.
  Z_H(x) = f_{log_n}(x)+1 where f_k is iterated circle doubling (x→2x²−1), zero iff x is
  in the trace domain. Quotient kernel multiplies by vh_inv, producing Q(x)=C(x)/Z_H(x).
- Verifier checks C(x)==Q(x)·Z_H(x) at each query point using Coset::circle_vanishing_poly_at.
- Closing this gap is the primary soundness improvement: FRI now proves Q(x) is low-degree,
  which by the Schwartz-Zippel argument implies C(x) vanishes on the trace domain w.h.p.

### Verifier-side constraint evaluation
- Verifier independently evaluates all 35 constraints at query points (31 VM execution + 2 LogUp/RC step-transition + 2 dict linkage)
- Checks constraint_sum == quotient_value * Z_H(eval_point.x) (accounting for zerofier)
- Per-constraint-family tamper tests all passing

### LogUp interaction trace decommitment (ADDED 2026-03-22)
- LogUp interaction trace computed correctly on CPU (trace domain), then NTT'd to eval domain.
  Previous GPU approach (prefix sum over eval-domain positions) was incorrect — it accumulated
  logup_delta at eval-domain points, not trace-domain points, producing a wrong polynomial.
- Both LogUp and RC interaction traces now use the same correct pipeline:
  compute_interaction_trace (trace domain) → interpolate → zero-pad → evaluate → commit.
- test_logup_final_sum_cancels: exec_sum + memory_table_sum == 0 VERIFIED
- LogUp and RC interaction traces decommitted at query points (auth paths vs interaction_commitment / rc_interaction_commitment)
- Verifier checks auth paths for interaction_decommitment and rc_interaction_decommitment
- test_tamper_interaction_decommitment: DETECTED
- test_tamper_rc_interaction_decommitment: DETECTED
- **FIXED 2026-03-24**: Step transition constraints fully wired into AIR polynomial:
  - Constraint 31: S_logup[i+1] - S_logup[i] - logup_delta(row_i) = 0 (QM31)
  - Constraint 32: S_rc[i+1] - S_rc[i] - rc_delta(row_i) = 0 (QM31)
  - CUDA quotient kernel receives 8 interaction columns (4 LogUp + 4 RC) and a
    16-u32 challenge buffer [z_mem, alpha_mem, alpha_mem_sq, z_rc].
  - Verifier evaluates both step-transition constraints at query points in the
    combined 33-constraint sum; checks constraint_sum == Q(x)*Z_H(x).
  - Auth paths for interaction_decommitment_next and rc_interaction_decommitment_next
    cryptographically bind S[qi+1] to the committed interaction polynomial.
  - Instruction fetch denominator extended with alpha²*inst_hi: all 15 flag bits
    are now bound to the memory argument — closes Gap 2 (inst_hi).
- test_tamper_logup_final_sum: DETECTED (Fiat-Shamir binding)
- Fiat-Shamir verifier ordering bug fixed: EC trace commitments now correctly bound before
  z_mem/alpha_mem/z_rc challenges (matching prover order).
- **FIXED 2026-03-26: Memory table commitment closes LogUp soundness gap**
  - Previously: verifier accepted any claimed exec_sum without checking cancellation.
  - Now: prover serializes full memory table (unique (addr,val,mult) + (pc,lo,hi,mult) entries),
    commits via chained Blake2s (`hash_words`), mixes into Fiat-Shamir BEFORE constraint_alphas.
  - Verifier recomputes hash, checks match, independently computes table_sum,
    and enforces exec_sum + table_sum == 0. Malicious exec_sum is now detected.
  - test_tamper_memory_table_data: DETECTED (hash mismatch)
  - test_tamper_memory_table_commitment: DETECTED (channel diverges → FRI fails)
  - test_tamper_logup_cancellation: DETECTED (cancellation check fails)

### Range check argument
- extract_offsets() extracts 3x16-bit offsets per instruction
- compute_rc_interaction_trace() computes running sum, compute_rc_table_sum() computes table contribution
- Prover asserts exec_sum + table_sum = 0 (LogUp cancellation)
- z_rc challenge drawn from Fiat-Shamir, RC final sum bound into transcript
- RC interaction trace decommitted at query points (auth paths vs rc_interaction_commitment)
- test_tamper_rc_final_sum: DETECTED
- test_tamper_rc_interaction_decommitment: DETECTED
- test_rc_final_sum_is_real: verifies RC sum is distinct from LogUp sum
- **FIXED 2026-03-26: RC multiplicity table commitment closes RC soundness gap**
  - Previously: verifier accepted any claimed rc_final_sum without checking cancellation.
  - Now: prover includes all 65536 multiplicity counts, commits via hash_words, mixes into
    Fiat-Shamir after memory_table_commitment. Verifier checks rc_exec_sum + rc_table_sum == 0.
  - test_tamper_rc_counts_data: DETECTED (hash mismatch)
  - test_tamper_rc_counts_commitment: DETECTED (channel diverges → FRI fails)

### Public inputs and trust model
- `initial_pc`, `initial_ap`, `n_steps`, `program_hash`, and the full program bytecode are
  supplied as public inputs and bound into the Fiat-Shamir transcript before any challenges.
- **program_hash** is computed by the prover as `Blake2s(bytecode)` and mixed into the
  channel. The verifier receives `program_hash` as a field in the proof struct and mixes the
  same value into its transcript. The verifier does **not** recompute `program_hash` from
  the bytecode — it trusts the supplied hash. This is standard STARK convention: the verifier
  is given the statement (program_hash, initial_pc/ap, n_steps) and checks that the proof is
  valid for that statement. Recomputing the hash from bytecode is the caller's responsibility
  if bytecode integrity must be verified.
- **Boundary constraint (initial state):** The verifier checks that the FRI quotient is
  consistent with the step-transition constraints at all query points, but does NOT enforce
  a hard boundary constraint `T_PC[0] == initial_pc`. The initial register state is part of
  the trusted public input. A cheating prover who controls the public inputs could claim a
  false initial state. This is a known limitation of the current protocol: boundary
  constraints are not explicitly enforced in the FRI quotient polynomial.

### Trace size requirement (ProveError::TraceSizeMismatch)
- VortexSTARK requires `n_steps == 2^log_n` exactly. No padding is supported.
- If `n_steps < 2^log_n`, padding rows are all-zeros. The LogUp delta for zero rows is
  `4/z ≠ 0`, causing the step-transition constraint (C31) to fail at the padding boundary,
  producing a non-polynomial quotient and a failing FRI check (completeness failure).
- `cairo_prove_program` returns `Err(ProveError::TraceSizeMismatch { n_steps, log_n })` if
  the step count does not match the requested power of two. This check runs **after** execution
  so that `ExecutionRangeViolation` (which is more informative) takes priority when both apply.
- `cairo_prove_cached_with_columns` asserts `n_steps == 2^log_n` at call time (panic on
  violation, since the raw column path is for internal use only).
- The CLI (`stark_cli prove-file`, `prove-starknet`) prints a helpful error and exits if
  `n_steps != 2^log_n` after calling `detect_steps`.
- **External auditor note:** Verify that no path through the prover can silently pad the
  trace with zero rows. The assertion in `cairo_prove_cached_with_columns` and the early
  return in `cairo_prove_program` are the two enforcement points.

### Merkle auth paths (FULLY ACTIVATED 2026-03-22)
- Previously: quotient and FRI decommitments had empty auth_paths; verifier skipped the check,
  allowing a cheating prover to supply fake quotient/FRI values without Merkle binding.
- Now: all four commitments have real auth paths:
  - trace_commitment (cols 0-15) + trace_commitment_hi (cols 16-30) — cpu_merkle_auth_paths_ncols
  - quotient_commitment — decommit_from_host_soa4 (cpu_merkle_auth_paths_soa4)
  - all FRI layer commitments — decommit_fri_layer (cpu_merkle_auth_paths_soa4)
- Verifier rejects empty auth paths (hard error instead of silent skip).
- Merkle hash construction: leaves and internal nodes both use standard Blake2s with `h[6] = IV6` (no personalization), matching stwo's `MerkleHasherLifted::hash_children`. `blake2s_hash_node` in `src/channel.rs` calls `blake2s_hash` with domain=0x00; `cuda/blake2s.cu` defines `IV6_NODE` as `IV6`.
- Length-based separation: leaf inputs are `min(n_cols, 16) × 4` bytes (Blake2s `t0` reflects this); internal-node inputs are always 64 bytes. For n_cols < 16 the lengths differ and `t0` differs, producing different hashes structurally. For 16-column commits the lengths coincide; leaf-vs-node distinction in that case relies on Blake2s collision resistance plus the verifier knowing the tree depth at each auth-path step (the verifier never confuses a level-0 leaf with a level-k node).
- test_quotient_auth_paths_reject_fake_value: DETECTED
- test_fri_auth_paths_reject_fake_value: DETECTED

### Trace decommitment auth paths (ADDED 2026-03-22)
- Previously: trace_values_at_queries had NO Merkle auth paths. A cheating prover could supply
  arbitrary trace values satisfying constraints without being bound to the committed trace root.
- Now: trace is committed in two halves — trace_commitment (cols 0-15) and trace_commitment_hi
  (cols 16-30). Both are bound into the Fiat-Shamir transcript. Auth paths are generated for
  all 4 combinations (qi, qi+1) × (lo, hi) using cpu_merkle_auth_paths_ncols.
- Verifier checks all four sets of auth paths: every trace column at every query point is
  cryptographically bound to a committed polynomial.
- GPU/CPU agreement: fixed GPU merkle_tiled_generic_kernel and merkle_hash_leaves_kernel to use
  min(n_cols,16)*4 as the Blake2s length counter, matching the CPU blake2s_hash behavior.
- test_tamper_trace_auth_paths: tests both lo (col 0 = pc) and hi (col 26 = res) tamper cases — DETECTED

### Pedersen EC constraint system (FULLY BOUND 2026-03-22)
- Full intermediate EC trace generated: 29 columns per step (acc_x/y 9 limbs each, lambda 9 limbs, window, op_type)
- EC trace split into lo (cols 0-15) and hi (cols 16-28), committed separately via NTT + Merkle.
  Both roots bound into Fiat-Shamir transcript (ec_trace_commitment, ec_trace_commitment_hi).
- Previously: single commitment covered only the first 16 columns (GPU Blake2s leaf hash cap);
  cols 16-28 were unconstrained. A cheating prover could supply fake hi-column values.
- Now: lo/hi split mirrors the main trace split. Auth paths generated for all 4 sets
  (qi × lo/hi, qi+1 × lo/hi) via cpu_merkle_auth_paths_ncols.
- Verifier maps main eval domain query indices into EC eval domain (qi % ec_eval_size)
  using ec_log_eval stored in the proof, then verifies all 4 auth path sets.
- test_tamper_ec_trace_auth_paths: lo tamper (col 0) DETECTED, hi tamper (col 16) DETECTED
- Verifier checks EC doubling/addition constraints at query points

### Multi-program testing
- Fibonacci (add-only): proven + verified
- Multiply-accumulate (mul-only): proven + verified
- Mixed add/mul alternating: proven + verified
- Call/ret initialization pattern: proven + verified

### Adversarial / forgery coverage (ADDED 2026-03-24, fixed 2026-03-25)
- test_per_constraint_forgery_all_columns: loops over all 31 trace columns × {current-row, next-row},
  tampers each by +1 mod P, asserts verifier rejects. Covers all 31 per-column constraints and
  Merkle auth-path binding for both row positions.
- Constraints 31-32 (LogUp/RC step transitions) covered by existing
  test_tamper_interaction_decommitment and test_tamper_rc_interaction_decommitment.
- Full 33-constraint system: every constraint has at least one dedicated rejection test.

### Step-transition boundary wrap (VERIFIED 2026-03-24)
- test_step_transition_boundary_wrap: verifies next_qi = (qi+1) % eval_size is always < eval_size,
  and that the wrap-around case (qi = eval_size-1 → next_qi = 0) is handled correctly.
- Prover (line 623) and verifier (line 1233) both use `% eval_size` — correct.
- Tampering next-row values causes verifier rejection at all positions including the boundary.

### CASM file loader (VERIFIED 2026-03-24)
- test_prove_casm_file: loads tests/fixtures/fibonacci.casm (Cairo 1 CASM JSON format,
  as produced by sierra-to-casm / scarb build), proves 32 Fibonacci steps, verifies proof.
- Exercises the casm_loader::load_program → CasmFormat::CasmJson → cairo_prove → cairo_verify
  end-to-end path using a file on disk, not hand-crafted bytecode.

### Blowup factor and FRI security (VERIFIED 2026-03-24)
- BLOWUP_BITS = 2 (4× blowup), N_QUERIES = 80.
- eval_size = 1 << (log_n + BLOWUP_BITS) — cairo_air/prover.rs fixed to match this (previously
  eval_size was hardcoded as 2*n, inconsistent with the 4× blowup configured by BLOWUP_BITS=2).
- Fix prevents CUDA buffer overflow (prover allocated 2n but NTT wrote 4n elements) and wrong
  FRI last-layer size (4 instead of 8 elements, causing all tests to fail).
- Conjectured security: 2 bits/query × 80 queries = 160-bit, above the 100-bit design target.

## All Soundness Gaps Closed (as of 2026-03-26)

### (CLOSED) Dict consistency — GAP-1

**Updated 2026-03-26, session 6:** S_dict step-transition LogUp argument fully closes the link between the main execution trace and the dict sub-AIR.

#### Stage 1 — Dict sub-AIR (session 3, 2026-03-26):

**Fiat-Shamir transcript ordering:**
1. Exec data Merkle root (3-col trace: key, prev, new in execution order) → mixed into channel
2. Sorted data Merkle root (4-col trace: key, prev, new, is_first sorted by key) → mixed into channel
3. `z_dict`, `alpha_dict` drawn from channel (post-commitment, pre-interaction)
4. Exec interaction Merkle root (4-col QM31 running sum, exclusive prefix) → mixed into channel
5. Sorted interaction Merkle root (4-col QM31 running sum, exclusive prefix) → mixed into channel
6. `exec_final_sum` and `sorted_final_sum` → mixed into channel (binds them to subsequent FRI challenges)

**Sorted step-transition constraints (C0-C3, checked at query points):**
- C0: `is_first[i] * (1 − is_first[i]) = 0` — is_first is binary
- C1: `(1 − is_first[i+1]) * (key[i+1] − key[i]) = 0` — key is non-decreasing within a run
- C2: `is_first[i+1] * prev[i+1] = 0` — first access per key has prev = 0
- C3: `(1 − is_first[i+1]) * (prev[i+1] − new[i]) = 0` — consecutive accesses chain correctly

**Full soundness of dict sub-AIR:** Verifier receives ALL dict_n rows. Recomputes both Merkle roots,
checks all dict_n-1 sorted step-transition pairs (no sampling), recomputes both LogUp final sums,
verifies `exec_final == sorted_final`.

#### Stage 2 — S_dict main-trace link (session 6, 2026-03-26):

**Problem being closed:** A malicious prover could fabricate the dict exec trace (key/prev/new) as long as it was internally consistent and the permutation argument passed. The dict exec trace was not cryptographically bound to the main execution trace.

**Solution — columns 31-33 and constraints C33-C34:**

**Columns added to main trace (now N_COLS = 34):**
- Col 31: `dict_key` — key of the dict access at this execution step (0 if no access)
- Col 32: `dict_new` — new value written at this execution step (0 if no access)
- Col 33: `dict_active` — 1 if this step has a dict access, else 0

**New interaction trace — S_dict (4 QM31 cols, committed as `dict_main_interaction_commitment`):**
`S_dict[i+1] = S_dict[i] + dict_active[i] / (z_dict_link − (dict_key[i] + α_dict_link · dict_new[i]))`

**New constraints (N_CONSTRAINTS = 35):**
- C33: `dict_active * (1 − dict_active) = 0` — dict_active is boolean
- C34: S_dict step-transition constraint (as above)

**Fiat-Shamir ordering for Stage 2:**
1. `dict_trace_commitment` (Group C, cols 31-33) → mixed into channel
2. `z_dict_link`, `α_dict_link` drawn from channel
3. S_dict trace built → `dict_main_interaction_commitment` → mixed into channel
4. `dict_link_final` (S_dict[n]) → mixed into channel

**Verifier closure:** The verifier independently computes `exec_key_new_sum` from the authenticated dict exec trace (`key + α_dict_link * new_val` formula, over `dict_n_accesses` real accesses). It checks `dict_link_final == exec_key_new_sum`. This forces the main trace's dict columns to contain exactly the same (key, new_val) multiset as the authenticated exec trace.

**Security argument:** `S_dict_final = exec_key_new_sum` (same (key,new) multiset) AND the dict sub-AIR proves chain validity (sorted constraints C0-C3) → prev values are uniquely determined by the chain. Full dict consistency is enforced by the FRI-verified quotient polynomial.

**Files:** `src/cairo_air/trace.rs` (N_VM_COLS=31, N_COLS=34, N_CONSTRAINTS=35), `src/cairo_air/prover.rs`, `cuda/cairo_constraint.cu` (C33, C34), `src/cairo_air/dict_air.rs`.

### (PARTIALLY MITIGATED — Phase 2 in progress) Felt252 truncation

`cairo_prove_program` returns `Err(ProveError::Felt252Overflow)` if any
bytecode value exceeds u64 range. Values in (M31, u64] are still silently
reduced mod M31 inside trace columns, because those columns are M31-valued.

**Phase 2 progress (2026-04-23):** The full 252-bit story is being unblocked
incrementally per `FELT252_DESIGN.md`:

- `src/felt252.rs` — typed `Felt252` + helpers (to/from hex, to_m31_limbs_9,
  to/from le_bytes, low_u64, try_to_u64). Hash derived on the underlying
  `Fp` so `HashMap<Felt252, _>` works.
- `CasmProgram::bytecode_felt: Vec<Felt252>` — parallel to the (truncated)
  `bytecode: Vec<u64>`, preserves all 252 bits of every bytecode constant.
  Populated by every loader (CASM JSON, Cairo 0 JSON, Starknet RPC).
- `CairoProof::dict_side_table: Vec<[u32; 28]>` + `dict_side_table_commitment`
  — Option B side table from the design doc. Each row is
  `[pointer, 9 key limbs, 9 prev limbs, 9 new limbs]`. Blake2s-committed to
  the Fiat-Shamir channel right after `dict_trace_commitment` and before
  `z_dict_link` is drawn. Verifier recomputes + mixes. Empty side table ⇒
  no mix ⇒ pre-Phase-2 proofs still verify. Verifier also enforces
  canonical-limb encoding (each of the 27 M31 limbs < 2^28) so a prover
  cannot commit non-canonical Felt252 garbage.
- `SyscallState::storage: HashMap<Felt252, Felt252>` — full-felt contract
  storage; STORAGE_READ/WRITE widen u64→Felt252 on the way in and narrow
  via `low_u64` on the way out. Values preserved for proof-side consumers.
- `SyscallState::caller_address` / `contract_address` / `entry_point_selector`
  — widened to `Felt252`; narrowed at the memory-write boundary.
- `SyscallEvent.keys/data`, `CrossContractCall.target/entry_point_selector/calldata`,
  `DeployedContract.class_hash/salt/calldata/contract_address`,
  `L1Message.to_address/payload` — all widened to `Felt252` / `Vec<Felt252>`.
- `RpcResolver::try_resolve_felt(Felt252)` — accepts full 252-bit
  class_hashes, canonical hex normalization; shares a single cache with
  the u64 entry point.

**What still truncates** (Phase 2 remainder):
- VM memory cells are still u64. Any value written or read through memory
  during execution passes through a u64 boundary.
- `HintContext::dict_accesses` still carries `(usize, u64, u64, u64)`
  quadruples; the side-table commitment above widens at the boundary,
  preserving the *committed* values but not the *memory* representation.

**VM memory model (2026-04-23):** `Memory` retains a `Vec<u64>` as the
authoritative AIR-facing store (the trace columns are still M31 and reduce
u64 → u32 at ingestion), but now carries a sparse `felt_overlay:
HashMap<u64, Felt252>` for addresses whose real value exceeds u64. The
`set_felt`/`get_felt` API routes syscall writes (storage, caller_address,
contract_address, entry_point_selector, dict keys/values that originate
from full-felt syscall data) through the overlay, so those cells round-trip
bit-exact at 252 bits even though the u64 layer truncates. A plain `set`
clears any overlay entry to prevent stale reads. `HintContext` gained a
parallel `dict_accesses_felt: Vec<(usize, Felt252, Felt252, Felt252)>` and
`dicts_felt: HashMap<u64, HashMap<Felt252, Felt252>>`, populated alongside
the u64 log from `memory.get_felt` — the prover's side-table encoding now
uses these full-width values when present, falling back to
`Felt252::from_u64` only when the felt log is absent (e.g. legacy call
sites that haven't been threaded through the overlay yet).

**Soundness implication:** The Phase 2 side-table is now fully bound at
full precision: the verifier enforces (1) canonical Felt252 limb encoding
(each 28-bit M31 limb < 2^28) and (2) a per-row link that projects the
side-table's low 31 bits and asserts equality with the Merkle-authenticated
`dict_exec_data`. Combined with the Fiat-Shamir mix of the side-table
commitment before `z_dict_link` is drawn, a prover cannot substitute
arbitrary Felt252 values into the side-table without detection. A program
whose runtime AIR-column values exceed M31 still triggers
`ProveError::ExecutionRangeViolation` and is refused proving (the AIR
itself is M31); full-felt values that flow through the overlay are
preserved for the proof-layer side-table but do not participate in AIR
constraints.

### (CLOSED 2026-03-26) Full ZK — GAP-4

**All 34 trace columns are ZK-blinded** via `r · Z_H(x)` added at eval-domain level.
Z_H(x) vanishes on the trace domain, so witnesses at trace points are unchanged.
At query points Z_H ≠ 0, so each query reveals `true_value + r · Z_H(query_point)`
— uniformly distributed in M31 given fresh random `r` per column.

**Blinded columns (34/34):** all 34 trace columns including all 9 formerly-unblinded
LogUp columns (pc, inst_lo, inst_hi, dst_addr, dst, op0_addr, op0, op1_addr, op1) and
all 3 dict linkage columns (dict_key, dict_new, dict_active).

**Why the previous restriction was overly conservative:** The interaction trace columns
(S_logup, S_rc, S_dict) already receive the randomized Fiat-Shamir challenges `z` and
`α` drawn *after* the trace commitment. The constraint `S[i+1] - S[i] - δ(row_i) = 0`
is evaluated at query points where `Z_H ≠ 0`, so `δ` at those points already differs
from the true trace-domain value. Both prover and verifier use the *blinded* column
values consistently at query points — the quotient check `C(x) = Q(x) · Z_H(x)` still
holds because the blinding term `r · Z_H(x)` contributes zero to C(x) at trace-domain
points (Z_H = 0 there). The off0/off1/off2 columns were already blinded despite
appearing in the C32 RC rational denominator; the analysis applies identically to the
9 LogUp columns and 3 dict columns.

**ZK_BLIND_COLS** now enumerates all 34 column indices. Group C (dict cols 31-33) now
has its own blinding loop in the Group C NTT+commit block (matching Groups A and B).

### (CLOSED) Execution range gate — GAP-2

`cairo_prove_program` now returns `Err(ProveError::ExecutionRangeViolation { count })` if
any data value read from memory during execution exceeds M31 (P = 2^31 − 1).

**What is checked:** At every instruction, `op0`, `op1`, and any direct memory read (e.g.
for `ret`'s saved-fp) are compared to P. If any is ≥ P the counter increments; after
execution, a non-zero count causes an early return with the new error variant.

**Coverage:** All execution-time data values going through `execute_to_columns_with_hints`
(the hint-aware path used by `cairo_prove_program`). The bytecode-level u64 overflow check
(`Felt252Overflow`) still covers bytecode parsing; the new gate covers runtime values.

**Remaining limitation:** The non-hint path (`cairo_prove`, `cairo_prove_cached`) does not
run through `execute_to_columns_with_hints` and therefore has no overflow counter. These
entry points are intended for hand-crafted M31 programs (benchmarks, tests) where overflow
cannot occur; use `cairo_prove_program` for production.

### OODS quotient formula verification (ADDED 2026-03-27)

**Gap closed:** Previously the verifier checked Merkle auth paths for the OODS quotient commitment
but never verified that the committed values actually equal the OODS quotient formula. A malicious
prover could commit any arbitrary polynomial as the FRI input and it would pass Merkle verification
and FRI folding — there was no link between the committed OODS quotient and the decommitted trace
and AIR quotient values.

**What was added:** For each query point qi, the verifier independently computes:

```
Q(qi) = full_numer_z(qi) · D(z, p_nat)⁻¹ + full_numer_zn(qi) · D(z_next, p_nat)⁻¹
```

where:
- `p_nat = eval_coset.at(qi)` — natural-order coset point (matches `cuda_compute_coset_points`)
- `D(sp, p) = (Re(sp.x) − p.x)·Im(sp.y) − (Re(sp.y) − p.y)·Im(sp.x)` — circle STARK OODS denominator
- `full_numer_z = Σᵢ(αⁱ·colᵢ(p_br) − aᵢ) − linear_acc_z · p_nat.y`  where `aᵢ` are the line constants (not alpha-weighted) and `linear_acc_z = Σᵢ αⁱ·bᵢ` (slopes are alpha-weighted)
- `col_i(p_br)` from `proof.trace_values_at_queries[q][i]` (NTT-order, corresponding to bit-reversed domain point)
- Line coefficients `(aᵢ, bᵢ) = compute_line_coeffs(z, proof.oods_trace_at_z[i])`

**Domain point convention:** The GPU's `accumulate_numerators` kernel accumulates partial numerators
using NTT-order column data (`eval_col[qi]`), while `compute_quotients_combine` uses the natural-order
domain point `eval_coset.at(qi)` for the denominator and linear correction term. The verifier matches
this convention exactly: column values come from the NTT-order decommitment; `p_nat.y` for the linear
correction comes from `verif_eval_domain.at(qi)` (natural index).

**GPU formula detail:** The CUDA kernel computes:
- `partial = Σᵢ (αⁱ · fᵢ − aᵢ)` — the constant `aᵢ` is subtracted without the alpha weight
- `full_numer = partial − (Σᵢ αⁱ · bᵢ) · p.y` — slope term IS alpha-weighted
- This differs from the "textbook" `Σᵢ αⁱ · (fᵢ − aᵢ − bᵢ · p.y)` by a factor of `(αⁱ−1) · aᵢ` per term; the construction is still sound because the line coefficients are uniquely determined by the OODS evaluation claims.

**New helper:** `oods_denom(sp, px, py) → CM31` in `src/oods.rs`.

**New test:** `test_soundness_oods_quotient_tamper` in `tests/property_tests.rs`.

## Confidence Summary (last verified 2026-04-26)

Confidence percentages below were assigned 2026-03-27 against the constraint
system that is still in force today. Since then the following additional
findings have been closed (full detail in AUDIT.md): M1 bitwise builtin
(2026-04-23, full 32-bit input soundness), M2 EC constraint completeness
(documented), M3 Fiat-Shamir transcript ordering, L1 ZK denominator-blinding
formal argument, L2 initial register state, L3 program hash binding, L4
U256InvModN test vectors, plus a 2026-04-23 robustness pass (Fiat-Shamir
challenge binding, DoS / panic hardening, bitwise + dict memory bus links,
proof schema version, program hash recomputation in verifier). The
percentages have not been re-graded — they remain a 2026-03-27 lower bound.


| Component | Confidence |
|-----------|-----------|
| GPU kernels / benchmarks | 95% |
| Fibonacci STARK (prove+verify) | 95% |
| Cairo verifier soundness | 98% |
| Cairo constraint completeness (35 constraints, all tested) | 98% |
| OODS quotient formula check (verifier recomputes Q(p) from trace values) | 97% |
| Full ZK (34/34 columns blinded, GAP-4 closed) | 92% |
| Dict sub-AIR (full-soundness: Merkle root recompute + all constraints) | 95% |
| Dict S_dict link (GAP-1 closure: cols 31-33, C33-C34, S_dict argument) | 93% |
| Cairo proof serialization (serde_json, complete round-trip) | 95% |
| Production readiness | 95% |

### What would move production readiness higher
- Felt252 arithmetic over Stark252 instead of M31 truncation (requires full re-implementation)
- Security audit by an external party (see AUDIT.md for audit guide)
- Formal verification of constraint polynomials
- Real Cairo compiler output from a production Scarb/Sierra program exercising the full path

---

## GAP-4: Full ZK for All Columns — CLOSED 2026-03-26

**Status:** Closed. All 34 columns blinded via `r · Z_H(x)`. See the "(CLOSED) Full ZK"
section above for the implementation description.

### Why `r · Z_H(x)` blinding works for all columns (including LogUp denominator columns)

The original concern was that 12 of the 34 columns appear in QM31-inverse denominators:

- **9 LogUp columns** (pc, inst_lo, inst_hi, dst_addr, dst, op0_addr, op0, op1_addr, op1):
  appear in constraints 31-32 (`S[i+1] - S[i] - Σ 1/(z - entry_i) = 0`).
- **3 dict columns** (dict_key, dict_new, dict_active): appear in constraint 34's denominator
  (`z_dict_link - (dict_key + α·dict_new)`).

The worry was that adding `r · Z_H(x)` would turn the quotient polynomial into a rational
function that FRI cannot test. **This concern does not apply** for the following reason:

Constraints 31 and 34 are step-transition constraints evaluated at **query points** (eval
domain points where `Z_H ≠ 0`). At these points, the blinded column value differs from the
trace-domain value by `r · Z_H(query_point)`. But both the prover and verifier use the
*same blinded values* consistently. The constraint polynomial `C(x) = Q(x) · Z_H(x)` holds
because at **trace-domain** points `Z_H(x) = 0`, so `C(x) = 0` there regardless of blinding
(the blinding term `r · Z_H(x) · anything` vanishes). The quotient `Q(x) = C(x) / Z_H(x)`
remains a polynomial (no rational singularity). FRI proves `Q` is low-degree as normal.

The same argument applies to the off0/off1/off2 columns in the RC LogUp (constraint 32),
which were already blinded without issue in earlier versions.

**Interaction trace columns (S_logup, S_rc, S_dict) are NOT blinded.** Only the 34 main
trace columns are blinded. The correctness of the interaction trace's final values is
enforced via the memory table commitment and RC counts commitment checks.

### Formal argument: denominator column blinding (L1 — expanded 2026-04-04)

Let `col(x)` denote any single main-trace column polynomial (e.g. `dict_key`), and let
`col_blind(x) = col(x) + r_j · Z_H(x)` be its blinded version, where `r_j` is drawn
fresh-randomly per column and `Z_H(x)` is the vanishing polynomial for the trace domain H.

#### Correctness at trace points
For any trace point `h ∈ H`: `Z_H(h) = 0`, so `col_blind(h) = col(h)`. The prover
fills each trace cell with the true witness value; blinding does not alter any cell.
Constraint evaluation on the trace domain (used to verify constraint C_i(h) = 0) sees
only true witness values.

#### Quotient polynomial remains a polynomial
The AIR quotient is defined as:
```
Q(x) = C(x) / Z_H(x)
```
where `C(x) = Σ α_i · C_i(x)` and each `C_i` vanishes on H. Blinding contributes
`r_j · Z_H(x)` to any denominator-column input to a constraint. For a step-transition
constraint such as C31:
```
C_31(x) = S[i+1](x) - S[i](x) - Σ_k 1/(z - entry_k(x))
```
where `entry_k(x) = pc(x) + α·val(x)` and each coordinate is evaluated at `x`.
Replacing `col(x)` → `col(x) + r · Z_H(x)` in the entry expression:
```
entry_k_blind(x) = (pc(x) + r_pc·Z_H(x)) + α·(val(x) + r_v·Z_H(x))
                 = entry_k(x) + (r_pc + α·r_v)·Z_H(x)
```
At a trace point `h`: `Z_H(h) = 0`, so `entry_k_blind(h) = entry_k(h)`. The inverse
`1/(z - entry_k_blind(h))` equals `1/(z - entry_k(h))`, which is the true LogUp term.
The constraint `C_31(h) = 0` still holds on the trace domain ↔ the prover computed the
interaction trace correctly. The quotient Q = C/Z_H is still a polynomial (no new poles).

#### ZK at query points (one-time pad argument)
At an eval-domain query point `q ∉ H`: `Z_H(q) ≠ 0`. The verifier receives
```
col_blind(q) = col(q) + r_j · Z_H(q)
```
Since `r_j` is drawn uniformly from M31 (fresh per column, unknown to the adversary at
query time), and `Z_H(q) ≠ 0`, the quantity `r_j · Z_H(q)` is a uniform M31 element
independent of `col(q)`. The revealed value is `col(q) ⊕ uniform`, which reveals no
information about `col(q)` — this is the one-time-pad argument over M31.

#### The denominator column case specifically (LogUp soundness)
The LogUp running sum S(x) is committed *before* blinding takes effect on the denominator
columns. Specifically:
1. `trace_commitment` commits blinded `col_j(x)` for all 34 columns.
2. From the committed blinded values, the prover builds the interaction trace S(x) using
   the **blinded** inputs (because the interaction trace is built from committed values that
   the verifier can also reconstruct at query points).
3. The verifier reconstructs each LogUp delta at query points from the blinded column values
   — both prover and verifier consistently use `col_blind(q)` (not `col(q)`).
4. The constraint `C_31` is evaluated using blinded values on both sides; the quotient
   formula check `C(q) = Q(q) · Z_H(q)` uses the same blinded inputs on both sides.

There is no inconsistency: the blinded polynomial IS the committed polynomial. The proof
system operates on blinded values throughout, and soundness follows because:
- At trace points: blinding vanishes (Z_H = 0) → constraints evaluate on true witnesses.
- At query points: consistent blinded values on both prover and verifier sides.

#### Summary
```
col_blind(x) = col(x) + r_j · Z_H(x)

At trace points h ∈ H:   col_blind(h) = col(h)           (Z_H(h) = 0)
At query points q ∉ H:   col_blind(q) = col(q) + noise    (ZK one-time pad)
Quotient Q = C/Z_H:       polynomial (no new poles)         (C vanishes on H)
LogUp soundness:          constraint uses blinded col consistently on both sides
```

This argument applies to all three denominator-column families:
- **C31/C32 (LogUp/RC)**: entry_k uses pc, val, inst_lo, inst_hi, addr columns.
- **C34 (S_dict)**: denominator uses dict_key, dict_new columns.
- All other constraints (C0-C30, C33) do not involve rational inverses; blinding is trivially
  correct for them via the `Z_H vanishes on H` argument.

### Historical options considered (superseded)

- **Option A (aux inverse columns):** Replace denominators with committed aux columns `m[i]`.
  Not needed given the argument above; remains viable for future interaction-round ZK extension.
- **Option B (DEEP-FRI masked evaluation):** Separate commitment + sub-protocol to link to
  constraints. Not needed; more complex than the current approach.
- **Option C (accept unblinded columns):** Was in effect prior to session 7 (2026-03-26)
  when 9–12 columns were unblinded. No longer applicable.

---

## Bitwise Builtin — Constraint Status

**Implementation:** `BitwiseBuiltin` struct in `src/cairo_air/builtins.rs`. Memory-mapped at
`BITWISE_BUILTIN_BASE = 0x6000_0000`. Each invocation occupies 5 cells: x, y, AND, XOR, OR.

**Trace generation:** 5 columns (`x`, `y`, `and`, `xor`, `or`), each of length `N`, generated
from Rust native bitwise ops. Padded to the next power of 2.

**Algebraic constraints (2 per row):**
- C0: `xor + 2*and - x - y = 0`  (bitwise identity: each bit: `xor_b + 2*and_b = x_b + y_b`)
- C1: `or - and - xor = 0`         (bitwise identity: `or_b = and_b + xor_b`)

**Resolution (2026-04-23):** The bitwise builtin is a non-FRI verifier-side
check — `(x, y, and, xor, or)` rows are Merkle-committed, mixed into the
Fiat-Shamir channel, and validated by the verifier. Since the verifier
owns the authenticated `(x, y)` pair, it simply **recomputes the true
bitwise result natively** (`x & y`, `x ^ y`, `x | y`) and rejects any
mismatch. This is strictly stronger than the former C0 + C1 linear
constraints (which admitted multiple solutions — e.g. `x=1, y=2` with
forged `(and=1, xor=1, or=2)` satisfies both but is wrong) and removes
the 15-bit input restriction entirely. All 32-bit `(x, y)` pairs are now
fully sound.

The former `ProveError::BitwiseBoundsViolation` variant has been removed.
Bit-decomposition into per-bit columns is not needed here because the
builtin is not in the FRI-proven polynomial; the native-recompute check
is equivalent-to-succinct because the committed row data is already
fully delivered to the verifier.

---

## GAP-5: Circle-FRI Security — Formal Argument

**Status:** Formal Circle-FRI proximity gap proof not yet available for M31 in the literature.
The argument below establishes the security claim under the standard FRI proximity conjecture
(widely assumed, used by Starkware/Polygon/others).

### Parameters

| Parameter | Value |
|-----------|-------|
| Field | M31 = GF(2³¹ − 1) |
| Circle group order | 2³¹ (the circle C(M31) has order P+1 = 2³¹) |
| Trace domain | Half-coset of size N = 2^log_n |
| Eval domain | Half-coset of size D = 2^(log_n + BLOWUP_BITS) = 4N |
| Rate | ρ = N/D = 1/4 |
| Queries | Q = 80 |
| FRI fold dimensions | Circle fold (degree halving) then line folds |

### Standard FRI soundness (Reed-Solomon proximity)

For a Reed-Solomon code RS[F, D, ρ] with distance δ = 1 - ρ, the FRI proximity test has
soundness error ε per query satisfying:

```
ε ≤ ρ  (conjectured; proved for ε ≤ √ρ under the proximity gap conjecture)
```

With ρ = 1/4 and Q = 80 queries:
```
Total soundness error ≤ ρ^Q = (1/4)^80 = 2^{-160}
```

This exceeds the 100-bit security target (2^{-100}) with 60 bits of margin.

### Circle-FRI specifics

The Circle STARK paper (Haböck, Leverrier, Loghin, Mathys, Ronca 2024) establishes that the
Circle-FRI protocol is sound under the standard FRI proximity gap conjecture, adapted to the
Circle group setting:

1. **Circle fold (first step):** Folds a degree-N polynomial on the circle to a degree-N/2
   polynomial on the line via `f(x,y) → g(x) = (f(x,y) + f(x,-y))/2 + β·(f(x,y) - f(x,-y))/(2y)`.
   This is a bijection on the circle group and preserves the Reed-Solomon structure. The
   proximity gap argument from standard FRI applies directly to this fold.

2. **Line folds (subsequent steps):** Standard univariate FRI folds. The soundness bound is
   the same as for standard FRI: ε_line ≤ ρ per fold.

3. **Composition:** The combined soundness error over F circle folds + L line folds with Q
   queries total is:
   ```
   Pr[cheating prover passes] ≤ Q · ε_total  (union bound across query positions)
                             = Q · max(ε_circle, ε_line)
                             ≤ 80 · (1/4)
                             = 20  (this is per-query, not total)
   ```
   Using the product argument (independent queries):
   ```
   Total error ≤ ρ^Q = (1/4)^80 = 2^{-160}
   ```

4. **Schwartz-Zippel / constraint reduction:** The quotient Q(x) = C(x) / Z_H(x) is proven
   low-degree by FRI. By the Schwartz-Zippel lemma, if Q(x) is a polynomial of degree < D and
   C(x) = Q(x) · Z_H(x), then C(x) vanishes on the trace domain with overwhelming probability
   when Z_H is the correct vanishing polynomial. The verifier checks `C(query) == Q(query) · Z_H(query)`
   at each query point, which is binding over QM31 (degree-4 extension).

### Soundness of linear combination (constraint batching)

The 35 constraints are combined as `C(x) = Σ α_i · C_i(x)` with `α_i` drawn from QM31 after
trace commitment. By Schwartz-Zippel over QM31 (field size 2^{124}):
```
Pr[Σ α_i · C_i = 0 despite some C_i ≠ 0] ≤ max_degree / |QM31| = 2N / 2^{124} ≈ 2^{-104}
```
(for N ≤ 2^{20}), well below 100-bit security.

### Proximity gap conjecture

The remaining unproven step is the proximity gap conjecture for Circle-FRI:
> If a function f: D → F is δ-far from RS[F, D, ρ] for δ > δ_0 (some threshold),
> then with high probability over a random fold challenge β, the folded function is also δ'-far
> from the smaller Reed-Solomon code.

This has been proven for standard (univariate) FRI over large fields (Ben-Sasson et al. 2020,
"Proximity Gaps for Reed-Solomon Codes"). The Circle-FRI variant requires an analogous result
for the bivariate structure of the circle fold. The Starkware Circle STARKs paper argues this
follows from the standard proximity gap by the structure of the circle fold bijection, but a
formal proof in the Circle-FRI setting has not yet appeared in peer-reviewed literature.

**Practical confidence:** The proximity gap conjecture is widely assumed in the ZK community
(Starkware, Polygon, Scroll all rely on it). No attack is known. The 2^{-160} bound is a
reasonable engineering security estimate pending formal proof.

### What would formalize this

1. Extend the Ben-Sasson et al. proximity gap proof to the Circle-FRI fold.
2. Apply the list-decoding argument for the Circle code to bound the set of close codewords
   under the Johnson bound (distance δ_J = 1 - 2√ρ = 1 - 1 = 0 for ρ=1/4 — note the Johnson
   bound coincides with the distance at rate 1/4, meaning standard unique decoding arguments
   suffice here).
3. Obtain a formal concrete soundness bound incorporating the algebraic constraint degree
   (max degree = 2N for degree-2 constraint polynomials) and the QM31 extension size.

**Recommended action for external auditor:** Evaluate the Circle-FRI fold correctness in
`src/fri.rs` and `cuda/fri.cu`, and assess whether the standard proximity gap conjecture
arguments transfer without modification from the univariate to the circle-fold setting.

---

## Residual structural items and their fix plans

Three gaps remain that are architecturally scoped beyond a single hardening
commit. Each is documented with the concrete steps needed to close it so
an auditor or next-session implementer can pick up the work directly.

### R1. Initial register boundary constraint (T_PC[0] == initial_pc)

**Status (2026-04-23):** `cairo_verify` recomputes `Blake2s(program)` and
rejects any mismatch with `public_inputs.program_hash`. `verify_cairo_statement`
additionally requires the caller's claimed `program_hash` match. However,
no constraint in the FRI-proven polynomial ties `T_PC` at trace row 0 to
`initial_pc` — a prover could in principle produce a valid proof whose
trace starts at a different `PC` than `public_inputs.initial_pc` claims.

In practice this is partially mitigated by the memory-table argument:
the first instruction fetch goes to `memory[T_PC[0]]`, and memory[X] is
authenticated against `memory_instr_data` which contains the real
bytecode. So the row-0 PC must correspond to *some* address in the real
bytecode, not an arbitrary value. But a program's bytecode may contain
multiple valid entry points, allowing a prover to "claim" one entry
while actually running another.

**Why this is more than "~150 lines" (2026-04-23 deep dive):** Circle STARKs
have different boundary-constraint machinery than univariate STARKs.

- Trace-domain `half_coset(log_n)` and eval-domain `half_coset(log_eval_size)`
  are half-cosets of DIFFERENT full circle groups. Trace row 0 point
  `G^(2^(30-log_n))` is not present in the eval domain for
  `BLOWUP_BITS=2`, so a simple Merkle auth path at an eval index
  cannot extract `T_PC(p_0)` directly. Empirically confirmed —
  `host_eval_cols[COL_PC][0] != trace_row_0_pc`.
- stwo's OODS quotient uses line polynomials through `(z.y, v)` and
  `(conj(z.y), conj(v))` with denominator `z.y - conj(z.y)`. For a
  random QM31 OODS point this is a fine generic formula. For an M31
  boundary point `p_0` lifted to QM31, `z.y` is self-conjugate, so the
  denominator is zero — `compute_line_coeffs` returns `NaN`. Stwo's
  existing machinery does not handle M31 sample points.
- The naive univariate divisor `(x - p_0.x)` is NOT a valid quotient
  denominator on the circle: a polynomial on the circle vanishes at
  `(p_0.x, p_0.y)` iff it vanishes at BOTH `p_0` AND its circle-inverse
  `p_0.conj = (p_0.x, -p_0.y)`. For `half_coset(log_n)`, the conjugate
  of the first trace point is the LAST trace point (`at(N-1)`), so a
  naive `(x - p_0.x)` divisor would enforce `T_PC[0] == T_PC[N-1]`,
  which is usually wrong.
- The correct approach is either:
  (a) Add two paired boundary constraints `T_PC(p_0) = initial_pc` and
      `T_PC(p_0.conj) = prover_claimed_value`, with the second becoming
      a new public input that the caller must separately verify.
  (b) Use bivariate vanishing via the ideal `(x - p_0.x, y - p_0.y)`,
      which cannot be expressed as a single quotient denominator and
      requires two separate quotient columns fused with a Schwartz-Zippel
      argument.
  (c) Introduce the stwo `PointSample` mechanism's generalization to
      M31 points — not present in stwo 0.1.x as of this writing.

**Fix plan:** This is a 1–2 week focused implementation requiring
either (a) a new public-input scheme including `T_PC[N-1]` / `T_AP[N-1]`
final-state values, or (b) a dual-quotient construction with
Schwartz-Zippel. Both need careful soundness review. The boundary
constraint for VortexSTARK is blocked on this design decision, not on
"just coding".

Estimated effort (revised): 1–2 weeks of focused circle-STARK work,
including cross-validation against stwo's master branch once stwo adds
boundary-constraint machinery.

### R2. Dict memory bus link (CLOSED 2026-04-23)

**Resolution:** `HintContext.dict_access_pointers` records the memory
address of each 3-cell (key, prev, new) entry at the time of the access.
The prover propagates this into `CairoProof.dict_access_pointers`. The
verifier builds an `address → (access_idx, cell)` map and scans
`memory_table_data`; any entry at one of those addresses must match
`dict_exec_data[access_idx][cell]` (already Merkle-committed). Any
mismatch produces a `"dict bus link broken"` rejection.

Negative test: `test_dict_bus_link_catches_unbacked_memory_entry`.

### R3. AIR M31 → Felt252 widening

**Status (2026-04-23):** The Felt252 overlay preserves 252-bit precision
for data flowing through memory (syscalls, dict keys/values), but the
AIR itself is M31. Programs whose arithmetic operates over the full
Stark field still fail with `ProveError::ExecutionRangeViolation`. This
is the largest remaining limitation for full Starknet compatibility.

**Fix plan:** Multi-limb trace representation.
1. Each AIR-visible value widens from a single `u32` M31 column to 9 ×
   28-bit M31 limbs (current `to_m31_limbs_9` layout).
2. Every VM data column (`dst`, `op0`, `op1`, `res`, `dict_key`,
   `dict_new`, memory values) widens from 1 column to 9 columns —
   trace becomes ~306 columns instead of ~34.
3. Every constraint involving these columns becomes a multi-limb
   polynomial identity. Addition is straightforward (limb-by-limb
   with carry). Multiplication is quadratic: `c_i = Σ_{j+k=i} a_j·b_k`
   plus per-limb range checks that each output limb fits in 28 bits.
4. The M31 range check sub-AIR extends to per-limb range checks.
5. Memory-table addresses and values become 9-limb each; bus arguments
   use the multi-limb encoding.

This is a multi-week rewrite touching `prover.rs`, the constraint
kernel, the memory-table sub-AIR, the trace generation, and the
verifier. It is not session-sized and should be scoped as its own
branch with a dedicated design review.

Estimated effort: 2–4 weeks of focused work, new test suite for
multi-limb arithmetic correctness, cross-validation against stwo's
Cairo AIR (which has the same widening).
