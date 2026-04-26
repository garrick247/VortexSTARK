# Felt252-in-Dicts — Design Sketch

Status: **Phase 1 complete, Phase 2 ~90% complete (2026-04-26)**. The Felt252
plumbing, side-table commitment, and Felt-overlay VM memory model have all
landed; the AIR-level dict columns remain M31. See "Phased Implementation Plan"
below for the per-checkbox state.

## Original problem statement (2026-04-22)

`src/cairo_air/trace.rs` defines the 34-column main trace:
- Columns 0..31: Cairo VM registers, flags, operands, offsets
- Columns 31..34: dict linkage (`COL_DICT_KEY`, `COL_DICT_NEW`, `COL_DICT_ACTIVE`),
  one `u32` (M31 field element) each

`src/cairo_air/hints.rs::HintContext`:
```rust
pub dict_accesses: Vec<(usize, u64, u64, u64)>,   // (step, key, prev, new)
pub dicts: HashMap<u64, HashMap<u64, u64>>,        // base → key → value
```
Every dict entry is a `u64`, and we reject inputs ≥ M31 with
`ExecutionRangeViolation`. A program that does
`dict.insert(key_that_exceeds_M31, felt252_value_that_exceeds_u64)` cannot be
proven today.

## Why This Matters

Real Starknet contracts store user balances, amounts, addresses, and hashes
as `felt252`. Dicts (`Felt252Dict`) are the idiomatic storage primitive in
Cairo 1. Without full felt252 support in dicts, the set of provable programs
is restricted to toy code that deliberately stays in M31. The user-visible
effect is:
> "We can prove your program, but only if every value it touches happens to
> be smaller than 2^31."

That is not a "production Starknet prover."

## Design Options

### Option A — 9-limb main-trace columns (heavy)

Replace the three dict columns with **17 columns**:
- `dict_key_limb[0..9]`    (9 M31 limbs, 9 × 28 = 252 bits)
- `dict_new_limb[0..9]`    (9 M31 limbs)
- `dict_active`            (1 column, unchanged)

Plus changes:
- `N_COLS` 34 → 52 (or 51 if limbs can be packed as 4 × QM31)
- Step-transition LogUp challenge must absorb all 18 key+new limbs
- Constraint C34 becomes a per-limb rational expression
- All per-column GPU kernels (NTT, Merkle, decommit) process ~50% more data
- Verifier redecomposition of each limb for boundary checks

**Cost:** ~2500 LoC across `trace.rs`, `prover.rs`, `verifier.rs`, `hints.rs`,
`dict_air.rs`, `cuda/cairo_constraint.cu`, plus audit rework for GAP-1 (the
step-transition LogUp argument's soundness proof needs to be re-derived with
18 coordinates hashed). Perf: ~15–20% slower Cairo proofs due to wider trace.

### Option B — Side table with pointer columns (recommended)

Main trace keeps 3 dict columns, but they become **pointers into a
felt252-valued side table**:

- `dict_key_ptr`, `dict_new_ptr`  — M31 integer indices into the side table
- `dict_active`                    — unchanged

Side table layout (committed as an additional Merkle tree, small sub-AIR):
- Rows: `(ptr, felt252_key_limb0..8, felt252_value_limb0..8)` — 17 M31 per row
- Verifier recomputes the Merkle root from the full side-table data (mirrors
  the existing `memory_table_commitment` pattern)
- LogUp bus links main-trace `(dict_key_ptr, dict_new_ptr)` to side-table rows
- felt252 equality / ordering happens outside the SNARK: on the side-table row
  that the prover supplies, the verifier recomputes Poseidon252 (or Blake2s) of
  the limbs to bind pointer ↔ value

Changes by file:
- `hints.rs`: `dict_accesses: Vec<(step, Felt252, Felt252, Felt252)>` where
  `Felt252 = [u32; 9]`
- `dict_consistency.rs`: `verify_chain` operates on `Felt252`, not `u64`
- `dict_air.rs`: S_dict LogUp hashes full-felt rows into `z_dict_link` /
  `alpha_dict_link`; N_INTERACTION_COLS unchanged (QM31 absorbs multiple M31s)
- `trace.rs`: `COL_DICT_KEY`/`COL_DICT_NEW` now hold pointer indices, not
  values — column count stays at 34
- `prover.rs`: build + commit the side table; tamper tests for each felt
  coordinate
- `cuda/cairo_constraint.cu`: unchanged for main constraints; side-table
  decommitment uses existing Merkle helpers
- `verifier.rs`: one new section that (1) recomputes side-table Merkle root,
  (2) verifies LogUp bus consistency
- Public API: `cairo_prove_program_with_syscalls` accepts `Felt252` in
  `SyscallState::storage` and the public-input `initial_*` fields

**Cost:** ~1500 LoC across ~8 files. Perf: ~5–8% overhead on Cairo proofs
(one extra Merkle commit + LogUp bus). Security: GAP-1 soundness argument
adapts cleanly — the multiset permutation is over 18-coordinate rows instead
of 2-coordinate rows, which LogUp handles natively with a bigger random
linear combination.

### Option C — Blake2s-hashed dict columns (smallest, weakest)

Keep 3 columns, but `dict_key` / `dict_new` hold the **low 32 bits of a
Blake2s hash** of the full felt252. Equality is by hash. Collisions are
negligible under 2^32 entries.

**Cost:** trivial — maybe 200 LoC.

**Risk:** this is a cryptographic shortcut, not a proof. A malicious prover
could craft a collision and swap values. **Do not ship.** Documenting only
as a "cheap prototype to see what integration breaks" option.

## Recommendation

Go with **Option B (side table with pointers)**.

- Main trace stays at 34 columns — no pervasive changes
- Perf impact is 5–8%, acceptable for correctness
- Soundness argument is a straightforward extension of the existing S_dict
  LogUp with a wider challenge, not a rebuild
- Verifier cost is one extra Merkle path + LogUp check
- Aligns with the existing `memory_table_commitment` / `rc_counts_commitment`
  pattern (full data in proof, verifier recomputes hash)

## Phased Implementation Plan

**Phase 1 — Felt252 plumbing (1 week)** — landed 2026-04-23
- [x] `crate::felt252` module: `Felt252` is a transparent alias of
      `cairo_air::stark252_field::Fp` with `from_u64`, `from_hex`,
      `to/from_le_bytes`, `low_u64`, `try_to_u64`, `to_m31_limbs_9`. See
      `src/felt252.rs`.
- [x] `SyscallState`: `storage: HashMap<Felt252, Felt252>`,
      `caller_address`, `contract_address`, `entry_point_selector`,
      `SyscallEvent.keys/data`, `CrossContractCall.*`, `DeployedContract.*`,
      `L1Message.*` — all widened. See `src/cairo_air/hints.rs`.
- [x] CASM loader: produces `bytecode_felt: Vec<Felt252>` alongside the
      truncated `bytecode: Vec<u64>`; `parse_hex_felt252` is lossless. See
      `src/cairo_air/casm_loader.rs::parse_hex_felt252`.
- [ ] `ProveError::Felt252Overflow` deprecated (no overflows to report)
      — **kept**: bytecode entries that exceed u64 are still rejected at
      `prover.rs:716-717`. Removing this would let truncation happen
      silently, which is the bug Phase 1 was meant to prevent. The error
      now signals "your program needs Phase 2 dict-side-table coverage,
      not just Phase 1 plumbing"; an audit may rephrase it accordingly.

**Phase 2 — Dict side table (1 week)** — landed 2026-04-23 except item #1
- [~] `HintContext::dict_accesses` → `(step, Felt252, Felt252, Felt252)`:
      **partial**. The original u64 quadruple log
      (`dict_accesses: Vec<(usize, u64, u64, u64)>`) is preserved as the
      AIR-authoritative log; a parallel
      `dict_accesses_felt: Vec<(usize, Felt252, Felt252, Felt252)>` was
      added for full-precision capture from syscall paths. The side-table
      commit reads from `dict_accesses_felt` when populated, falling back
      to `Felt252::from_u64(dict_accesses)` otherwise. Replacing the u64
      log entirely would force every direct (non-syscall) dict write to
      route through Felt252, which still reduces mod M31 at the AIR
      boundary — i.e. a refactor with no soundness gain. Left as-is.
- [x] `dict_consistency.rs::verify_chain` on Felt252:
      `verify_chain_felt` in `src/cairo_air/dict_consistency.rs:194`.
- [x] Side-table construction + Merkle commit in `prover.rs`:
      `dict_side_table: Vec<[u32; 28]>` (pointer + 9 key limbs +
      9 prev limbs + 9 new limbs) with `dict_side_table_commitment`
      (Blake2s of flat u32 serialization). Mixed into channel right
      after `dict_trace_commitment`, before `z_dict_link` is drawn —
      see `src/cairo_air/prover.rs:1185, 3011-3012`.
- [x] Verifier: side-table reconstruction + LogUp bus check:
      `prover.rs:3304-3322` (canonical-limb encoding enforced
      `< 2^28` per limb, side-table re-Blake2sed and compared to
      committed root, low-31-bit projection cross-checked against the
      Merkle-authenticated `dict_exec_data`).
- [x] Soundness note in `SOUNDNESS.md`: see SOUNDNESS.md
      "(PARTIALLY MITIGATED — Phase 2 in progress) Felt252 truncation"
      and AUDIT.md "Dict memory bus link / R2 (2026-04-23)".

**Phase 3 — Test + audit**
- [x] Tamper tests: corrupt side-table row, corrupt pointer, wrong limb.
      Three dedicated tests in `src/cairo_air/prover.rs`:
      `test_dict_side_table_tampered_low_limb_rejected` (key-limb bit
      flip + commitment rehash → S_dict link cascade rejection),
      `test_dict_side_table_noncanonical_limb_rejected` (limb ≥ 2^28
      caught by the canonical-encoding check at prover.rs:3310-3320),
      `test_dict_side_table_tampered_pointer_rejected` (row pointer
      corruption + commitment rehash → downstream link rejection). All
      three pass on 2026-04-26.
- [ ] End-to-end: prove `LegacyMap<felt252, felt252>` style contract
      with non-M31 values; verify. Requires a real Cairo 1 contract
      that exercises the syscall→overlay→side-table path. The unit test
      `test_dict_felt252_values_roundtrip` (per VortexSTARK status
      memory) covers the prover-side path; an end-to-end Cairo
      contract test is the real validation.
- [ ] External audit pass — out of session scope.

**Phase 4 — Deployment (as needed)**
- [x] Update README constraint coverage — see "Remaining limitations"
      and "Known remaining weak points" sections, refreshed 2026-04-26.
- [ ] Update `STARKNET_POST.md` to remove the "felt252 dict" caveat
      (line 104 still references this design doc as 3–4 weeks of work;
      update to reflect Phase 2 closure when Phase 3 audit lands).
- [ ] Bump version, tag release — out of session scope.

Original estimate: 3–4 weeks with one engineer, plus ~1 week for audit.
Actual elapsed: Phase 1 + Phase 2 main body landed in a single session
(2026-04-23). Phase 3 audit-side work and Phase 4 deployment remain.

## Known Interactions

- **RPC auto-resolve (shipped 2026-04-22):** class_hashes are 252-bit
  poseidon hashes; once `Felt252` is threaded through, the resolver can
  accept full class_hashes instead of truncating to u64.
- **stwo wire format:** stwo already represents all values as `Felt252`
  (`M31 × [4]` / `QM31` pairs). VortexSTARK's internal M31 representation
  will continue to match at the stwo boundary; the Felt252 lift is internal
  to dict handling.
- **Performance:** Phase 2 adds one Merkle tree (depth log_n). At log_n=26
  this is ~0.5s additional, ~0.3% of the 169s prove time. Negligible.
