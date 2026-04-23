# Felt252-in-Dicts — Design Sketch

Status: **scoping, not yet implemented**. This document enumerates the work
required to lift VortexSTARK's `Felt252Dict` support from M31-only (the
current limitation) to full 252-bit values, so that real Starknet contracts
that store `felt252` values in dicts (almost all of them) can be proven.

## Current State (2026-04-22)

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

**Phase 1 — Felt252 plumbing (1 week)**
- [ ] `crate::felt252` module: `pub struct Felt252([u32; 9])` with `add`,
      `sub`, `mul_mod_p`, `to_limbs`, `from_hex`
- [ ] `SyscallState`: storage, caller, contract_address, etc. typed as `Felt252`
- [ ] CASM loader: stop truncating values > u64; produce `Felt252` directly
- [ ] `ProveError::Felt252Overflow` deprecated (no overflows to report)

**Phase 2 — Dict side table (1 week)**
- [ ] `HintContext::dict_accesses` → `(step, Felt252, Felt252, Felt252)`
- [ ] `dict_consistency.rs::verify_chain` on Felt252
- [ ] Side-table construction + Merkle commit in `prover.rs`
- [ ] Verifier: side-table reconstruction + LogUp bus check
- [ ] Soundness note in `SOUNDNESS.md`

**Phase 3 — Test + audit (1 week)**
- [ ] Tamper tests: corrupt side-table row, corrupt pointer, wrong limb
- [ ] End-to-end: prove `LegacyMap<felt252, felt252>` style contract with
      non-M31 values; verify
- [ ] External audit pass (likely needed — GAP-1 changes)

**Phase 4 — Deployment (as needed)**
- [ ] Update README constraint coverage
- [ ] Update `STARKNET_POST.md` to remove the "felt252 dict" caveat
- [ ] Bump version, tag release

Total: 3–4 weeks with one engineer, plus ~1 week for audit.

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
