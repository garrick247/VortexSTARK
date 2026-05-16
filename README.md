# VortexSTARK

GPU-native Circle STARK prover with end-to-end proof generation and verification on NVIDIA Blackwell (RTX 5090) and Ada Lovelace (RTX 4090). Rust + CUDA.

## What is this?

To our knowledge, the only public Circle STARK prover with a real GPU backend. The upstream `stwo` reference implementation (Starkware) runs on CPU; VortexSTARK is a from-scratch CUDA implementation of the same construction — trace generation, NTT, Merkle commitment, LogUp interaction, FRI, and proof-of-work all run kernel-side with no host round-trips on the hot path.

Two STARK flavors ship in-tree:
- **Fibonacci STARK** — the canonical 1-column / 1-transition example, used as the prover/verifier baseline and benchmark target
- **Cairo VM STARK** — full 34-column trace with 35 transition constraints, prove arbitrary Cairo programs (.casm or fetched from Starknet)

Nine of the prover's GPU kernels are emitted by **[Forge](https://github.com/garrick247/forge)** (a formally-verified systems language with Z3 proof discharge) and run default-on. Total of **0 user-supplied `assume()`** across the forge-emitted surface — every fact in the production prover is SMT-discharged.

```
Forge (.fg)  ──►  CUDA C  ──►  nvcc  ──►  cubin  ──►  GPU (in-tree prover)
                  ↑ 9 kernels in production: NTT, NTT-batch, FRI, blake2s,
                    permute, bit-reverse, gather, barycentric, grind
```

Open-toolchain build (no NVIDIA compiler) is also supported via the [OpenCUDA](https://github.com/garrick247/opencuda) + [OpenPTXas](https://github.com/garrick247/openptxas) pair — they consume the same Forge-emitted CUDA C / PTX. Cross-stack tooling (run / compare / benchmark / classify any kernel through any backend) lives in [forge-workbench](https://github.com/garrick247/forge-workbench).

Source-available under [BSL 1.1](LICENSE), converts to Apache 2.0 on **2029-03-20**. Non-production use permitted today; commercial licensing available before the conversion date — contact garrick.wagner@gmail.com.

## Status

### End-to-end proven and verified on hardware

- **Fibonacci STARK**: Full prove → verify pipeline, 100-bit conjectured security, 18+ tamper-detection tests
- **Cairo VM STARK**: 34-column trace, 35 transition constraints, verifier independently evaluates all constraints at query points
- **LogUp memory consistency**: Full cancellation check — memory table committed as explicit proof data (all unique entries), verifier independently checks exec_sum + table_sum == 0
- **Pedersen hash**: GPU windowed EC scalar multiplication, 40.9M hashes/sec at 1M-batch, verified against CPU reference
- **Poseidon2 hash**: GPU trace generation + NTT, 5.7M hashes/sec at log_n=28 (30 rows/perm, RF=8 RP=22)
- **RPO-M31 hash**: Circle STARK–native hash (eprint 2024/1635), 3.5M hashes/sec at log_n=28 (14 rows/perm, 24 cols)
- **FRI**: Circle fold + line folds, GPU-resident decommitment, all fold equations verified

### Benchmarked (RTX 5090, CUDA 13.2, driver 595.58.03)

Default build, `BLOWUP_BITS=2`, 160-bit security, N_QUERIES=80, PoW=26 bits:

| Workload | Scale | Prove | Verify |
|----------|-------|-------|--------|
| Fibonacci log_n=24 | 16.8M elements | 133ms | 5.9ms |
| Fibonacci log_n=28 | 268M elements | 2.52s | 7.4ms |
| Fibonacci log_n=29 | 537M elements | currently fails (`cudaMalloc(2.1 GB)` panic on fresh process despite >30 GB VRAM free) — tracking as an allocator regression; pre-regression value was 8.9s | — |
| Cairo VM log_n=20 | 1M steps | 1.37s | 0.11s |
| Cairo VM log_n=22 | 4.2M steps | 5.65s | 0.39s |
| Cairo VM log_n=24 | 16.8M steps | 23.96s | 1.80s |
| Cairo VM log_n=25 | 33.5M steps | 80s | ~5s |
| Cairo VM log_n=26 | 67M steps | runs via chunked-slab quotient kernel (commit `7a7b2b2`); requires ≥40 GB host RAM for the parallel `cn`/`hc` host copies. Native-Linux wallclock TBD — WSL2's 31 GB host ceiling caps in-house validation at log_n=22 | — |
| Poseidon2 trace+NTT log_n=28 | 8.9M hashes | 1.57s | — |
| RPO-M31 trace+NTT log_n=28 | 19.2M hashes | 5.51s* | — |
| Pedersen GPU batch | 1M hashes | 25.7ms | — |

\* RPO-M31 log_n=28 currently fails on fresh process (same `cudaMalloc` regression as Fibonacci log_n=29); value shown is the last known measurement. RPO-M31 log_n=24 measured today at 5.55M hashes/sec.

`bench-max-size` feature build (`cargo build --release --features bench-max-size`),
`BLOWUP_BITS=1`, 80-bit security — used to measure the 1B-element headline, which
requires log_eval ≤ 31 (the order of M31's circle subgroup):

| Workload | Scale | Prove | Verify |
|----------|-------|-------|--------|
| Fibonacci log_n=28 | 268M elements | 2.97s | 6.9ms |
| Fibonacci log_n=29 | 537M elements | 8.95s | 7.8ms |
| Fibonacci log_n=30 | 1.07B elements | 19.68s | 7.6ms |

log_n=30 requires ~20 GB of free VRAM and a single 8 GB pinned→device transfer.
On WSL2 this works on systems with enough allocated to the VM; native Linux is
recommended for the headline measurement.

All Cairo VM numbers include 34 columns, 35 constraints, full LogUp+RC memory table, S_dict LogUp, OODS quotient, and 26-bit proof-of-work. Fibonacci numbers are the standalone single-column prover.

Default-build numbers above re-measured 2026-05-16 via fresh-process-per-size warm-iterate harness (cold + warm prove, warm time reported). Cairo VM log_n=25 and bench-max-size headlines retained from prior measurement; flag if these need re-validation.

### Adversarial soundness (constraint coverage)

34-column trace, 35 transition constraints. The following are now enforced:

- **Operand address verification**: dst_addr, op0_addr, op1_addr constrained against register + offset - bias
- **JNZ fall-through**: dst_inv auxiliary column, fall-through constrained to pc + inst_size when dst=0
- **JNZ inverse consistency**: dst * dst_inv = 1 enforced when jnz and dst != 0
- **Op1 source exclusivity**: pairwise products of op1_imm, op1_fp, op1_ap all constrained to zero
- **PC update exclusivity**: pairwise products of jump_abs, jump_rel, jnz constrained to zero
- **Opcode exclusivity**: pairwise products of call, ret, assert constrained to zero
- **Flag binary**: all 15 flags constrained to {0, 1}
- **Instruction decomposition**: all 63 bits verified — inst_lo + inst_hi ≡ off0 + off1·2^16 + off2·2^32 + flags·2^48 (mod P)
- **LogUp memory consistency**: memory table committed as explicit proof data; verifier checks exec_sum + table_sum == 0
- **Range check argument**: all 16-bit offsets verified via LogUp against precomputed table, wired into prover with z_rc challenge
- **Merkle hash construction**: leaf and internal-node hashes both use standard Blake2s without h[6] personalization, matching stwo's `MerkleHasherLifted::hash_children` (see `blake2s_hash_node` in `src/channel.rs`, which calls `blake2s_hash` with domain=0x00, and `IV6_NODE = IV6` in `cuda/blake2s.cu`). Leaf inputs are sized as `min(n_cols, 16) × 4` bytes; internal-node inputs are always 64 bytes — different lengths produce different Blake2s `t0` counters when n_cols < 16. For 16-column commits, leaf-vs-node distinction relies on Blake2s collision resistance and protocol structure (verifier knows the depth at each auth-path step) rather than algebraic personalization
- **Full ZK**: all 34 trace columns blinded via `r · Z_H(x)` — GAP-4 closed 2026-03-26

### Remaining limitations

- **Felt252 arithmetic**: VM operates over M31 (2^31 − 1). Three layers of detection prevent silent misproofs:
  1. Bytecode values wider than u64 are rejected at load (`ProveError::Felt252Overflow`).
  2. Any `res_add` / `res_mul` operation on operands ≥ M31 bumps `hint_ctx.execution_overflows`; the prover rejects with `ProveError::ExecutionRangeViolation` if non-zero.
  3. Pass-through of > M31 values (memory moves, syscall blobs, dict writes) is **intentionally allowed**: trace columns store low-31 bits but the full felt precision is preserved via the `dict_side_table` (Phase 2, Blake2s-committed, mixed into Fiat-Shamir). The low-31 ↔ exec-log link is verified at `prover.rs:3515–3556` (verifier-side `cairo_verify` recomputes `exec_key_new_sum` from `dict_exec_data` and rejects if it does not equal `proof.dict_link_final`). So a program can carry felt252 values through memory and dicts without triggering rejection, and the proof binds the full 252-bit value at every dict access.
  A program that performs M31-overflowing **arithmetic** is correctly rejected; a program that just moves felts around is correctly proven at full precision.
- **Starknet syscalls**: All 9 syscall selectors fully implemented. `CallContract` and `LibraryCall` execute registered callees in-process via `HintContext::register_contract` — retdata is written back into the caller's response buffer with nesting up to depth 8. Unregistered targets return empty retdata. `Deploy` returns a deterministic mock address (`salt XOR class_hash`). All syscall state (events, calls, deployed contracts, L1 messages) recorded in `SyscallState` and available after proving.
- **Dict consistency proofs**: Dict read/write execution is fully functional. An execution-side chain consistency check runs at prove time (`ProveError::DictConsistencyViolation`). The S_dict step-transition LogUp (C34) links main trace dict columns to an authenticated exec trace; verifier checks `dict_link_final == exec_key_new_sum`. Soundness holds against a malicious prover for dict-heavy programs.
- **Bitwise builtin (32-bit sound)**: The bitwise builtin commits `(x, y, x&y, x^y, x|y)` rows into the Fiat-Shamir channel and the verifier recomputes the true bitwise result natively from the authenticated `(x, y)`, rejecting any mismatch. All 32-bit inputs are supported.
- **Initial register state (boundary constraint)**: The verifier does not enforce a hard AIR boundary constraint `T_PC[0] == initial_pc`. The initial register state (`initial_pc`, `initial_ap`) is part of the **public input** (the *statement* being proven) and is trusted by the verifier as the starting point. `cairo_verify` does enforce a sanity guard that `initial_pc < program.len()` so a proof cannot claim execution from outside the committed bytecode, but it does not check `initial_pc` / `initial_ap` against any caller-supplied expected value. Callers who need to verify that a proof corresponds to a specific initial state should use `verify_cairo_statement(&proof, &stmt)`, which compares `proof.public_inputs.{initial_pc, initial_ap, n_steps, program_hash}` against `stmt` before delegating to `cairo_verify`. This is the standard STARK convention: the verifier checks that the computation starting from the stated initial state satisfies all transition constraints; it does not re-derive the initial state from the program.
- **Program-hash binding (verifier-enforced; caller still picks the program)**: `proof.public_inputs.program_hash` is `Blake2s(bytecode)`. `cairo_verify` recomputes `Blake2s(proof.public_inputs.program)` and rejects any internal hash/bytecode inconsistency before Fiat-Shamir replay (closed 2026-04-23 — see AUDIT.md). What the verifier does **not** do is decide which program the caller intended to run — it accepts whatever bytecode the proof's public input carries. To bind a proof to a specific program, supply the expected `program_hash` in a `CairoStatement` and use `verify_cairo_statement(&proof, &stmt)`, which compares `proof.public_inputs.program_hash` against `stmt.program_hash` (use `compute_program_hash(&bytecode)` to derive the expected hash). The remaining caller responsibility is choosing which `CairoStatement` to verify against — not recomputing the hash, which the verifier already does.

See [SOUNDNESS.md](SOUNDNESS.md) for the full constraint-by-constraint analysis.

## Architecture

- **Full-group NTT** on subgroup(2^31) via ForwardTwiddleCache
- **Fused quotient + circle fold**: zero host transfer for quotient data
- **FRI arena**: MOVE layers into storage, zero clones, GPU-resident decommitment
- **Two-phase commitment**: trace → Fiat-Shamir → LogUp interaction → quotient → FRI
- **GPU LogUp interaction**: batch QM31 inverse + parallel prefix sum (16ms at 16.7M steps)
- **Pinned DMA**: async trace download overlapped with FRI folding at PCIe 5.0 bandwidth

### stwo CudaBackend — zero CPU fallbacks

All stwo proof system operations run GPU-native. No host round-trips inside the hot path:

| Operation | Kernel | Notes |
|-----------|--------|-------|
| Circle NTT evaluate/interpolate | `circle_ntt_stwo.cu` | Cached twiddle tree per coset |
| Merkle leaf hashing (Blake2s) | `merkle_leaves_lifted.cu` | Grouped by log_size, chunked ×16 |
| Merkle leaf hashing (Poseidon252) | `merkle_poseidon252.cu` | Montgomery-form Fp252 throughout |
| Blake2s PoW grind | `grind.cu` | 16M nonces/launch, ~1ms/batch |
| Blake2s M31-output grind | `grind_m31_output.cu` | Applies M31 reduction before trailing-zero check |
| Poseidon252 PoW grind | `grind_poseidon.cu` | GPU Poseidon permute in Montgomery form |
| GKR fix_first_variable (M31→QM31) | `gkr.cu` | fold_mle_evals per element |
| GKR fix_first_variable (QM31→QM31) | `gkr.cu` | fold_mle_evals per element |
| GKR gen_eq_evals | `gkr.cu` | log_k sequential doubling passes |
| GKR next_layer (all 4 variants) | `gkr.cu` | Grand product + 3 LogUp variants |
| GKR sum_as_poly (all 4 variants) | `gkr.cu` | Parallel block reduction + CPU accumulate |
| lift_and_accumulate | `accumulate_lift.cu` | QM31 per-channel, src_idx = (i>>shift<<1)\|(i&1) |
| QM31 bit-reverse | `bit_reverse_wide.cu` | Out-of-place, thread i → bit_reverse(i) |
| pack_leaves_input | `pack_leaves.cu` | 4×N → 64×(N/16) gather transpose |
| eval_at_point_by_folding | `fri.cu` (existing) | GPU circle+line fold, twiddle cache; CPU for n<1024 |

## Cairo VM AIR

- **Instruction decoder**: 15 flags, 3 biased offsets, full Cairo encoding
- **VM executor**: add, mul, jump, jnz, call, ret, assert_eq (26ns/step fused)
- **34-column trace**: registers(3), instruction(2), flags(15), operands(7), offsets(3), dst_inv(1), dict linkage(3)
- **35 transition constraints** evaluated on GPU (single CUDA kernel) and independently by verifier
- **LogUp memory consistency**: permutation argument with execution + table sum cancellation
- **Range check argument**: 16-bit offset validation via LogUp bus, wired into prover pipeline
- **Instruction decomposition**: algebraic constraint tying inst_lo/inst_hi to offsets and flags

## Builtins

| Builtin | Status | Throughput (log_n=28) |
|---------|--------|----------------------|
| Poseidon2 | GPU kernel, proven (RF=8 RP=22, 30 rows/perm) | 4.7M hashes/sec |
| RPO-M31 | GPU kernel, proven (14 rows/perm, 24 cols) | 3.5M hashes/sec |
| Pedersen | GPU kernel, proven (windowed 4-bit EC, Montgomery Jacobian) | 37.7M hashes/sec |
| Bitwise | Trace generation + verifier native recompute against Fiat-Shamir-bound (x, y) — see AUDIT.md §M1 | AND/XOR/OR on full 32-bit inputs |

## CLI

```bash
stark_cli prove 24 1 1 -o proof.bin          # Fibonacci STARK
stark_cli prove-file program.casm -o proof.bin # Cairo program
stark_cli prove-starknet --class-hash 0x...    # From Starknet mainnet
stark_cli inspect program.casm                 # Disassemble CASM
stark_cli fetch-block --block 100000           # Starknet block info
stark_cli verify proof.bin                     # Verify a proof
stark_cli bench 28                             # Benchmark
```

## Building

Requires: Rust 1.85+ (stable), CUDA 13.0+, RTX 5090 (SM 12.0) or RTX 4090 (SM 8.9).

```bash
cargo build --release
cargo test --release --workspace -- --test-threads=1
# Workspace totals: 396 lib + 49 vortex-cuda-backend + 35 integration = 480 pass, 0 fail, 5 #[ignore] (3 lib + 1 integration live-RPC opt-in + 1 doctest)
cargo run --release --bin full_benchmark
cargo run --release --bin gpu_bench     # pre-flight checks + per-section GPU telemetry
```

The default build enables all 9 Forge-emitted kernel paths (`forge-ntt`, `forge-ntt-batch`, `forge-fri`, `forge-blake2s`, `forge-permute`, `forge-bit-reverse`, `forge-gather`, `forge-barycentric`, `forge-grind`). Compare against the hand-written CUDA baseline with `cargo test --no-default-features` (394 pass).

## Tests

396 lib + 49 vortex-cuda-backend + 35 integration = **480 total (480 pass, 0 fail, 5 marked `#[ignore]` for benchmark/live-RPC opt-in)** covering: M31/CM31/QM31 field arithmetic, Circle NTT, Merkle tree (commit, auth paths, tiled, SoA4), FRI (fold, circle fold, deterministic), STARK prover + verifier (multiple sizes, tamper detection), Cairo VM (decoder, executor, Fibonacci, constraints, LogUp, range checks, instruction decomposition), Poseidon, Pedersen (Stark252 field, EC ops, GPU vs CPU), Bitwise (memory segment, trace generation, verifier native recompute against Fiat-Shamir-bound (x, y), prove/verify round-trip, tamper detection, 32-bit input acceptance, forged-row rejection), LogUp/RC soundness (memory table commitment, cancellation check, RC counts commitment), OODS quotient formula correctness, GPU constraint eval (bytecode VM, warp-cooperative), GPU leaf hashing (Blake2s, domain separation), CASM loader, Cairo hints (AllocSegment, AllocFelt252Dict, dict entry lifecycle, squash, U256InvModN with 7 comprehensive test vectors), Fiat-Shamir transcript ordering (12 commitment points), property tests (completeness, soundness, random mutations), cross-validation (reference VM comparison for 9 program types).

## Break This System

If you can craft a malformed trace that the verifier accepts, that is a real bug. Open an issue.

### Known remaining weak points

- **Felt252 arithmetic**: bytecode values wider than u64 are rejected at load (`Felt252Overflow`). Arithmetic (`res_add` / `res_mul`) on operands ≥ M31 is rejected at prove time (`ExecutionRangeViolation`). Pass-through of felt252 values through memory and dict writes is sound: trace stores low-31 bits, full precision preserved via the Phase 2 dict side table. The previously-documented "(M31, u64] silently reduces" caveat applied only before the Phase 2 side table landed; the current pipeline binds full felt precision at every dict access.
- **Cross-contract execution**: `CallContract` and `LibraryCall` execute registered callees in-process (`HintContext::register_contract`). Unregistered targets return empty retdata. External contract resolution (fetching bytecode from Starknet RPC at call time) is not automatic.
- **Felt252 dict values**: AIR-level dict columns are M31, but the Phase 2 `dict_side_table` (Blake2s-committed, pointer + 9 M31 limbs each for key/prev/new) binds the full 252-bit value at every dict access. **The standard Cairo 1 `Felt252Dict` hint path (`Felt252DictEntryUpdate` in `hints.rs:1132`) pushes to both the u64 log (`dict_accesses`) and the full-felt log (`dict_accesses_felt`) on every write** — whether the value originated from a syscall lift (`mem.set_felt`) or from direct in-program code that writes to the dict entry cell. The prover-side side-table builder uses the felt log when populated (which it always is via the hint), so any normal Cairo 1 program that uses `Felt252Dict<felt252, felt252>` round-trips full 252-bit values bit-exact through the proof. The only theoretical residual is code that populates dict memory cells without going through the standard Cairo 1 hint — which isn't reachable from normal Cairo source

### Expected to hold (guarantees today)

These should **not** break. If they do, that's a real soundness bug:

- Honest prover produces proofs that verify for Fibonacci, add, mul, call/ret, and mixed programs
- Verifier independently evaluates all 35 constraints at query points and rejects any mismatch
- Operand addresses verified against register + offset for all three operands (dst, op0, op1)
- JNZ fall-through constrained: dst_inv auxiliary column forces next_pc = pc + inst_size when dst = 0
- Flag exclusivity enforced: op1 source, PC update mode, and opcode are pairwise mutually exclusive
- Tampering any committed value (trace, quotient, FRI, commitment) is detected
- LogUp final-value enforcement: corrupting the memory consistency sum breaks FRI verification
- FRI fold equations: algebraic consistency checked at every query across all layers
- Merkle auth paths: data integrity verified for trace, quotient, OODS quotient, and all FRI layers
- OODS quotient formula: verifier independently recomputes Q(p) from decommitted trace and AIR quotient values and rejects any mismatch
- Fiat-Shamir transcript: any mutation to public inputs, commitments, or challenges cascades into rejection

### Tamper tests (all passing)

| Test | What it tampers | Result |
|------|----------------|--------|
| `test_tamper_flag_binary` | Set flag to non-binary value | REJECTED |
| `test_tamper_result_computation` | Corrupt `res` column | REJECTED |
| `test_tamper_pc_update` | Corrupt `next_pc` | REJECTED |
| `test_tamper_ap_update` | Corrupt `next_ap` | REJECTED |
| `test_tamper_fp_update` | Corrupt `next_fp` | REJECTED |
| `test_tamper_assert_eq` | Corrupt `dst != res` | REJECTED |
| `test_tamper_logup_final_sum` | Corrupt LogUp sum | REJECTED |
| `test_tamper_rc_final_sum` | Corrupt range check sum | REJECTED |
| `test_tamper_memory_table_data` | Corrupt memory table entry | REJECTED |
| `test_tamper_memory_table_commitment` | Corrupt memory table hash | REJECTED |
| `test_tamper_logup_cancellation` | Corrupt exec_sum (cancellation fails) | REJECTED |
| `test_tamper_rc_counts_data` | Corrupt RC multiplicity count | REJECTED |
| `test_tamper_rc_counts_commitment` | Corrupt RC counts hash | REJECTED |
| `test_cairo_tampered_program_hash` | Corrupt program hash | REJECTED |
| `test_cairo_prove_verify_tampered_quotient` | Corrupt quotient value | REJECTED |
| `test_cairo_prove_verify_tampered_fri` | Corrupt FRI value | REJECTED |
| `test_tamper_ec_trace` | Corrupt EC trace commitment | REJECTED |
| `test_soundness_oods_quotient_tamper` | Corrupt OODS quotient decommitment value | REJECTED |
| `test_cairo_prove_verify_tampered_commitment` | Corrupt trace commitment root | REJECTED |
| `test_tamper_trace_auth_paths` | Corrupt trace Merkle auth-path | REJECTED |
| `test_tamper_ec_trace_auth_paths` | Corrupt EC trace Merkle auth-path | REJECTED |
| `test_tamper_interaction_decommitment` | Corrupt LogUp interaction decommit value | REJECTED |
| `test_tamper_rc_interaction_decommitment` | Corrupt RC interaction decommit value | REJECTED |

## Use as a Stwo CudaBackend

VortexSTARK also exposes `vortex-cuda-backend` — a `stwo::prover::Backend`
implementation that routes Stwo's prove pipeline through the same GPU
kernels described above. This makes it a drop-in CudaBackend for the
upstream [`stwo`](https://github.com/starkware-libs/stwo) crate (via
[`garrick247/stwo-fork`](https://github.com/garrick247/stwo-fork) for the
PoC trait extensions).

End-to-end measured on stwo-cairo's 12 shipped `test_data/`
programs (RTX 5090 + Core Ultra 9 285K, warm prove vs CPU SimdBackend
with `target-cpu=native`):

| Program | CPU warm | CUDA warm | Speedup |
|---|---:|---:|---:|
| `test_prove_verify_ret_opcode`            | 3.08 s | 0.368 s | **8.4x** |
| `test_prove_verify_bitwise_builtin`       | 3.12 s | 0.395 s | **7.9x** |
| `test_prove_verify_range_check_bits_128`  | 3.08 s | 0.400 s | **7.7x** |
| `test_prove_verify_range_check_bits_96`   | 2.93 s | 0.395 s | **7.4x** |
| `test_prove_verify_add_mod_builtin`       | 2.88 s | 0.403 s | **7.2x** |
| `test_prove_verify_poseidon_builtin`      | 3.31 s | 0.482 s | **6.9x** |
| `test_prove_verify_mul_mod_builtin`       | 2.90 s | 0.445 s | **6.5x** |
| `test_poseidon_aggregator`                | 2.83 s | 0.469 s | **6.0x** |
| `test_prove_verify_all_opcode_components` | 2.89 s | 0.583 s | **5.0x** |
| `test_prove_verify_pedersen_builtin`      | 5.37 s | 1.523 s | **3.5x** |
| `test_pedersen_aggregator`                | 5.34 s | 1.504 s | **3.5x** |
| `test_prove_verify_all_builtins`          | 5.64 s | 2.057 s | **2.7x** |

**Median 6.7x, range 2.7x – 8.4x.** All 12 programs produce
**byte-identical proofs** to the CPU SimdBackend and verify via
`verify_cairo_ex`. Reproducible measurement protocol and per-PR
attribution in [BENCHMARKS.md](BENCHMARKS.md) (commit `f4d2bb6`, sweep
dated 2026-05-14).

To prove a Cairo program with this backend, see the
[`cuda-backend-poc` branch of stwo-cairo](https://github.com/garrick247/stwo-cairo/tree/cuda-backend-poc),
which has the integration wired into `run_and_prove_cuda`. The
`stwo_cairo_prover` crate enables it behind a `cuda-backend` feature
flag that pulls `vortex-cuda-backend` from this repository.

## Author

Garrick Wagner — independent GPU systems work. Contact: garrick.wagner@gmail.com.

## License

Business Source License 1.1 ([LICENSE](LICENSE)). Non-production use permitted. Converts to Apache License 2.0 on 2029-03-20. Commercial licensing: garrick.wagner@gmail.com.
