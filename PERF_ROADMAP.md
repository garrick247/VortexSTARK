# VortexSTARK Cairo Prove — Performance Roadmap

Status: **measurement, not plan**. Numbers are real, prioritization is mine.

## Measured (2026-04-22, BigDaddy WSL2 / RTX 5090 / clean GPU)

Phase timings via `VORTEXSTARK_PROFILE=1 stark_cli prove-file fibonacci.casm --log-n <N>`.
All measurements at `BLOWUP_BITS=2` (production default, 160-bit security).

### log_n=20 — 8.9s prove, 2^20 trace rows

| Phase                  | Time     | %      |
|------------------------|---------:|-------:|
| oods                   | 2808.6ms | 31.5%  |
| ntt_blind_commit       | 2548.5ms | 28.6%  |
| phase2_logup_rc        | 1788.5ms | 20.1%  |
| phase3_quotient        |  794.6ms |  8.9%  |
| phase4_fri             |  388.8ms |  4.4%  |
| phase5_pow_decommit    |  234.5ms |  2.6%  |
| sdict_interaction      |  189.1ms |  2.1%  |
| setup + all others     |   <1ms   |  <1%   |

**Top-3 = 80.2% of prove time.**

### log_n=22 — 45.7s prove, 2^22 trace rows

| Phase                  | Time      | %      |
|------------------------|----------:|-------:|
| ntt_blind_commit       | 15107.2ms | 33.0%  |
| oods                   | 11172.4ms | 24.4%  |
| phase2_logup_rc        | 11149.7ms | 24.4%  |
| phase3_quotient        |  3615.5ms |  7.9%  |
| phase4_fri             |  1554.2ms |  3.4%  |
| sdict_interaction      |  1512.7ms |  3.3%  |
| phase5_pow_decommit    |   905.1ms |  2.0%  |

**Top-3 = 81.8% of prove time.**

### log_n≥24 — not measured this session

WSL2 hit `Wsl/Service/E_UNEXPECTED` catastrophic-failure on multiple attempts.
Symptom: GPU allocation or CUDA-runtime crash propagates to WSL host and kills
the VM. Not reproduced on native Linux / stark_cli Fibonacci (which uses a
different prove path). Treat as a WSL-environment issue to investigate
separately — likely related to the Cairo prover's peak-memory footprint at
log_n≥24 interacting with WSL2 GPU-PV limits.

## Scaling Behavior

8.9s (log_n=20) → 45.7s (log_n=22) = **5.1×** for 4× data. Roughly `n log n`.

Extrapolating naively to log_n=26 (64× the data of log_n=20) predicts ~700s,
but the README cites **169s** at that size. Real prove time scales sub-linearly
in practice because GPU parallelism hides constant-factor work as the batch
grows. The top-3 phases continue to dominate — their proportions stay within
a few percentage points across the measured sizes.

**Expected top-3 at log_n=26:** three phases each in the 30–50s range, totaling
~130–140s of the 169s budget. Everything else combined is ~30s.

## Optimization Targets by Phase

### 1. `oods` — 24–31% of total

Out-of-domain sampling currently involves:
- Evaluating every trace column at the OODS point `z` and `z_next` via INTT + circle fold
- Computing AIR quotient values at `z` (4 quotient columns × INTT + fold)
- Accumulating OODS-quotient numerators over ~50 columns with alpha weighting
- Two rounds of `mix_felts` (at `z`, at `z_next`)

**Low-hanging fruit:**
- [ ] **Fuse INTT + point-evaluation into a single kernel.** The per-column INTT
      emits a full polynomial; we only need one evaluation point. Barycentric
      evaluation on the eval-domain samples avoids the INTT entirely and runs in
      O(n) GPU work per column with high parallelism. Rough win: 30–50% of
      `oods`.
- [ ] **Batch the column scans.** We currently loop per column with per-column
      cudaMalloc/free cycles. Batch all 50+ columns in one kernel launch with
      shared-memory tiling.

**Bigger lift:**
- [ ] Move to stwo's `quotients_on_line_coset` accumulator — a single kernel
      that produces all OODS quotient components without materializing per-column
      intermediate polynomials.

### 2. `ntt_blind_commit` — 29–33% of total

This phase does:
- Group-batched forward NTT over all committed trace columns (34 cols + 3 dict + 12 interaction ≈ 50)
- Per-column `r · Z_H(x)` ZK blinding addition
- Merkle tree construction (Blake2s leaves grouped by `log_size`)

**Low-hanging fruit:**
- [ ] **Fuse blinding into NTT kernel.** Currently NTT writes eval-domain
      column, then a second kernel adds `r · Z_H`. Fold the blinding into the
      NTT post-butterfly stage — zero extra round-trips to HBM. Rough win:
      10–15% of `ntt_blind_commit`.
- [ ] **Reuse twiddle cache across the three NTT batches** (Group A, B, C).
      Already done for the eval twiddles; verify the INTT twiddles are shared.

**Bigger lift:**
- [ ] **Switch to radix-64 NTT** using SM_120's shuffle primitives. The current
      radix-4 kernel leaves half the SM utilization on the table for very large
      `log_n`. Rough win: 20–30% at log_n≥24.

### 3. `phase2_logup_rc` — 20–24% of total

LogUp interaction (memory argument) + range-check table construction:
- Per-column fraction arithmetic over QM31 (mul, inverse)
- Batch modular inverse via Montgomery batch-inverse
- Parallel prefix sum for cancellation
- Commit of three interaction trees (t1/t2/t3 for LogUp, u1/u2 for RC)

**Low-hanging fruit:**
- [ ] **Merge prefix-sum + inverse into a single kernel.** Today: batch inverse
      kernel → prefix-sum kernel → per-row combine kernel. All three touch the
      same data and can be fused. Rough win: 15–25% of `phase2_logup_rc`.
- [ ] **Precompute the range-check multiplicity table more aggressively.**
      `rc_counts` is computed CPU-side today for some paths; move to GPU.

**Bigger lift:**
- [ ] **Replace LogUp with Plonky3-style GrandProduct** for the memory
      argument — stwo's recent work shows 2–3× wins for wide traces. Requires
      constraint-level rework and verifier changes.

## What 2–3× Gets You vs. Block Cadence

Starknet mainnet block cadence is ~5s. Current Cairo prove at log_n=26 is
~169s. **Delta: 34×.**

Ceiling of each path:
- Top-3 optimization pass (everything above): 2–3× total speedup → ~60–80s
- Algorithmic swap (GrandProduct + barycentric OODS + fused kernels): another
  2–3× → ~20–30s
- Plus broader architectural wins (dynamic batching of multiple small blocks,
  partial recursion to aggregate proofs, different field choice): another
  2–3× → 8–12s

**Realistic end state after 2 quarters of focused perf work with 1 engineer:**
~20–30s prove at log_n=26 — **inside an order of magnitude of block cadence**,
not at it. To actually hit 5s, you need one of:
- Larger GPU or multi-GPU proving
- Proof aggregation so a single proof covers many blocks, amortizing prove
  cost (e.g., a 20-block bundle proved in 30s = 1.5s per block effective)
- Moving to a different proving protocol (Plonky3 with FRI-Binius, Boojum, etc.)

## Prioritization for Next Session

If the goal is **"make Cairo prove 2× faster this week":**
1. Fuse OODS INTT + point-evaluation (biggest single win)
2. Fuse NTT + blinding
3. Fuse LogUp prefix-sum + inverse

If the goal is **"match StarkWare's prover":**
- Forget incremental fuses — plan the proof-aggregation path + multi-GPU
  scaling directly.

## How to Reproduce

```bash
cargo build --release --bin stark_cli
VORTEXSTARK_PROFILE=1 ./target/release/stark_cli prove-file \
    tests/fixtures/fibonacci.casm --log-n 22 -o /tmp/out.bin
```

Data in this file reflects `9b365f2 .. f0933ce` main-branch code.
