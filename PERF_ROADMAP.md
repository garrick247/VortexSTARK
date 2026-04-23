# VortexSTARK Cairo Prove — Performance Roadmap

Status: **shipped — most of the projected speedup from the original baseline
landed in this branch.** This document tracks the measured before/after plus
remaining work.

## Headline

At log_n=22 (fibonacci.casm padded to 2^22 rows, 24-core Intel 285K, RTX 5090):

| Version | Prove time | Δ from baseline |
|---------|-----------:|---:|
| Baseline (`9b365f2`) | 45.7s | — |
| After 15 perf commits (`147dd7f`) | **13.25s** | **−71.0%** |

Proof size and program hash unchanged modulo ZK blinding randomness.
All tamper tests pass. Proof verifies OK.

## Phase Breakdown

### Baseline (log_n=22, pre-optimization)

| Phase                  | Time      | %      |
|------------------------|----------:|-------:|
| ntt_blind_commit       | 15107.2ms | 33.0%  |
| oods                   | 11172.4ms | 24.4%  |
| phase2_logup_rc        | 11149.7ms | 24.4%  |
| phase3_quotient        |  3615.5ms |  7.9%  |
| phase4_fri             |  1554.2ms |  3.4%  |
| sdict_interaction      |  1512.7ms |  3.3%  |
| phase5_pow_decommit    |   905.1ms |  2.0%  |

### After (log_n=22, same binary)

| Phase                  | Before     | After     | Δ       |
|------------------------|-----------:|----------:|--------:|
| ntt_blind_commit       | 15107.2ms  |  3831.1ms | −74.6%  |
| oods                   | 11172.4ms  |  2887.8ms | −74.2%  |
| phase2_logup_rc        | 11149.7ms  |  3398.2ms | −69.5%  |
| phase3_quotient        |  3615.5ms  |   854.9ms | −76.4%  |
| phase4_fri             |  1554.2ms  |   288.3ms | −81.5%  |
| sdict_interaction      |  1512.7ms  |   480.1ms | −68.3%  |
| phase5_pow_decommit    |   905.1ms  |   804.6ms | −11.1%  |
| **Total**              | **45736.8ms** | **13250.4ms** | **−71.0%** |

Every phase got faster; no phase was left untouched.

## What Landed (15 commits, in order)

1. **`f27c45a`** — Avoid D→H→D round-trip in 3 NTT commit groups. `ntt_blind_commit` −9.4%.
2. **`57591b1`** — Same pattern in sdict/logup/rc/commit_qm31. `sdict` −15%.
3. **`3cb11d2`** — GPU permute kernel (`cuda_permute_hc_to_canonic_brt`) for quotient commit. `phase3_quotient` −17.7%.
4. **`cab9bd8`** — Reuse GPU interaction buffers for OODS, delete dead `srcs_hc`. `oods` −11.5%.
5. **`2286603`** — rayon parallelize T1/T2/T3 + U1/U2 logup/RC CPU loops. `phase2_logup_rc` −11.6%.
6. **`a97a875`** — rayon parallelize N_COLS trace-column OODS evaluation. `oods` −38.7%.
7. **`dd67c53`** — rayon parallelize OODS Phase 1.5 interaction-poly evals. `oods` −19.5%.
8. **`302b696`** — rayon parallelize canonic-BRT→hc-natural permute (3 commit groups, 34 cols). `ntt_blind_commit` −50.2%.
9. **`5644cf2`** — rayon parallelize the 4-column permute sites (sdict/logup/rc/commit_qm31). `phase2_logup_rc` −51.7%, `sdict_interaction` −55.8%.
10. **`3a208d4`** — GPU permute in `ntt_col_save` + `stwo_ntt_lde` helpers. `phase2_logup_rc` −21.9%.
11. **`1a4ef10`** — rayon parallelize `vh_inv` + `trans_factor` eval-domain precomputes. `phase3_quotient` −64.7%.
12. **`a1fa5ca`** — rayon parallelize `d_zh`, quotient numerator, domain-point precomputes. `ntt_blind_commit` −41.4%, `oods` −32.1%.
13. **`6fafe6b`** — rayon parallelize FRI circle-fold twiddle precompute. `phase4_fri` −80.9%.
14. **`9cda8c9`** — rayon parallel gather in `extract_memory_table`. `total` −3.1%.
15. **`147dd7f`** — keep trace hc-natural buffers GPU-resident from commit to quotient (`cuda_permute_canonic_brt_to_hc_natural`). `phase3_quotient` −17.7%.

## Pattern: what actually moved the needle

The optimizations that shipped fall into three classes:

1. **Parallelize the big CPU loops.** The prover had many
   `for i in 0..eval_size` sequential loops building domain-point lookups,
   running per-row QM31 arithmetic, or gathering data for Merkle commits.
   At eval_size = 2^24+ these are 100s of ms of single-threaded work while
   the GPU idles. rayon par_iter + computing each index independently
   (via `coset.at(i)` instead of `pt = pt.mul(step)`) scales linearly to
   the 24 available cores. **This was the dominant win.**

2. **Stop round-tripping through host.** Many commit/INTT paths downloaded
   GPU data to host, permuted on CPU, re-uploaded. Replaced with a GPU
   permute kernel (`cuda_permute_hc_to_canonic_brt` + inverse) and
   kept committed data GPU-resident when possible.

3. **Kill dead code.** One N_COLS×eval_size CPU permute result was
   computed but never read. Deleted.

## What's NOT done

These are still on the table for further speedup, but in each case the
implementation is genuinely new kernel/algorithm work — not a one-line fuse:

- **Fuse OODS INTT + barycentric eval.** We still INTT 12 interaction
  polys + 34 trace polys on GPU, then evaluate on CPU. A
  one-kernel barycentric evaluator (existing dormant kernel in
  `cuda/barycentric_eval.cu` is too limited — needs rework for n > 64).
  Projected remaining win: ~30–50% of the remaining `oods` phase.
- **Replace LogUp with GrandProduct.** Protocol-level change, 2–3× win
  on wide traces per stwo upstream.
- **Radix-64 NTT using SM_120 shuffle primitives.** Current radix-4
  kernel leaves SM utilization on the table for very large log_n.
- **Proof aggregation.** To hit Starknet block cadence (~5s), per-block
  prove can't keep up at a single log_n=26 proof. Aggregating many small
  proofs is the architectural path there, not just kernel tuning.

## Scaling Notes

log_n=22 went from 45.7s to 13.25s (−71%). The relative cost of each
remaining phase is now much flatter — no single phase owns >30% of the
prove. Future optimization needs to attack several simultaneously, or
move to the architectural items above.

**log_n=24, 26 not re-measured this session.** WSL2 kernel 6.6 + Cairo
prove at that size still crashes the VM (documented in `WSL_SETUP.md`).
The perf wins should scale proportionally — most are eval_size-denominated
parallelism — so log_n=26 should drop from ~169s to somewhere in the
~50–60s range. Needs a clean-WSL measurement to confirm.

## How to reproduce

```bash
cargo build --release --bin stark_cli
VORTEXSTARK_PROFILE=1 ./target/release/stark_cli prove-file \
    tests/fixtures/fibonacci.casm --log-n 22 -o /tmp/out.bin
./target/release/stark_cli verify /tmp/out.bin
```

Data in this file reflects `9b365f2 .. 147dd7f` on `main`.
