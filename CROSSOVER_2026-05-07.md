# stwo-cairo SimdBackend vs vortex-cuda-backend — byte-identical proofs, Cuda wins

**Date:** 2026-05-07
**Hardware:** RTX 5090 (32 GB), Linux box
**Code:** stwo-cairo dev-head + vortex-cuda-backend (cuda-backend branch in stwo-fork)
**Benchmark program:** scaled array_sum (recursive populate + recursive sum), proof_mode

## TL;DR

Cuda is faster than Sim at every measured size AND produces a
byte-identical proof. End-to-end MD5(prove\_cuda) = MD5(prove\_sim) at
N=100/2000/10000/50000.

| N      | Sim prove | Cuda prove | Cuda lead | Proof MD5 match |
|--------|-----------|------------|-----------|-----------------|
|   100  |  4.06 s   |  4.01 s    |  -0.05 s  | YES             |
|  2,000 |  4.23 s   |  4.05 s    |  -0.18 s  | YES             |
| 10,000 |  4.16 s   |  4.06 s    |  -0.10 s  | YES             |
| 50,000 |  4.56 s   |  4.26 s    |  -0.30 s  | YES             |

## Three fixes that closed the loop

### Fix 1 — barycentric path masquerading as eval_at_point (~1100x OOD speedup)

The first sweep showed Cuda 10-15x slower than Sim with a 55 s constant
overhead in EvaluateOutOfDomain. Counter instrumentation showed the
slow code path was actually `barycentric_eval_at_point`, not
`eval_at_point` — `Poly::eval_at_point` falls back to the barycentric
branch when `store_polynomials_coefficients = false`, which is the
production stwo-cairo default. Setting the flag to `true` routed OOD
through the GPU Horner-fold `eval_at_point` instead. **OOD: ~55 s →
~50 ms.** Cuda then sat ~2 s behind Sim.

Also added an `EvalAtPointWorkspace` buffer pool (mirrors the existing
`FoldWorkspace` pattern) and a small-poly CPU fallback (`n < 1024`)
in `vortex-cuda-backend/src/poly_ops.rs`.

### Fix 2 — subdomain-aware QuotientOps (~16x FRI quotient speedup)

Profiling localized the remaining ~2 s gap inside `Compute FRI
quotients` (Sim 40-60 ms, Cuda 2.3-3.1 s). Both
`accumulate_numerators` and `compute_quotients_and_combine` had hard
CPU fallbacks for `log_blowup_factor != 0` (Cairo runs with
`log_blowup_factor = 1`), because upstream stwo had moved both
functions onto the subdomain. The fallbacks pulled all GPU columns to
host, ran CpuBackend, and pushed results back. Fix:

1. **`accumulate_numerators`**: drop the fallback. Compute
   `subdomain_size = column_size >> log_blowup_factor` and pass that as
   the kernel's `n_rows`. The bit-reversed-prefix property
   guarantees the first `subdomain_size` slots of each column are the
   right input — no kernel changes needed.
2. **`compute_quotients_and_combine`**: drop the fallback. Pass
   `subdomain_log_size` as the kernel's `lifting_log_size` to
   produce subdomain-sized output, then lift each of the 4 SoA
   channels with `CudaBackend::interpolate` (IFFT on subdomain) +
   `CudaBackend::evaluate` (FFT on full eval domain). Both are
   existing GPU ops; they look up GPU twiddles via the internal coset
   cache.

After both fixes, Cuda was 0.05-0.29 s FASTER than Sim at every N. But
the proof was still not byte-identical to Sim's.

### Fix 3 — match SimdBackend's grind nonce-search pattern (byte-parity)

Diffing the two proofs showed: `claim`, `commitments[0]`
(preprocessed merkle root), and `commitments[1]` (base trace merkle
root) were identical, but `commitments[2]` (interaction trace),
`commitments[3]` (composition), `interaction_pow`, all
`interaction_claim.*.claimed_sum` values, and all `sampled_values`
diverged. Working backward, the divergence was `interaction_pow` (Sim
21475236060 vs Cuda 42628793) — the proof-of-work nonce drawn after
base trace commit. Both nonces are valid, but the search strategies
differ:

- `SimdBackend::grind`: enumerates nonces of the form
  `(hi << 32) | low` with `hi = 0, 1, ...` outer and
  `low ∈ [0, 2^20)` inner. Returns the smallest qualifying nonce in
  that family.
- `CudaBackend::grind` (before): linear search 0, 1, 2, ... in
  batches of 2^24. Returns the smallest qualifying nonce overall.

Different families → different smallest-qualifying nonces → different
`interaction_pow` value → channel state diverges → all downstream
proof bytes differ.

**Fix:** rewrite `grind_gpu_blake2s` to match Sim's pattern. Iterate
`hi = 0, 1, ...` and call `cuda_grind_pow` with
`batch_offset = (hi << 32)` and `batch_size = 2^20`. The kernel's
existing atomicMin returns the smallest qualifying nonce in each
batch, so returning on the first `hi` that finds one gives the
smallest `(hi << 32) | low` overall — same as Sim. Search cost is
~64 launches at 2^20 each for Cairo's pow_bits=26 (~64 ms), barely
visible in total prove time (~4 s).

Result: byte-identical proofs at every N tested.

## Validation

- Full VortexSTARK suite green: 396 passed / 0 failed / 3 ignored.
- Sim and Cuda proofs are byte-identical (MD5 match) at N=100, 2000,
  10000, 50000.
- The new grind matches Sim's nonce form so the proof's pow nonce
  is also reduced mod P (each 32-bit half < P = 2^31 - 1) — same
  structural invariant Sim asserts.

## Code changes shipped

| File | Change |
|------|--------|
| `crates/cuda-backend/src/poly_ops.rs` | `EvalAtPointWorkspace` cache + small-poly CPU fallback (n<1024) + `eval_at_point_stats_take()` AtomicU64 counter |
| `crates/cuda-backend/src/quotient_ops.rs` | Drop CPU fallback for log_blowup_factor != 0 in both `accumulate_numerators` (subdomain n_rows) and `compute_quotients_and_combine` (subdomain compute + IFFT/FFT lift) |
| `crates/cuda-backend/src/grind_ops.rs` | Match SimdBackend's (hi<<32 \| low) nonce search: outer loop on hi, inner batch of 2^20 nonces. Same code path for both Blake2sChannelGeneric variants |
| `crates/cuda-backend/src/lib.rs` | Re-export `eval_at_point_stats_take` |
| `stwo-cairo prover/src/lib.rs` | Re-export `eval_at_point_stats_take` behind cuda-backend feature |
| `stwo-cairo dev_utils/src/bin/prove_cuda_from_input.rs` | `store_polynomials_coefficients: true`, `[stats]` print, `serialize_proof_to_file` (matches Sim) |
