# VortexSTARK: GPU-Native Circle STARK Prover for Starknet

## TL;DR

I built a GPU-native Circle STARK prover that produces `stwo_cairo_verifier`-compatible proofs on a single NVIDIA RTX 5090. The full stwo `CudaBackend` is implemented with zero CPU fallbacks. Every test passes, every proof verifies. Security audit closed.

Two measured modes:

| Mode | Fibonacci log_n=30 | Security | Use |
|------|---|---|---|
| Default (`BLOWUP_BITS=2`) | not available — max log_n=29 at 8.9s | 160-bit | production proofs |
| `bench-max-size` feature | **19.68s / 1.07B elements, verified** | 80-bit | scaling headline |

Cairo VM proving runs end-to-end through the GPU pipeline. The CLI emits `cairo-serde` JSON proofs submittable directly to the `stwo_cairo_verifier` Starknet contract.

## What's Implemented

**stwo CudaBackend (4,200+ lines, 50+ tests):**
- `PolyOps`, `FriOps`, `MerkleOpsLifted`, `GrindOps`, `QuotientOps`, `ComponentProver`, `ColumnOps`
- All GPU-native — zero CPU fallbacks inside the proving pipeline
- Two Merkle leaf hash choices (Blake2s + Poseidon252 for mainnet hash compat)

**GPU Kernels (all CUDA, SM_120 optimized):**
- Circle NTT — stwo-compatible twiddle format, fused shared-memory tiling
- FRI — `fold_line`, `fold_circle_into_line`, on-demand twiddle computation
- Merkle trees — GPU Blake2s + Poseidon252 (leaves, nodes, tiled subtree roots)
- Constraint evaluation — two kernels (per-thread ≤1024 registers; warp-cooperative via `__shfl_sync` for >1024)
- FRI quotient accumulation with batch column evaluation
- Batch modular inverse, bit-reverse permutation

**Architecture:**
- M31 / CM31 / QM31 field stack — identical to stwo
- Pinned DMA at full PCIe 5.0 bandwidth (37 GB/s measured)
- GPU-resident FRI layer decommitment (tile-based, ~3MB vs 16GB)
- Lazy pinned buffer pool amortized across proofs
- Chunked pinned→device transfers (512 MB chunks) for WSL2 GPU-PV compatibility

## Benchmarks (RTX 5090, CUDA 13.2, clean GPU)

### Default build — `BLOWUP_BITS=2`, 160-bit security, 80 queries, 26-bit PoW

| Workload | Scale | Prove | Verify |
|----------|-------|-------|--------|
| Fibonacci log_n=24 | 16.8M elements | 214ms | 6.2ms |
| Fibonacci log_n=28 | 268M elements | 1.55s | 8.2ms |
| Fibonacci log_n=29 | 537M elements | 8.9s | 7.8ms |
| Cairo VM log_n=22 | 4.2M steps | ~7s | ~1s |
| Cairo VM log_n=24 | 16.8M steps | 23.9s | ~3s |
| Cairo VM log_n=25 | 33.5M steps | 86s | ~5s |
| Cairo VM log_n=26 | 67M steps | OOM (needs >32 GB VRAM) | — |
| Poseidon2 trace+NTT log_n=28 | 8.9M hashes | 1.92s | — |
| Pedersen GPU batch | 1M hashes | 26.6ms | — |

### `bench-max-size` feature — `BLOWUP_BITS=1`, 80-bit security (scale demo)

| Workload | Scale | Prove | Verify |
|----------|-------|-------|--------|
| Fibonacci log_n=28 | 268M elements | 2.97s | 6.9ms |
| Fibonacci log_n=29 | 537M elements | 8.95s | 7.8ms |
| Fibonacci log_n=30 | **1.07B elements** | **19.68s** | 7.6ms |

### What the proof actually looks like

1B-element Fibonacci proof: 2.7 MB, 20 FRI layers, 80 queries, PoW nonce.
Verifies in 7.6ms. Full Merkle auth paths, OODS quotient, LogUp final sums,
RC multiplicity commitments all checked.

## End-to-End Starknet Pipeline

```bash
# Fetch + prove a mainnet contract with cross-contract auto-resolve
stark_cli prove-starknet \
    --class-hash 0x... \
    --resolve-rpc-callees \
    --rpc https://starknet-mainnet.public.blastapi.io \
    --stwo-output proof.cairo-serde.json
```

The proof goes directly to the `stwo_cairo_verifier` Starknet contract as
cairo-serde calldata. No reformatting step.

When a `CallContract` / `LibraryCall` targets an unregistered class_hash,
the resolver auto-fetches its compiled CASM via JSON-RPC, registers it
in-process, and executes — so a single `stark_cli` invocation covers the
full call graph of a real contract interaction.

## Security

**Audit closed (2026-04-04):** M1 bitwise bounds, M2 EC completeness, M3 FS
ordering, L1 ZK blinding proof, L2 initial register state, L3 program hash,
L4 U256InvModN. All findings fixed and retested.

**Soundness:** 34-column trace, 35 transition constraints. 17+ tamper tests
(trace, quotient, FRI, LogUp sums, RC sums, memory table, program hash, PoW)
all reject. Full ZK blinding on all 34 columns via `r · Z_H(x)`.

**Field:** M31 by design — stwo's Circle STARK field, not a limitation.
Proofs are submittable directly to Starknet's on-chain verifier.

## What's Not Yet

- **Felt252 in dicts**: dict values are M31 today. Real Starknet contracts
  store `felt252` — design plan is in `FELT252_DESIGN.md`, ~3–4 weeks + audit.
  Option B (side table + pointer columns) recommended — 5–8% perf hit, no
  pervasive trace changes, GAP-1 soundness argument extends cleanly.
- **Cairo VM prove speed**: log_n=25 measures 198s (OODS is 40%, 80s). log_n=26 currently OOMs on 32 GB VRAM — the full 34-column eval-domain trace requires >32 GB. Super-linear scaling in OODS, phase3_quotient, and phase5_pow_decommit is the dominant issue. Profiler identifies top-3
  phases (`oods`, `ntt_blind_commit`, `phase2_logup_rc`) = 80% of prove time.
  Roadmap in `PERF_ROADMAP.md` projects 2–3× via kernel fusion, further 2–3×
  via algorithmic swaps (GrandProduct LogUp, barycentric OODS). Block cadence
  (~5s) reaches via proof aggregation or multi-GPU, not just kernel work.
- **Full bitwise soundness**: inputs constrained < 2^15 today (rejects above);
  31-bit needs bit-decomposition.

## Why This Matters for Starknet

Starknet's production prover runs on CPU. Decentralized proving will need
competitive GPU economics. VortexSTARK shows the stwo Circle STARK protocol
runs entirely on GPU with significant speedups — and the `CudaBackend`
plugs directly into the existing stwo-cairo pipeline.

Nethermind's `stwo-gpu` effort (189 commits, no published benchmarks or
releases) is the only other GPU attempt. VortexSTARK ships with:
- Passing test suite (356/356)
- Real benchmarks from real measurements
- Closed audit
- On-chain-submittable proofs
- A working RPC auto-resolve flow against mainnet

## About

Built over several months — GPU architecture research, kernel development,
and low-level NVIDIA reverse engineering. All benchmarks on a single RTX 5090
with SM_120 hand-optimized kernels. Apache-2.0-converting BSL 1.1 license
(converts 2029-03-20).

Looking for:
- Starknet Foundation grant support for the Felt252 rework and perf work
- Proving-marketplace partnerships (Gevulot-style prover-node integrations)
- Direct engagement with the stwo team on `CudaBackend` upstreaming

Happy to run benchmarks on specific workloads, walk through the code, or
demo the full cairo-serde → Starknet verifier round-trip.

**Repo:** https://github.com/garrick99/VortexSTARK
