# Proof Aggregation — Design Sketch

Status: **scoping, not yet implemented**. Aggregation is the architectural
path from "169s per-block Cairo prove" to effective Starknet block cadence
(~5s). Single-proof perf tuning can't close the last order of magnitude —
aggregation can.

## Why aggregation, not more kernel tuning

Post the 2026-04-22 perf push, the prover at log_n=26 projects to roughly
50–60s on a clean RTX 5090 (extrapolating from the 71% reduction measured
at log_n=22). Remaining kernel fuses — OODS GPU barycentric, GrandProduct
LogUp, radix-64 NTT — plausibly add another 2–3× single-proof speedup.
**Best-case end state for single-proof prove at log_n=26: ~20–30s.**

Starknet block cadence is ~5s. To close the final gap, one proof has to
amortize across many blocks. Three candidate shapes:

1. **Batch-within-proof.** Prove N blocks in a single larger AIR. Linear
   cost scaling — doesn't help unless the AIR amortizes setup.
2. **Recursive aggregation.** Prove N individual blocks normally, then
   prove a single statement "all N block proofs verify." Recursion depth
   log(N). The aggregation proof is cheap (verifier circuit, not Cairo VM)
   and amortizes the per-block prove across the wall-clock window.
3. **Parallel prove + fold.** N provers run in parallel on N blocks; a
   folding/FRI-batching step combines their outputs. Needs multi-GPU or a
   cluster.

**Recommended: (2) recursive aggregation.** Cleanest integration with
existing stwo verifier, lowest operational complexity, and explicitly
what StarkWare's own roadmap is aimed at.

## What "recursive aggregation" requires

### New building block: a verifier circuit

A Cairo-level (or raw-AIR-level) *circuit* that takes a serialized
VortexSTARK proof as input and re-executes the verifier logic. Output:
one bit (accept/reject). Proving this circuit then yields a proof that
"this input proof verifies."

Components of the verifier circuit:
- Blake2s / Poseidon252 (channel + merkle)
- M31 / QM31 field arithmetic
- FRI fold equations
- Merkle path verification
- OODS quotient formula check

All are already available in the underlying stwo primitives. The work
is (a) expressing each as an AIR-level constraint, and (b) wiring the
proof bytes as public input.

**Effort: 4–8 weeks.** This is the single biggest item.

### Aggregation proof layout

Given N base proofs `π_1 .. π_N`:

1. Each base proof proves "this Cairo block transitioned state correctly."
2. Build an aggregation circuit that takes `[π_1..π_N]` as public inputs
   and verifies each one.
3. Prove the aggregation circuit → `π_agg`. Submit `π_agg` to Starknet.

**Amortization math** (sketch): if base prove is 30s and the aggregation
circuit proves 10× faster than a block (because its AIR is simpler — no
VM execution, just verifier ops), a 10-block aggregation takes
`10 × 30s (parallel base prove) + 3s agg = ~33s for 10 blocks = 3.3s per
block effective`. Beats 5s cadence.

### Multi-GPU / cluster shape

The N base proves in step 1 are embarrassingly parallel. On a single
8-GPU box, run 8 provers in parallel. On a cluster, run more. The
aggregation step is single-GPU and cheap.

Once aggregation is in place, multi-GPU is a scheduling/ops problem
(which block goes to which GPU), not a prover-engine change.

## Phased implementation plan

**Phase A — Verifier circuit (4–8 weeks)**
- [ ] Inventory every operation the verifier performs, by primitive
- [ ] Express each primitive as Cairo-level operations (Blake2s is the
      heaviest — may need a dedicated builtin / sub-AIR)
- [ ] Build a "verify one proof" Cairo program that consumes a proof as
      public input
- [ ] End-to-end test: prove any base proof, feed into verifier circuit,
      check it verifies

**Phase B — Aggregation driver (2 weeks)**
- [ ] Aggregation Cairo program that consumes `[π_1..π_N]` and invokes
      the verifier circuit N times
- [ ] Prove that program → `π_agg`; test the `π_agg` submits cleanly
      to the existing `stwo_cairo_verifier` contract
- [ ] Benchmark per-block effective cost as N varies

**Phase C — Operational plumbing (2 weeks)**
- [ ] Multi-GPU scheduler (probably a Rust service that shards block
      proving across N GPUs)
- [ ] Pipeline: ingest Starknet blocks → shard → base prove → aggregate
      → submit
- [ ] Monitoring, retries, checkpoint recovery

**Phase D — Production hardening (2–4 weeks)**
- [ ] Handle Starknet reorgs: prove-in-flight block may be orphaned
- [ ] Fee / gas accounting across the aggregated batch
- [ ] Failover across GPUs if one crashes mid-batch

Total: **10–16 weeks** with one engineer. With two engineers, phases A
and B can partially overlap, cutting ~3 weeks.

## Known interactions with other roadmap items

- **Felt252 Phase 2** lands first. The verifier circuit consumes proof
  bytes which include felt252 values (OODS z, sampled_values, etc.).
  Felt252 support has to be solid before the verifier circuit can even
  start — otherwise the circuit itself hits M31 truncation on its own
  input.
- **Kernel-level perf work** (OODS fuse, GrandProduct LogUp) continues
  in parallel. Every 2× on base prove compounds with aggregation — a
  10-block aggregation at 15s/block is 1.5s effective cadence with a
  3s aggregation overhead.
- **stwo-fork** is already public and patched for VortexSTARK needs;
  the verifier circuit work builds on the same fork.

## What the single-GPU sequencer-replacement path looks like *without* aggregation

If aggregation is not pursued, the realistic end-state is
~20–30s per block on a single RTX 5090. That is defensible for:
- Decentralized prover marketplaces (Gevulot-style) where each prover
  handles one block at a time and earns fees for it — cadence not
  critical
- Research and public-verifiability use cases
- Non-consensus workloads (private computations outsourced to a GPU)

It is **not** a drop-in replacement for a Starknet sequencer. That
requires aggregation.

## Risk / open questions

- **Blake2s in-circuit cost.** The biggest unknown. If in-circuit Blake2s
  is too expensive, the verifier circuit itself becomes a bottleneck.
  Mitigation: use Poseidon252 throughout the aggregation layer instead of
  Blake2s, accepting the leaf cost difference.
- **Proof size at aggregation.** 10 × 2.7 MB base proofs = 27 MB
  aggregation input. Whether the aggregation AIR can fit depends on how
  the public input is handled (typically hashed, with the hash committed
  as a single field value). This is solvable but needs care.
- **Recursive verifier recursion depth.** Aggregating aggregations
  (aggregate-of-aggregates) requires the verifier circuit to also verify
  aggregation proofs. In practice, two levels (base → agg → final) is
  usually sufficient for block-cadence targets.

## Recommendation

Do not start aggregation until **Felt252 Phase 2 is shipped + audited**.
Aggregation's verifier circuit rests on Felt252 correctness; a soundness
flaw in the base felt arithmetic would invalidate every aggregation
proof ever produced.

**Order of operations** for the "ready to sell" trajectory:
1. Felt252 Phase 2 complete (dict rewire + LogUp bus + SOUNDNESS.md)
2. External audit pass on Phase 2
3. Remaining kernel-level perf (OODS fuse, etc.) — in parallel with 1–2
4. Aggregation Phase A–D
5. Production deployment

Realistic timeline: **6–9 months** from today's state to "sequencer
replacement" ready.

Interim checkpoints: after (1)+(2), VortexSTARK can prove any real
Starknet contract (marketplace-ready). After (4), it can handle block
cadence. Each milestone has its own value even if the later ones slip.
