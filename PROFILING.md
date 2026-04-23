# VortexSTARK Profiling Guide

Quick reference for the built-in phase profiler in the Cairo prover.

## Enabling

Set the environment variable before running any binary that calls
`cairo_prove_program`, `cairo_prove_program_with_ctx`, or
`cairo_prove_cached` (transitively via `stark_cli prove-file`,
`stark_cli prove-starknet`, `bench_cairo_heavy`, etc.):

```bash
VORTEXSTARK_PROFILE=1 ./target/release/stark_cli prove-file <program.casm>
```

The profile is **zero-cost when unset** — a single `std::env::var_os` check
at prove entry decides whether to emit any timing output.

## What You See

Each phase emits one line with the phase name, wall-clock ms inside that
phase, and cumulative ms since prove start:

```
  [perf] setup+public_inputs              0.2ms  (cum     0.2ms)
  [perf] phase1_trace_extend              0.1ms  (cum     0.3ms)
  [perf] ntt_blind_commit              2548.5ms  (cum  2548.8ms)
  [perf] sdict_interaction              189.1ms  (cum  2737.9ms)
  [perf] ec_trace                         0.0ms  (cum  2737.9ms)
  [perf] dict_sub_air                     0.0ms  (cum  2737.9ms)
  [perf] phase2_logup_rc               1788.5ms  (cum  4526.4ms)
  [perf] bitwise_commit                   0.0ms  (cum  4526.4ms)
  [perf] mem_table_commit                 0.2ms  (cum  4526.6ms)
  [perf] rc_counts_commit                 0.8ms  (cum  4527.4ms)
  [perf] phase3_quotient                794.6ms  (cum  5322.0ms)
  [perf] oods                          2808.6ms  (cum  8130.6ms)
  [perf] phase4_fri                     388.8ms  (cum  8519.4ms)
  [perf] phase5_pow_decommit            234.5ms  (cum  8753.9ms)
```

## Phase Glossary

| Phase                   | What it does |
|-------------------------|---|
| `setup+public_inputs`   | Argument parsing, program-hash commitment, public-input assembly |
| `phase1_trace_extend`   | Append dict-linkage columns (31–33) to the 31-col VM trace |
| `ntt_blind_commit`      | Group-batched forward NTT over all committed columns + `r · Z_H` blinding + Merkle commit |
| `sdict_interaction`     | S_dict step-transition LogUp interaction trace (GAP-1 closure) |
| `ec_trace`              | EC trace for Pedersen builtin (no-op if no Pedersen inputs) |
| `dict_sub_air`          | Dict consistency sub-AIR (exec vs sorted permutation argument) |
| `phase2_logup_rc`       | Fused memory-LogUp + range-check interaction with batch inverse |
| `bitwise_commit`        | Bitwise builtin full-data Blake2s commitment |
| `mem_table_commit`      | Memory table (addr,val,mult) Blake2s commitment |
| `rc_counts_commit`      | Range-check multiplicity table Blake2s commitment |
| `phase3_quotient`       | AIR quotient evaluation on GPU (35 constraint bytecode) |
| `oods`                  | Out-of-domain sampling: eval all columns at z/z_next + OODS-quotient construction + Merkle commit |
| `phase4_fri`            | FRI on the OODS quotient (circle fold + line folds) |
| `phase5_pow_decommit`   | Blake2s 26-bit PoW grind + query decommitments across all 7+ trees + FRI layers |

## Capture to File

Profiling output goes to **stderr**. Redirect appropriately:

```bash
VORTEXSTARK_PROFILE=1 ./target/release/stark_cli prove-file prog.casm 2> profile.log
```

## Reading the Output

1. **Dominant phase** = largest ms. At log_n ≥ 20, expect `oods`,
   `ntt_blind_commit`, and `phase2_logup_rc` to each be 20–33% of total.
2. **Skewed to one phase** = dig in. Each phase maps to a module:
   - `oods` → `src/cairo_air/prover.rs:1982` through the OODS block
   - `ntt_blind_commit` → `src/cairo_air/prover.rs:909` (group-batched NTT)
   - `phase2_logup_rc` → `src/cairo_air/prover.rs:1303` (LogUp interaction)
3. **Very small trace sizes** (log_n ≤ 10) are dominated by kernel-launch
   overhead and do not reflect the prove cost at scale. Use log_n ≥ 20 for
   meaningful optimization-target measurements.

## Known Caveats

- **Sync-measured**: each `phase_tag` captures wall-clock at that point on
  the host thread. GPU kernels launched inside a phase that don't sync
  before the tag are under-counted (next phase gets the tail). Phase
  boundaries are placed at natural sync points (commitments, FRI rounds),
  so accuracy is good at the inter-phase granularity but **don't trust
  intra-phase splits without a `cudaDeviceSynchronize`**.
- **WSL2 variance**: large-trace (`log_n ≥ 24`) Cairo runs can hit
  `Wsl/Service/E_UNEXPECTED` catastrophic failure on some WSL2 kernel
  versions. Unrelated to the profiler; documented in `PERF_ROADMAP.md`.
- **Feature flag interaction**: `bench-max-size` feature changes
  `BLOWUP_BITS` from 2 to 1. Profile numbers differ between modes —
  always note which mode you were in.

## See Also

- `PERF_ROADMAP.md` — measured phase data at log_n=20 and 22, plus ranked
  optimization targets with estimated size-of-win per phase.
- `src/cairo_air/prover.rs` — search for `phase_tag` to find the exact
  code locations of each checkpoint.
