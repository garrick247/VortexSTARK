# VortexSTARK Benchmark Artifact

## CHECKPOINT: stwo-cairo End-to-End Sweep, 12 Programs (2026-05-14)

### Commit
```
garrick247/VortexSTARK    main @ f4d2bb6  (PRs #13/#14/#15/#17/#18 landed)
garrick247/stwo-fork      cuda-backend-dev-head @ post-#1  (Column::gather + commitment.decommit batching)
garrick247/stwo-cairo     main @ 4c54e2e5  (prover bumps VortexSTARK rev)
```

### What this benchmark covers
End-to-end Cairo prove + verify on the 12 `test_data/` programs that ship with `stwo-cairo`, using the same harness pattern on both backends (`run_and_prove_cuda` and the newly-added `run_and_prove_iter` for the CPU path). Each program is run with `--iterations 3 --verify`; cold = first iter, warm = third iter. `bench-gpu.sh` evicts any other GPU resident (Ollama / Qwen) before each run so the GPU is uncontested.

All 12 programs `PROOF VALID` on both backends. The CUDA-side proofs are byte-identical with the SimdBackend proofs for the same input.

### Hardware / Toolkit
```
GPU:          NVIDIA GeForce RTX 5090 (32 GB GDDR7, SM 12.0 Blackwell)
Driver:       595.58.03
CUDA:         13.2 (V13.2.78, built 2026-03-19)
Rust:         1.95 (stable, 2026-04-14)
CPU:          Intel Core Ultra 9 285K (24C/24T, max 5.8 GHz)
RAM:          64 GB DDR5
OS:           Ubuntu 25.10 "questing", kernel 6.17.0-23-generic
nvcc flags:   -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
PoW params:   pow_bits=26, log_blowup_factor=1, n_queries=70 (standard Cairo prover config)
```

### Warm-prove speedup vs CpuBackend (SimdBackend with `target-cpu=native` release build)

| Program                                      | CPU warm | CUDA warm | Speedup |
|----------------------------------------------|---------:|----------:|--------:|
| `test_prove_verify_ret_opcode`               |   3.08 s |   0.368 s |  **8.4x** |
| `test_prove_verify_bitwise_builtin`          |   3.12 s |   0.395 s |  **7.9x** |
| `test_prove_verify_range_check_bits_128`     |   3.08 s |   0.400 s |  **7.7x** |
| `test_prove_verify_range_check_bits_96`      |   2.93 s |   0.395 s |  **7.4x** |
| `test_prove_verify_add_mod_builtin`          |   2.88 s |   0.403 s |  **7.2x** |
| `test_prove_verify_poseidon_builtin`         |   3.31 s |   0.482 s |  **6.9x** |
| `test_prove_verify_mul_mod_builtin`          |   2.90 s |   0.445 s |  **6.5x** |
| `test_poseidon_aggregator`                   |   2.83 s |   0.469 s |  **6.0x** |
| `test_prove_verify_all_opcode_components`    |   2.89 s |   0.583 s |  **5.0x** |
| `test_prove_verify_pedersen_builtin`         |   5.37 s |   1.523 s |  **3.5x** |
| `test_pedersen_aggregator`                   |   5.34 s |   1.504 s |  **3.5x** |
| `test_prove_verify_all_builtins`             |   5.64 s |   2.057 s |  **2.7x** |

**Median 6.7x. Range 2.7x – 8.4x.**

The lower end (`all_builtins`, the pedersen pair) corresponds to programs whose constraint mix is dominated by per-component GPU launches and Merkle commit work where each individual kernel is short and launch overhead is a larger fraction of the budget. The high end is programs where a single constraint kind dominates and the GPU kernel runs long enough to fully amortize launch cost.

### Cold-prove comparison

Cold prove includes one-time CUDA driver / runtime initialization, JIT compilation of any newly-encountered kernel variants, and a fresh allocator state. It is **not** a steady-state metric — a long-running proof service amortizes it across many proves.

| Program                                  | CPU cold | CUDA cold |
|------------------------------------------|---------:|----------:|
| `test_prove_verify_ret_opcode`           |   4.24 s |   3.72 s  |
| `test_prove_verify_poseidon_builtin`     |   4.64 s |   ~3.28 s |
| `test_prove_verify_all_builtins`         |   7.01 s |   8.06 s  |
| `test_pedersen_aggregator`               |   6.88 s |   7.51 s  |

CUDA cold-prove can be slightly slower than CPU on the larger programs because of one-time GPU initialization plus first-iteration kernel JIT (~1-2 s of GPU warm-up tax). PR #18 closed ~20% of the gap on `all_builtins` (10.0 s → 8.06 s) by pooling the eval_at_point folding-factors buffer; the rest is inherent to the cold path.

### How we got here today (2026-05-14)

This sweep is the result of a single session's optimization work. Starting baseline (this morning) had `all_builtins` warm at **4.06 s** — a 1.38x outlier where all other programs were already at 3-5x. The bottleneck was identified in two layers:

1. **`Tree decommit (all trees)` = 1.94 s warm (48% of total)** — root cause: `CudaColumn::at(idx)` does one `cudaMemcpy D2H` per element. `stwo`'s `commitment.decommit` called this ~300k times (~150 columns × ~2000 queries plus Merkle layer walks).
2. **Per-call cudaMemcpy overhead in `eval_at_point`** — each call did `DeviceBuffer::from_host` for 88 bytes of folding factors, paying `cudaMalloc` + `cudaFree` × 2215 calls.

Fixes (today's PRs, all merged unless noted):

| Change | Repo / PR | Effect |
|--------|-----------|--------|
| Batched stwo NTT (v1 + v2 + `interpolate_columns` + `quotient_ops::lift`) | VortexSTARK #17 | Architectural cleanup; perf wash because NTT was already not the bottleneck |
| `Column::gather` trait method | stwo-fork #1 | Default impl is per-element `at()` (no behavior change for CPU/SIMD) |
| `commitment.decommit` batched gather (queried-values + Merkle layer walk) | stwo-fork #1 | Refactor to collect positions then issue one `gather()` per column / per layer |
| `CudaColumn::gather` overrides via `cuda_gather_u32/u256_forge` kernels | VortexSTARK #17 | -49% warm prove on all_builtins; -23 to -52% across all 12 |
| Pool `eval_at_point` folding-factors buffer in workspace cache | VortexSTARK #18 | -19% cold prove on all_builtins; warm unchanged within noise |

The pattern these all share: **per-element APIs on top of device memory create O(N) cudaMemcpy syscall load that dominates wall-clock when N is large.** GPU backends implementing single-element traits need to override every "iterate the trait method" call site with a batched variant.

### Scaling notes — speedup is not uniform across the suite

The 6.7× median compresses two distinct regimes. Programs by warm-prove speedup:

| Program | VM steps | CUDA warm | Speedup |
|---|---:|---:|---:|
| `ret_opcode` | 15 | 0.367 s | **8.1×** |
| `range_check_bits_128` | 323 | 0.386 s | **8.1×** |
| `bitwise_builtin` | 673 | 0.394 s | **8.0×** |
| `range_check_bits_96` | 323 | 0.385 s | **7.9×** |
| `add_mod_builtin` | 10423 | 0.400 s | **7.4×** |
| `poseidon_builtin` | 239 | 0.468 s | **7.1×** |
| `mul_mod_builtin` | 10423 | 0.447 s | **6.6×** |
| `poseidon_aggregator` | 6892 | 0.461 s | **6.1×** |
| `all_opcode_components` | 1495 | 0.540 s | **5.3×** |
| `pedersen_builtin` | 205 | 1.442 s | **3.7×** |
| `pedersen_aggregator` | 5872 | 1.468 s | **3.6×** |
| **`all_builtins`** | **9158** | **2.076 s** | **2.7×** |

VM step count does not predict speedup. `add_mod_builtin` at 10423 steps lands at 7.4× while `all_builtins` at 9158 steps lands at 2.7×. The pattern is **component diversity**: programs that exercise one dominant constraint kind amortize the GPU's per-component-launch overhead well; programs that exercise the full builtin set pay per-launch overhead × ~20+ distinct constraint kernels and lose ground.

The pedersen pair is a separate floor at ~3.5×: a single component is exercised, but it's the Pedersen curve op, which is bottlenecked by serial big-int arithmetic on both backends — we do not (yet) have a custom Pedersen GPU kernel, so both backends share roughly the same scalar-multiply work.

**Production implication.** Real Cairo workloads tend to be dominated by one or two heavy components (hash chains, range checks, signature verification). For those, the 6–8× regime is the right expectation. For workloads that look like the kitchen sink (`all_builtins`), expect 2.7–4× until per-component launch overhead is parallelized across CUDA streams.

The current bottleneck for low-speedup programs is **not** any single span — it is the cumulative cost of small per-component kernels in the `Composition` and trace-commit phases. See `next.md` in the development memory for the breakdown.


### Methodology notes for reproducers

```
ssh linux 'cd /home/garrick/stwo-cairo/stwo_cairo_prover &&
  cargo build --release -p stwo-cairo-dev-utils --bin run_and_prove_cuda --features cuda-backend &&
  cargo build --release -p stwo-cairo-dev-utils --bin run_and_prove_iter --features cuda-backend &&
  for prog in test_data/test_*; do
    /home/garrick/bin/bench-gpu.sh ./target/release/run_and_prove_cuda \
      --program $prog/compiled.json --iterations 3 --verify
    /home/garrick/bin/bench-gpu.sh ./target/release/run_and_prove_iter \
      --program $prog/compiled.json --iterations 3 --verify
  done'
```

- The `run_and_prove_iter` binary is the CPU-side mirror of `run_and_prove_cuda` (same iterations flag, same `[summary]` line). It runs `stwo_cairo_prover::prove_cairo::<Blake2sMerkleChannel>` on `SimdBackend` with `target-cpu=native`.
- `bench-gpu.sh` evicts Ollama / Qwen via `keep_alive:0` before each run so the GPU is not contended.
- All 12 programs verify with `PROOF VALID` on both backends. CUDA proofs are byte-identical with the SimdBackend reference for the same input — the gather kernels are pure value reads with no field arithmetic, and the NTT batching uses the same butterfly math as the single-poly path.



## CHECKPOINT: CudaBackend Full-GPU (2026-04-03)

### Commit
```
session 30 — all stwo CudaBackend CPU fallbacks eliminated
```

### What changed

All remaining CPU fallbacks in the stwo `CudaBackend` were replaced with GPU kernels:

| Eliminated fallback | New kernel file |
|---------------------|----------------|
| GKR fix_first_variable (M31→QM31, QM31→QM31) | `cuda/gkr.cu` |
| GKR gen_eq_evals | `cuda/gkr.cu` |
| GKR next_layer (GrandProduct, LogUpGeneric, LogUpMultiplicities, LogUpSingles) | `cuda/gkr.cu` |
| GKR sum_as_poly (all 4 variants) | `cuda/gkr.cu` |
| lift_and_accumulate (QM31 channel-wise) | `cuda/accumulate_lift.cu` |
| QM31 bit-reverse column | `cuda/bit_reverse_wide.cu` |
| pack_leaves_input (4×N → 64×(N/16)) | `cuda/pack_leaves.cu` |
| Blake2s M31-output PoW grind | `cuda/grind_m31_output.cu` |
| Poseidon252 PoW grind | `cuda/grind_poseidon.cu` |
| eval_at_point_by_folding (OODS, large polys) | existing `cuda/fri.cu` kernels |

Formal bounds proofs for the new kernels: `cuda/gkr.fg`, `cuda/accumulate_lift.fg`,
`cuda/bit_reverse_wide.fg`, `cuda/pack_leaves.fg`.

### Hardware / Toolkit
```
GPU:          NVIDIA GeForce RTX 5090 (32 GB GDDR7, SM 12.0 Blackwell)
Driver:       595.79
CUDA:         13.2
Rust:         1.94 (stable)
nvcc flags:   -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
CPU:          Intel Core Ultra 9 285K (24C/24T)
RAM:          64 GB DDR5
OS:           Windows 11
```

### CudaBackend vs CpuBackend (bench-stwo, 2026-04-03)

All GPU results verified correct (match CPU exactly).

```
--- Circle NTT (evaluate + interpolate roundtrip) ---
log_n          CPU (ms)     GPU (ms)    Speedup
------------------------------------------------
14                  0.3          0.2       1.5x
16                  1.4          0.4       4.1x
18                  7.1          0.4      17.9x
20                 34.4          0.5      71.5x
22                202.5          0.5     402.0x
24                861.9         11.3      76.4x

--- Bit Reverse ---
size           CPU (ms)     GPU (ms)    Speedup
------------------------------------------------
2^16                0.1          0.1       2.6x
2^18                0.8          7.5       0.1x    ← kernel overhead > work for tiny batches
2^20                4.8          0.0     153.6x
2^22               35.7          3.1      11.5x
2^24              277.4          0.5     543.0x

--- Poseidon252 Merkle: build_leaves (1 col per leaf) ---
n_leaves           CPU (ms)       GPU (ms)    Speedup
------------------------------------------------------
2^16                  515.9            3.4     151.7x
2^18                 2009.0            8.8     229.2x
2^20                 8053.4           20.0     403.1x
2^22                32176.1           69.6     462.2x
2^24               128915.9          274.8     469.2x

--- Poseidon252 Merkle: build_next_layer ---
n_parents          CPU (ms)       GPU (ms)    Speedup
------------------------------------------------------
2^16                  244.2            1.1     224.9x
2^18                  963.9            8.2     116.9x
2^20                 3841.0           10.7     358.8x
2^22                15229.4           34.8     437.6x
2^24                61798.7          126.2     489.7x
```

---

## CHECKPOINT: Pedersen-37M-Async (2026-03-14)

### Commit
```
4723bcd (dirty — uncommitted async stream + direct download + pinned memory)
```

### Hardware / Toolkit
```
GPU:          NVIDIA GeForce RTX 5090 (32 GB GDDR7, SM 12.0 Blackwell)
Driver:       595.79
CUDA:         13.0 (Build cuda_13.0.r13.0/compiler.36424714_0)
Rust:         1.94 (stable)
nvcc flags:   -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
CPU:          Intel Core Ultra 9 285K (24C/24T)
RAM:          64 GB DDR5
Power cap:    450W (max 600W)
OS:           Windows 11
```

### Pedersen Hash — 37.7M/sec (async stream, zero-alloc pipeline)
```
Architecture
────────────────────────────────────────────────────────────────
Scalar mul:     Windowed 4-bit fixed-base (62 windows per 248-bit scalar)
EC addition:    Mixed affine-Jacobian (11 fp_mont_mul, table Z=R)
EC doubling:    a=1 optimized (9 fp_mont_mul, skip identity mul)
Affine output:  Inline Fermat inverse on GPU (a^(p-2), 444 fp_mont_mul)
Data path:      Zero-copy reinterpret (Fp is repr(C), no flatten/repack)
Transfer:       Async CUDA stream (H2D + kernel + D2H pipelined)
Download:       Direct into result Vec (no intermediate to_host() alloc)
Tables:         L1-cached __device__ global memory (18KB)
Block size:     128 threads/block
Stack:          65536 bytes (cudaDeviceSetLimit)

Batch Benchmark
────────────────────────────────────────────────────────────────
Command: cargo run --release --bin bench_pedersen

Batch       Time        Throughput       Verified
1,000       0.5ms       1,957,330/sec    ✓
10,000      0.6ms       15,489,467/sec   ✓
100,000     2.6ms       38,248,231/sec   ✓
1,000,000   26.5ms      37,724,318/sec   ✓

Pipeline Timing (1M batch, sync instrumented path)
────────────────────────────────────────────────────────────────
Phase                Time      % Total
Flatten (zero-copy)    0.0ms     0.0%
H2D upload             4.6ms    13.6%
Alloc output           0.0ms     0.0%
GPU kernel+sync       18.1ms    53.1%   ← hash + Fermat inverse
D2H download          11.3ms    33.2%   ← async path overlaps w/ alloc
Repack                 0.0ms     0.0%   ← eliminated (direct download)
CPU batch inv          0.0ms     0.0%   ← eliminated (inline on GPU)
────────────────────────────────────────────────────────────────
Total                 34.0ms

Optimization History (this session)
────────────────────────────────────────────────────────────────
Stage                                 Throughput    vs Baseline
Baseline (Montgomery, bit-by-bit)      220,488/s      1x
+ Windowed 4-bit + mixed affine        234,328/s      1.06x
+ Parallel CPU batch inverse         2,762,143/s     12.5x
+ Inline Fermat inverse on GPU       12,915,858/s     58.6x
+ Zero-copy flatten/repack           22,964,169/s    104x
+ Kill to_host() + async stream      37,724,318/s    171x
```
- Correctness: byte-for-byte match against CPU (10K random vector regression, zero mismatches)
- GPU speedup vs CPU (61 hash/sec): 618,431x
- Remaining wall time: 53% GPU kernel (IMAD-bound), 47% PCIe bus
- The bus is the enemy. The hard part is done.

---

## CHECKPOINT: Pedersen-23M (2026-03-14)

### Commit
```
4723bcd (dirty — uncommitted inline Fermat inverse + zero-copy)
```

### Hardware / Toolkit
```
GPU:          NVIDIA GeForce RTX 5090 (32 GB GDDR7, SM 12.0 Blackwell)
Driver:       595.79
CUDA:         13.0 (Build cuda_13.0.r13.0/compiler.36424714_0)
Rust:         1.94 (stable)
nvcc flags:   -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
CPU:          Intel Core Ultra 9 285K (24C/24T)
RAM:          64 GB DDR5
Power cap:    450W (max 600W)
OS:           Windows 11
```

### Pedersen Hash — 23M/sec (GPU-native affine output)
```
Architecture
────────────────────────────────────────────────────────────────
Scalar mul:     Windowed 4-bit fixed-base (62 windows per 248-bit scalar)
EC addition:    Mixed affine-Jacobian (11 fp_mont_mul, table Z=R)
EC doubling:    a=1 optimized (9 fp_mont_mul, skip identity mul)
Affine output:  Inline Fermat inverse on GPU (a^(p-2), 444 fp_mont_mul)
Data path:      Zero-copy reinterpret (Fp is repr(C), no flatten/repack)
Tables:         L1-cached __device__ global memory (18KB)
Block size:     128 threads/block
Stack:          65536 bytes (cudaDeviceSetLimit)

Batch Benchmark
────────────────────────────────────────────────────────────────
Command: cargo run --release --bin bench_pedersen

Batch       Time        Throughput       Verified
1,000       0.5ms       1,960,784/sec    ✓
10,000      0.8ms       12,118,274/sec   ✓
100,000     4.3ms       23,176,045/sec   ✓
1,000,000   43.5ms      22,964,169/sec   ✓

Pipeline Timing (1M batch, 25.6M hash/sec timed run)
────────────────────────────────────────────────────────────────
Phase                Time      % Total
Flatten (zero-copy)    0.0ms     0.0%
H2D upload             4.4ms    11.2%
Alloc output           0.0ms     0.0%
GPU kernel+sync       18.1ms    46.2%   ← hash + Fermat inverse
D2H download           7.0ms    17.9%
Repack (memcpy)        9.6ms    24.6%   ← next target: kill to_host() alloc
CPU batch inv          0.0ms     0.0%   ← eliminated (was 283ms / 81.6%)
────────────────────────────────────────────────────────────────
Total                 39.1ms

Optimization History
────────────────────────────────────────────────────────────────
Stage                                Throughput    Multiplier
Baseline (Montgomery, bit-by-bit)    220,488/s     1x
+ Windowed 4-bit + mixed affine      234,328/s     1.06x
+ Parallel CPU batch inverse        2,762,143/s    12.5x
+ Inline Fermat inverse on GPU     12,915,858/s    58.6x
+ Zero-copy flatten/repack         22,964,169/s    104x
```
- Correctness: byte-for-byte match against CPU (10K random vector regression, zero mismatches)
- GPU speedup vs CPU (61 hash/sec): 376,462x
- GPU power during sustained load: 51-61W (13.3% of 450W cap), 40-58°C, zero throttle events

---

## Frozen Milestone: 2026-03-14

### Commit
```
N/A (pre-push)
```

### Hardware
```
GPU:          NVIDIA GeForce RTX 5090
VRAM:         32 GB GDDR7
SM:           12.0 (Blackwell)
Driver:       595.79
Power limit:  450W (capped, max 600W)
CPU:          Intel Core Ultra 9 285K (24C/24T)
RAM:          64 GB DDR5
OS:           Windows 11
Ambient:      ~22°C (home office)
```

### Toolkit
```
CUDA:         13.0 (Build cuda_13.0.r13.0/compiler.36424714_0)
Rust:         1.94 (stable)
nvcc flags:   -O3 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
```

### Pedersen Hash (GPU, Montgomery EC, Windowed 4-bit Scalar Mul)
```
Config
──────
Block size:       128 threads/block
Batch size:       100,000 hashes
CPU inverse:      Parallel chunked (all cores, std::thread::scope)
Tables:           L1-cached __device__ global memory (18KB)
EC addition:      Mixed affine-Jacobian (11 fp_mont_mul vs 16 standard)
Doubling:         a=1 optimized (9 fp_mont_mul, skip identity mul)
Stack:            65536 bytes (cudaDeviceSetLimit)

Batch Benchmark
──────────────────────────────────────────────────────
Batch       Time        Throughput      Verified
1,000       5.0ms       198,047/sec     ✓
10,000      8.9ms       1,124,923/sec   ✓
100,000     47.1ms      2,124,125/sec   ✓
1,000,000   362.0ms     2,762,143/sec   ✓

15-Minute Stress Test (100K batch, continuous)
──────────────────────────────────────────────────────
Sustained avg:    1,940,652 hashes/sec
Peak interval:    1,983,189 hashes/sec
Min interval:     1,839,854 hashes/sec
Variance (CoV):   1.91%
Total hashed:     1,746,600,000 (1.75 billion)
GPU power:        51-61W sustained (13.3% of 450W cap)
GPU temp:         40-58°C (no throttling)
Throttle events:  0 (zero power/thermal/SW throttle)
GPU speedup:      31,814x vs CPU
```
### Pipeline Timing Breakdown (1M batch, 2,877K hash/sec)
```
Phase                Time      % Total
─────────────────────────────────────────
CPU batch inverse    283.6ms    81.6%    ← dominant bottleneck
Flatten inputs        32.3ms     9.3%    ← CPU memcpy/reformat
D2H download          10.0ms     2.9%    ← PCIe gen5
GPU kernel+sync        9.4ms     2.7%    ← actual EC math (106M/sec raw)
Repack Fp vecs         7.7ms     2.2%    ← CPU reformat
H2D upload             4.5ms     1.3%    ← PCIe gen5
Alloc output           0.0ms     0.0%
─────────────────────────────────────────
Compute (kern+inv):  292.9ms    84.3%
Overhead (rest):      54.6ms    15.7%
```
- Next target: GPU batch inverse for Stark252 (eliminate 81.6% CPU bottleneck)
- Command: `cargo run --release --bin bench_pedersen` / `cargo run --release --bin stress_test`
- Correctness: byte-for-byte match against CPU (10K random vector regression, zero mismatches)
- Previous baseline: 220,488/sec (Montgomery, no windowing, single-core batch inverse)
- Improvement: 8.8x sustained throughput (parallel CPU) + 6.3% GPU kernel speedup (windowing)

---

## Frozen Milestone: 2026-03-15

### Commit
```
0413d3552bd3798d03d840819fc242756fc83419
```

### Hardware
```
GPU:          NVIDIA GeForce RTX 5090
VRAM:         32 GB GDDR7
SM:           12.0 (Blackwell)
Driver:       595.79
Power limit:  450W (capped via scheduled task)
CPU:          Intel Core Ultra 9 285K
RAM:          64 GB DDR5
OS:           Windows 11
```

### Toolkit
```
CUDA:         13.0 (Build cuda_13.0.r13.0/compiler.36424714_0)
Rust:         1.94 (stable)
nvcc flags:   -O3 -gencode arch=compute_120,code=sm_120
```

### Pedersen Hash (GPU, Montgomery EC, Projective Jacobian)
```
Batch       Time        Throughput      Verified
─────────────────────────────────────────────────
1,000       10.9ms      92,095/sec      ✓
10,000      49.4ms      202,353/sec     ✓
100,000     456.5ms     219,068/sec     ✓
1,000,000   4,535.4ms   220,488/sec     ✓
```
- Correctness: byte-for-byte match against CPU implementation
- Regression: 10,000 random vector test (deterministic seeds, zero mismatches)
- CPU baseline: 60 hashes/sec (projective Jacobian, Fermat inverse)
- GPU speedup: 3,675x

### Fibonacci STARK (proven + verified, 100-bit security)
```
log_n   Elements        Prove       Verify
─────────────────────────────────────────────
20      1,048,576       231ms       4.6ms   ✓
24      16,777,216      390ms       6.4ms   ✓
28      268,435,456     2,678ms     11.9ms  ✓
29      536,870,912     4,388ms     9.5ms   ✓
30      1,073,741,824   9,100ms     14.2ms  ✓
```

### Poseidon Hash (GPU trace gen + NTT, 8 columns, x^5 S-box)
```
log_n   Hashes          Time
─────────────────────────────
20      47,662          4.9ms
24      762,600         73ms
28      12,201,611      1,445ms
```
- GPU trace generation: 34.7M hashes/sec

### Cairo VM STARK (31 columns, 31 constraints, LogUp + range checks)
```
log_n   Steps           Elements        Time
──────────────────────────────────────────────
20      1,048,576       28,311,552      59ms
24      16,777,216      452,984,832     818ms
26      67,108,864      1,811,939,328   3,546ms
```

### GPU Kernel Profile (Pedersen)
```
Registers/thread:   210 (SM 12.0), 202 (SM 8.9)
Stack spill:        64 bytes
Shared memory:      0
Occupancy:          ~15% (register-limited)
Bottleneck:         Compute-bound (INT64 multiply throughput)
                    ~322K u64 multiplies per hash
                    ~71B u64 muls/sec (80% of theoretical peak)
Register cap test:  128 regs → 221K/sec (same), 96 regs → 221K/sec (same)
                    Occupancy increase offset by spill latency.
                    Confirmed: pure IMAD throughput bottleneck.
```

### Test Suite
```
149 tests, all passing
Includes: 10K GPU vs CPU Pedersen regression test
```

---

## CHECKPOINT: Full System Benchmark — Clean RTX 5090 (2026-03-21)

### Hardware / Toolkit
```
GPU:          NVIDIA GeForce RTX 5090 (32607 MiB GDDR7, SM 12.0 Blackwell)
Driver:       595.79
CUDA:         13.2
Rust:         stable
CPU:          Intel Core Ultra 9 285K
RAM:          64 GB DDR5
OS:           Windows 11
VRAM at start: 0 MB (clean system)
```

### Fibonacci STARK (1 column, degree-1 constraint)
```
log_n=20 |      1,048,576 elements | prove:   107.6ms | verify:  4.6ms | ✓
log_n=24 |     16,777,216 elements | prove:   211.9ms | verify:  6.4ms | ✓
log_n=28 |    268,435,456 elements | prove:  1545.4ms | verify:  8.4ms | ✓
log_n=29 |    536,870,912 elements | prove:  3671.4ms | verify:  8.9ms | ✓
log_n=30 |  1,073,741,824 elements | prove:  9468.2ms | verify:  9.4ms | ✓
```

### Cairo VM STARK (31 columns, 31 constraints, LogUp + range checks)
Full end-to-end cairo_prove() + cairo_verify() — not raw kernel timing.
```
log_n=20 |      1,048,576 steps | prove:  1007.4ms | verify:  0.4ms | ✓
log_n=24 |     16,777,216 steps | prove:  7988.4ms | verify:  0.5ms | ✓
log_n=26 |     67,108,864 steps | prove: 31967.1ms | verify:  0.5ms | ✓
```

### Poseidon Trace+NTT Throughput (8 columns, degree-5 S-box)
```
log_n=20 |     47,662 hashes | trace:   1ms | NTT:    3ms | total:   4.3ms | 11.0M hash/s
log_n=24 |    762,600 hashes | trace:  25ms | NTT:   83ms | total: 115.2ms |  6.6M hash/s
log_n=28 | 12,201,611 hashes | trace: 414ms | NTT: 1222ms | total:  1798ms |  6.8M hash/s
```

### Pedersen Hash (CPU, STARK curve EC)
```
100 hashes: 1662ms (16.6ms/hash, 60 hashes/sec)
```

### Bug Fixed This Session
```
cudaDeviceSetLimit(cudaLimitStackSize, 65536) removed from pedersen_gpu.cu.
Was pre-allocating ~21 GB of stack space on RTX 5090 (170 SMs × 2048 threads × 64KB).
Actual SM_120 kernel stack usage: 32 bytes (trace), 32 bytes (batch), 112 bytes (ec_trace).
Fix: removed all 4 calls — CUDA default (1KB) is sufficient.
```

### Test Suite
```
149 tests, all passing (single-threaded)
```

---

## CHECKPOINT: Poseidon2 Migration (2026-03-21)

### Changes
- Migrated from Poseidon (22 full rounds) → Poseidon2 (8 full + 22 partial = 30 total rounds)
- S-box count per permutation: 176 → 86 (51% fewer)
- Linear layers: M_E = circ(3,1,1,1,1,1,1,1) for full rounds; M_I = circ(2,1,1,1,1,1,1,1) for partial
- Rust nightly toolchain: nightly-2025-06-23 → nightly (2026-03-20 build, rustc 1.96.0-nightly)
- NUM_ROUNDS constant: 22 → 30
- Round constants: 176 values → 86 values (64 full + 22 partial)

### Hardware
```
GPU:    NVIDIA GeForce RTX 5090 (32607 MB GDDR7, SM 12.0)
Driver: 595.79 / CUDA 13.2
CPU:    Intel Core Ultra 9 285K, 64 GB DDR5
OS:     Windows 11 Build 26200
```

### Fibonacci STARK (unchanged)
```
log_n=20 |      1,048,576 elements | prove:   109.6ms | verify:  4.6ms | ✓
log_n=24 |     16,777,216 elements | prove:   211.6ms | verify:  6.3ms | ✓
log_n=28 |    268,435,456 elements | prove:  1547.3ms | verify:  8.2ms | ✓
log_n=29 |    536,870,912 elements | prove:  3614.9ms | verify:  8.7ms | ✓
log_n=30 |  1,073,741,824 elements | prove: 10591.3ms | verify:  9.3ms | ✓
```

### Cairo VM STARK (31 columns, 31 constraints — unchanged)
```
log_n=20 |      1,048,576 steps | prove:   626.9ms | verify:  0.4ms | ✓
log_n=24 |     16,777,216 steps | prove:  7259.2ms | verify:  0.4ms | ✓
log_n=26 |     67,108,864 steps | prove: 29592.5ms | verify:  0.5ms | ✓
```

### Poseidon2 Trace+NTT Throughput (8 columns, degree-5 S-box)
```
log_n=20 |     34,952 hashes | trace:   1ms | NTT:   4ms | total:   6.7ms |  5.2M hash/s
log_n=24 |    559,240 hashes | trace:  28ms | NTT:  62ms | total:  96.4ms |  5.8M hash/s
log_n=28 |  8,947,848 hashes | trace: 487ms | NTT: 1218ms | total: 1864.8ms | 4.8M hash/s
```

#### Comparison vs old Poseidon (22 full rounds, same trace size)
```
                    Poseidon (old)   Poseidon2 (new)   Ratio
Rounds/permutation:     22               30            1.36×
S-boxes/permutation:   176               86            0.49×
Hashes at log_n=24:    762,600          559,240        0.73×  (=22/30)
Hash/s at log_n=24:     6.6M             5.8M          0.88×
Hash/s at log_n=28:     6.8M             4.8M          0.71×
```
Note: Poseidon2 hash/s is lower because the trace size is fixed (2^log_n rows) but each
permutation uses 30 rows instead of 22 → fewer hashes per trace. The 22/30 = 0.73 ratio
matches exactly. The absolute NTT time improved slightly due to newer Rust nightly (1.96.0).

The benefit of Poseidon2 is standardization (Plonky3/Stwo ecosystem alignment) and
stronger algebraic security, not STARK proving throughput.

### Test Suite
```
150 tests (149 pass in parallel; 1 flaky GPU-pool test passes when run in isolation)
```

---

## CHECKPOINT: RPO-M31 (2026-03-22)

### Summary
Implemented RPO-M31 — a Circle STARK–native hash function from eprint 2024/1635 (Ashur & Tariq).
New files: `src/rpo_m31.rs`, `cuda/rpo_trace.cu`.

### Why RPO-M31
- Designed specifically for Circle STARKs over M31 (unlike Poseidon2 which is general-purpose)
- 14 rows/permutation vs Poseidon2's 30 → 2.14× more permutations per trace
- State width 24 (rate 16, capacity 8) vs Poseidon2's 8 (rate 4, capacity 4)
- Balanced FM/BM round structure: both forward (x^5) and inverse (x^(1/5)) S-boxes
- Round constants: SHAKE-256 derived, 360 total (15 steps × 24 elements)
- MDS: 24×24 circulant from 32-element root-of-unity construction (Appendix A.3)

### Architecture
```
Per round (7 total):
  FM: MDS → add RC → x^5       → write trace row 2r
  BM: MDS → add RC → x^(1/5)  → write trace row 2r+1
CLS: MDS → add RC              (final state, no trace row)

Total: 14 trace rows, 15 constant sets (360 u32 values)
MDS: 24×24 = 576 u32 values, uploaded to CUDA __constant__ memory
x^(1/5) = x^1717986917, implemented as square-and-multiply (~45 mults)
```

### Performance (RTX 5090, CUDA 13.2, 2026-03-22, updated)
```
log_n=20 |    74,898 hashes | trace:  101ms | NTT:    9ms | total:  112ms |  0.7M hash/s
log_n=24 | 1,198,372 hashes | trace:   82ms | NTT:  162ms | total:  250ms |  4.8M hash/s
log_n=28 |19,173,961 hashes | trace: 1404ms | NTT: 3867ms | total: 5507ms |  3.5M hash/s
```

### Comparison vs Poseidon2 (same trace size)
```
Metric                  | Poseidon2         | RPO-M31
─────────────────────────────────────────────────────────
Rows per permutation    | 30                | 14  (2.14x fewer)
State width             | 8                 | 24  (3x wider)
Hashes at log_n=24      | ~560K/perm        | ~1.2M/perm  (+2.14x)
Trace gen time (log_n=24)| ~27ms           | ~82ms
NTT cost (log_n=24)     | ~43ms (8 cols)    | ~162ms (24 cols)
Total at log_n=24       | 7.4M hash/s       | 4.8M hash/s
Total at log_n=28       | 4.7M hash/s       | 3.5M hash/s
S-box operations        | 86/perm (x^5 only)| 14*24 fwd + 14*24 inv/perm
```

### Memory Note
RPO-M31's 24 columns accumulate significant NTT cost. The benchmark NTT loop now drops
each eval column immediately after transform (instead of accumulating all 24), keeping
peak VRAM at trace_all (24 GB at log_n=28) + one eval col (2 GB) = ~26 GB — well within
32 GB. This was fixed in commit 5e67606; earlier builds would OOM at log_n=28.

### Correctness
- 7 CPU unit tests pass (permutation, S-box roundtrip, MDS, zero input)
- GPU trace correctness test passes: CUDA output matches CPU byte-for-byte for all 14 rows × 24 cols
- Round constants pre-verified against AbdelStark/rpo-xhash-m31 reference (MIT)

---

## CHECKPOINT: Full System — OOM Fix + gpu_bench (2026-03-22)

### Hardware / Toolkit
```
GPU:           NVIDIA GeForce RTX 5090 (32607 MiB GDDR7, SM 12.0 Blackwell)
Driver:        595.79
CUDA:          13.2
Rust:          stable
CPU:           Intel Core Ultra 9 285K (24C/24T)
RAM:           64 GB DDR5
OS:            Windows 11 Build 26200
Power cap:     450W (max 600W)
VRAM at start: 0 MB (clean)
```

### Bug Fixed
NTT benchmark loops were accumulating all eval columns simultaneously before dropping them.
RPO-M31 (24 cols) at log_n=28 produced 48 GB of live eval buffers, exceeding 32 GB VRAM.
Fix: drop each eval column immediately after its NTT. Peak VRAM for RPO log_n=28 drops
from ~72 GB to ~26 GB. All four benchmark binaries patched (full_benchmark, hash_bench,
rpo_bench, gpu_bench).

### Fibonacci STARK (1 column, degree-1 constraint)
```
log_n=20 |      1,048,576 elements | prove:   108.9ms | verify:  4.6ms | ✓
log_n=24 |     16,777,216 elements | prove:   214.3ms | verify:  6.2ms | ✓
log_n=28 |    268,435,456 elements | prove:  1556.1ms | verify:  8.2ms | ✓
log_n=29 |    536,870,912 elements | prove:  3564.0ms | verify:  8.8ms | ✓
log_n=30 |  1,073,741,824 elements | prove:  9786.9ms | verify: 10.7ms | ✓
```

### Poseidon2 Trace+NTT (8 cols, 30 rows/perm, RF=8 RP=22)
```
log_n=20 |        34,952 hashes | trace:   3ms | NTT:    3ms | total:    6.0ms |  5.8M hash/s
log_n=24 |       559,240 hashes | trace:  27ms | NTT:   43ms | total:   75.5ms |  7.4M hash/s
log_n=28 |     8,947,848 hashes | trace: 473ms | NTT: 1290ms | total: 1921.6ms |  4.7M hash/s
```

### Cairo VM STARK (31 cols, 31 constraints, LogUp + range checks)
```
log_n=20 |      1,048,576 steps | prove:    581.4ms | verify:  0.4ms | ✓
log_n=24 |     16,777,216 steps | prove:   7237.7ms | verify:  0.4ms | ✓
log_n=26 |     67,108,864 steps | prove:  29199.7ms | verify:  0.5ms | ✓
```

### RPO-M31 Trace+NTT (24 cols, 14 rows/perm)
```
log_n=20 |        74,898 hashes | trace:  101ms | NTT:    9ms | total:   112.0ms |  0.7M hash/s
log_n=24 |     1,198,372 hashes | trace:   82ms | NTT:  162ms | total:   250.3ms |  4.8M hash/s
log_n=28 |    19,173,961 hashes | trace: 1404ms | NTT: 3867ms | total:  5507.1ms |  3.5M hash/s
```

### Poseidon2-Full Trace+NTT [EXPERIMENTAL — no security analysis] (8 cols, 8 rows/perm, RF=8 RP=0)
```
log_n=20 |       131,072 hashes | trace:  23ms | NTT:    3ms | total:    27.0ms |  4.9M hash/s
log_n=24 |     2,097,152 hashes | trace:  26ms | NTT:   43ms | total:    71.9ms | 29.2M hash/s
log_n=28 |    33,554,432 hashes | trace: 475ms | NTT: 1288ms | total:  1916.2ms | 17.5M hash/s
```

### Pedersen Hash (GPU, windowed 4-bit scalar mul, STARK curve)
```
      1,000 hashes:   0.5ms   1,840,265 hash/s
     10,000 hashes:   0.7ms  15,330,369 hash/s
    100,000 hashes:   2.6ms  38,072,032 hash/s
  1,000,000 hashes:  26.6ms  37,658,826 hash/s
```

### New Tool: gpu_bench
Comprehensive benchmark binary with strict pre-flight GPU condition checks:
- Detects competing compute processes (aborts unless --force)
- Queries `nvidia-smi -q -d POWER` for draw, current limit, enforced limit, default, min, max
- Queries `nvidia-smi -q -d MEMORY` for FB total/used/free
- Checks temperature (warn >=70C, abort >=83C) and SM clock at idle
- Background sampler records per-section peak temp/power/VRAM/clock during benchmarks
- Correct VRAM estimates: (trace_cols + one_eval + 4 scratch) x n x 4 bytes


## CHECKPOINT: Cairo Prove Phase Profiling (2026-04-22)

Env-gated phase timing added to `cairo_prove_cached_with_columns`. Set
`VORTEXSTARK_PROFILE=1` to emit per-phase wall-clock + cumulative times.
Zero cost when off.

### log_n=20 — 8.9s total, fibonacci.casm padded to 2^20

| Phase                  | Time     | %      |
|------------------------|---------:|-------:|
| oods                   | 2808.6ms | 31.5%  |
| ntt_blind_commit       | 2548.5ms | 28.6%  |
| phase2_logup_rc        | 1788.5ms | 20.1%  |
| phase3_quotient        |  794.6ms |  8.9%  |
| phase4_fri             |  388.8ms |  4.4%  |
| phase5_pow_decommit    |  234.5ms |  2.6%  |
| sdict_interaction      |  189.1ms |  2.1%  |

**Top-3 = 80.2% of prove time.**

### log_n=22 — 45.7s total, fibonacci.casm padded to 2^22

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

Scaling 8.9s→45.7s over 4× data = **5.1×**, consistent with `n log n`.
Proportions stable across sizes — the optimization targets in
`PERF_ROADMAP.md` are robust.

### log_n=30 (1.07B elements) — `bench-max-size` feature

```
Fibonacci BLOWUP_BITS=1
  prove:      19683.5ms
  verify:         7.6ms
  proof size:    2.7 MB
  verified:      YES
```

First measurement of the headline since the BLOWUP_BITS=2 rework (stwo
FriVerifier compatibility). Security dropped to 80-bit via the
`bench-max-size` cargo feature — not for production, kept as a capability
demonstration of the GPU scaling envelope.

## CHECKPOINT: Post-perf-push Measurements (2026-04-22 late)

15 consecutive perf commits merged onto main. Same log_n=22 workload remeasured
on same host; binary rebuilt from `147dd7f`.

### log_n=22 — 13.25s total (was 45.7s, −71.0%)

| Phase                  | Before    | After    | Δ       |
|------------------------|----------:|---------:|--------:|
| ntt_blind_commit       | 15107.2ms |  3831.1ms | −74.6%  |
| oods                   | 11172.4ms |  2887.8ms | −74.2%  |
| phase2_logup_rc        | 11149.7ms |  3398.2ms | −69.5%  |
| phase3_quotient        |  3615.5ms |   854.9ms | −76.4%  |
| phase4_fri             |  1554.2ms |   288.3ms | −81.5%  |
| sdict_interaction      |  1512.7ms |   480.1ms | −68.3%  |
| phase5_pow_decommit    |   905.1ms |   804.6ms | −11.1%  |

**Every phase faster. All lib tests still pass (now 394/394 passing + 2 microbench `#[ignore]`s; verified 2026-04-26). Proofs verify OK.**

Dominant optimization pattern: single-threaded CPU loops building domain-point
lookups or doing per-row QM31 arithmetic were starving the GPU. Switching to
rayon par_iter with index-independent `coset.at(i)` (instead of sequential
`pt = pt.mul(step)`) scales to all 24 cores. Secondary pattern: eliminate
D→H→CPU→H→D round-trips by doing permutes on GPU via
`cuda_permute_hc_to_canonic_brt` / `cuda_permute_canonic_brt_to_hc_natural`.
See `PERF_ROADMAP.md` for commit-by-commit attribution.

## CHECKPOINT: FORGE feature surface end-to-end (2026-04-27)

Three-way comparison after defaulting `forge-ntt + forge-ntt-batch + forge-fri`
on (commit `c5c4fff`).  `forge_bench` cairo_prove (Fibonacci) at log_n=18,
3 timed iters post 2 warmup, RTX 5090 / driver 595.79 / CUDA 13.2 / WSL2.

| Config | Mean | Stddev | Δ vs hand-written |
|---|---:|---:|---:|
| `--no-default-features` (hand-written kernels) | 0.6071s | ±0.019s | — |
| default (forge-ntt + forge-ntt-batch + forge-fri) | 0.6233s | ±0.012s | +2.7% (within 1σ) |
| default + forge-bit-reverse + forge-gather + forge-blake2s | 0.6077s | ±0.007s | +0.1% (statistical tie) |

The 5× / 7.8× microbench wins documented for the NTT and FRI kernels do **not**
show up end-to-end at log_n=18 — the NTT layer is sub-1% of total prover
wallclock at this size, so Amdahl-bound. WSL2's sync-cudaMalloc fallback (the
pool API isn't supported on WSL2) also caps any pool-async benefit.

**Honest read: the FORGE-emitted variants ship formal verification at parity
end-to-end wallclock at log_n=18.**  The microbench wins should translate to
end-to-end speedup at log_n≥24 where NTT/FRI dominate the phase budget; the
clean measurement environment for that is the Linux native CI runner with
pool-async allocations enabled.

Raw bench data: `BENCH_2026-04-27_log_n18.txt`.
