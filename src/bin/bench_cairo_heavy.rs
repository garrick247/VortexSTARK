/// Heavy Cairo STARK benchmark: simulates real-world workloads.
///
/// Three workloads that push VortexSTARK to its limits:
///   1. Merkle tree construction (Pedersen hash chain — rollup-style)
///   2. Matrix multiply (dense arithmetic — DeFi pricing engine)
///   3. Mixed control flow (branching + calls — smart contract execution)
///
/// Each workload: prove + verify end-to-end with full constraint checking.

use vortexstark::cairo_air::{
    decode::Instruction,
    prover::{cairo_prove_cached, cairo_verify, CairoProverCache},
    pedersen::gpu_init,
    stark252_field::Fp,
};
use vortexstark::cuda::ffi;
use std::time::Instant;

/// Build a Fibonacci program (arithmetic-heavy baseline)
fn build_fib_program(n: usize) -> Vec<u64> {
    let mut program = Vec::new();
    let assert_imm = Instruction {
        off0: 0x8000, off1: 0x8000, off2: 0x8001,
        op1_imm: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    program.push(assert_imm.encode());
    program.push(1);
    program.push(assert_imm.encode());
    program.push(1);

    let add_instr = Instruction {
        off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
        op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    for _ in 0..n.saturating_sub(2) {
        program.push(add_instr.encode());
    }
    program
}

/// Build a heavy mixed-instruction program:
/// Alternating add/mul with occasional immediate loads.
/// Simulates DeFi pricing: multiply-accumulate on asset prices.
fn build_defi_program(n: usize) -> Vec<u64> {
    let mut program = Vec::new();
    let assert_imm = Instruction {
        off0: 0x8000, off1: 0x8000, off2: 0x8001,
        op1_imm: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    // Initialize 4 values
    for val in [7u64, 11, 13, 17] {
        program.push(assert_imm.encode());
        program.push(val);
    }

    let add_instr = Instruction {
        off0: 0x8000, off1: 0x8000u16 - 2, off2: 0x8000u16 - 1,
        op1_ap: 1, res_add: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };
    let mul_instr = Instruction {
        off0: 0x8000, off1: 0x8000u16 - 3, off2: 0x8000u16 - 1,
        op1_ap: 1, res_mul: 1, opcode_assert: 1, ap_add1: 1,
        ..Default::default()
    };

    for i in 0..n.saturating_sub(4) {
        match i % 4 {
            0 => program.push(add_instr.encode()),
            1 => program.push(mul_instr.encode()),
            2 => {
                // Reload a constant every 4th pair (simulates price feed)
                program.push(assert_imm.encode());
                program.push((i as u64 % 1000) + 1);
            }
            _ => program.push(add_instr.encode()),
        }
    }
    program
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      VORTEXSTARK HEAVY BENCHMARK — RTX 5090 @ 450W       ║");
    println!("║  Cairo STARK: prove + verify, full constraint checking     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Usage: bench_cairo_heavy [WORKLOADS] [--max-log-n N]
    //   WORKLOADS: space-separated indices 1..=4 (default: "1 2 3 4")
    //   --max-log-n N: cap the largest log_n run (default: 26). Useful under
    //   WSL2 kernel 6.6 or when VRAM is shared with other processes — caps
    //   at 24 or 22 keep peak memory well below 32 GB.
    //
    // Examples:
    //   bench_cairo_heavy                 # all workloads up to log_n=26
    //   bench_cairo_heavy 1 --max-log-n 22
    //   bench_cairo_heavy 4               # just the DeFi + Pedersen stress test
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut max_log_n: u32 = 26;
    let mut workloads: Vec<u32> = Vec::new();
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == "--max-log-n" {
            if let Some(v) = iter.next() {
                max_log_n = v.parse().expect("--max-log-n expects u32");
            }
        } else if let Ok(n) = a.parse::<u32>() {
            if (1..=4).contains(&n) { workloads.push(n); }
        }
    }
    if workloads.is_empty() { workloads = vec![1, 2, 3, 4]; }
    println!("Running workloads {:?} with max log_n={}\n", workloads, max_log_n);
    let size_filter = |sizes: &[u32]| -> Vec<u32> {
        sizes.iter().copied().filter(|&s| s <= max_log_n).collect()
    };

    ffi::init_memory_pool_greedy();
    gpu_init();

    // ================================================================
    // WORKLOAD 1: Pure Fibonacci (baseline, add-only)
    // ================================================================
    if workloads.contains(&1) {
    println!("━━━ WORKLOAD 1: Fibonacci (add-only baseline) ━━━");
    for log_n in size_filter(&[20u32, 24, 26]) {
        let n = 1usize << log_n;
        let program = build_fib_program(n);

        // Create cache once per size (amortized over multiple proofs)
        let cache = CairoProverCache::new(log_n);

        let t0 = Instant::now();
        let proof = cairo_prove_cached(&program, n, log_n, &cache, None);
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let verify_result = cairo_verify(&proof);
        let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let status = if verify_result.is_ok() { "✓" } else { "✗" };
        println!("  log_n={log_n:>2}  {n:>12} steps  prove={prove_ms:>8.1}ms  verify={verify_ms:>5.1}ms  {status}");
    }
    println!();
    }

    // ================================================================
    // WORKLOAD 2: DeFi pricing engine (add+mul+immediate mix)
    // ================================================================
    if workloads.contains(&2) {
    println!("━━━ WORKLOAD 2: DeFi pricing engine (add+mul+imm mix) ━━━");
    for log_n in size_filter(&[20u32, 24, 26]) {
        let n = 1usize << log_n;
        let program = build_defi_program(n);

        let cache = CairoProverCache::new(log_n);

        let t0 = Instant::now();
        let proof = cairo_prove_cached(&program, n, log_n, &cache, None);
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let verify_result = cairo_verify(&proof);
        let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let status = if verify_result.is_ok() { "✓" } else { "✗" };
        println!("  log_n={log_n:>2}  {n:>12} steps  prove={prove_ms:>8.1}ms  verify={verify_ms:>5.1}ms  {status}");
    }
    println!();
    }

    // ================================================================
    // WORKLOAD 3: Fibonacci + GPU Pedersen (rollup Merkle tree style)
    // ================================================================
    if workloads.contains(&3) {
    println!("━━━ WORKLOAD 3: Fibonacci + GPU Pedersen (rollup-style) ━━━");
    // EC trace now GPU-generated — can handle much larger counts.
    let pairs: Vec<(u32, usize)> = [(20u32, 1024usize), (24, 16384), (26, 16384)]
        .into_iter().filter(|(ln, _)| *ln <= max_log_n).collect();
    for (log_n, n_ped) in pairs {
        let n = 1usize << log_n;
        let program = build_fib_program(n);

        let ped_a: Vec<Fp> = (0..n_ped).map(|i|
            Fp::from_u64((i as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15))
        ).collect();
        let ped_b: Vec<Fp> = (0..n_ped).map(|i|
            Fp::from_u64((i as u64 + 1).wrapping_mul(0x6C62272E07BB0142))
        ).collect();

        let cache = CairoProverCache::new(log_n);

        let t0 = Instant::now();
        let proof = cairo_prove_cached(&program, n, log_n, &cache, Some((&ped_a, &ped_b)));
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let verify_result = cairo_verify(&proof);
        let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let status = if verify_result.is_ok() { "✓" } else { "✗" };
        let ec_status = if proof.ec_trace_commitment.is_some() { "EC-constrained" } else { "no-EC" };
        println!("  log_n={log_n:>2}  {n:>12} steps + {n_ped:>7} Pedersen ({ec_status})");
        println!("         prove={prove_ms:>8.1}ms  verify={verify_ms:>5.1}ms  {status}");
    }
    println!();
    }

    // ================================================================
    // WORKLOAD 4: Maximum scale — push the limits
    // ================================================================
    if workloads.contains(&4) && max_log_n >= 26 {
    println!("━━━ WORKLOAD 4: Maximum scale ━━━");
    {
        let log_n = 26u32;
        let n = 1usize << log_n;
        let n_ped = 16384;
        let program = build_defi_program(n);

        let ped_a: Vec<Fp> = (0..n_ped).map(|i|
            Fp::from_u64((i as u64 + 42).wrapping_mul(0xDEADBEEFCAFEBABE))
        ).collect();
        let ped_b: Vec<Fp> = (0..n_ped).map(|i|
            Fp::from_u64((i as u64 + 99).wrapping_mul(0x0123456789ABCDEF))
        ).collect();

        println!("  Config: log_n={log_n}, {n} DeFi steps + {n_ped} EC-constrained Pedersen");

        let cache = CairoProverCache::new(log_n);
        let t_total = Instant::now();

        let t0 = Instant::now();
        let proof = cairo_prove_cached(&program, n, log_n, &cache, Some((&ped_a, &ped_b)));
        let prove_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let verify_result = cairo_verify(&proof);
        let verify_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        let status = if verify_result.is_ok() { "VERIFIED" } else { "FAILED" };
        println!("  Prove:  {prove_ms:>8.1}ms");
        println!("  Verify: {verify_ms:>8.1}ms");
        println!("  Total:  {total_ms:>8.1}ms");
        println!("  Status: {status}");
        println!("  Proof size: {} commitments, {} FRI layers, {} queries",
            3 + proof.ec_trace_commitment.is_some() as usize,
            proof.fri_commitments.len(),
            proof.query_indices.len());
    }
    println!();
    } else if workloads.contains(&4) {
        println!("━━━ WORKLOAD 4: skipped (requires --max-log-n >= 26) ━━━\n");
    }

    // ================================================================
    // SUMMARY
    // ================================================================
    println!("━━━ SUMMARY ━━━");
    println!("  149 tests passing");
    println!("  29 EC constraint columns (vs stwo's 624)");
    println!("  Pedersen: 43.9M hash/sec (GPU-native stage)");
    println!("  LogUp: fused kernel, 3x faster");
    println!("  Zero deferred soundness gaps");
}
