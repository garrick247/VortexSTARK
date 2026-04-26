//! End-to-end prover wallclock without `#[cfg(test)]` debug overhead.
//! Used to compare baseline vs FORGE features honestly.

use vortexstark::cairo_air::decode::Instruction;
use vortexstark::cairo_air::prover::{cairo_prove, cairo_verify};
use vortexstark::cuda::ffi;
use std::time::Instant;

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

fn main() {
    let log_n: u32 = std::env::var("LOG_N").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(18);
    let iters: u32 = std::env::var("ITERS").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let n = 1usize << log_n;

    ffi::init_memory_pool();
    let program = build_fib_program(n);

    let features = if cfg!(all(feature = "forge-blake2s",
                                feature = "forge-ntt",
                                feature = "forge-fri")) {
        "[ALL FORGE]"
    } else if cfg!(any(feature = "forge-blake2s",
                        feature = "forge-ntt",
                        feature = "forge-fri")) {
        let mut parts = Vec::new();
        if cfg!(feature = "forge-blake2s") { parts.push("blake2s"); }
        if cfg!(feature = "forge-ntt")    { parts.push("ntt"); }
        if cfg!(feature = "forge-fri")    { parts.push("fri"); }
        Box::leak(format!("[FORGE: {}]", parts.join("+")).into_boxed_str())
    } else {
        "[NO FORGE]"
    };

    // Warmup
    let _ = cairo_prove(&program, n, log_n);
    let _ = cairo_prove(&program, n, log_n);

    let mut times: Vec<f64> = Vec::with_capacity(iters as usize);
    for i in 0..iters {
        let t0 = Instant::now();
        let proof = cairo_prove(&program, n, log_n);
        let elapsed = t0.elapsed().as_secs_f64();
        times.push(elapsed);
        let ok = cairo_verify(&proof).is_ok();
        eprintln!("  iter {i}: {:.4}s  verify={}", elapsed, ok);
        assert!(ok, "iter {i} verify failed");
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let stddev = (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
                  / times.len() as f64).sqrt();

    println!();
    println!("==== forge_bench cairo_prove @ log_n={log_n} {features} ====");
    println!("  mean: {:.4}s ± {:.4}s ({} iters, post-2-warmup)", mean, stddev, iters);
}
