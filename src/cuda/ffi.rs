//! Raw FFI bindings to CUDA runtime and VortexSTARK kernels.

use std::ffi::c_void;

// CUDA runtime
unsafe extern "C" {
    pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFree(dev_ptr: *mut c_void) -> i32;
    pub fn cudaMallocAsync(dev_ptr: *mut *mut c_void, size: usize, stream: *mut c_void) -> i32;
    pub fn cudaFreeAsync(dev_ptr: *mut c_void, stream: *mut c_void) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, count: usize, kind: i32, stream: *mut c_void) -> i32;
    pub fn cudaMemset(dev_ptr: *mut c_void, value: i32, count: usize) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaGetLastError() -> i32;
    pub fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn cudaFreeHost(ptr: *mut c_void) -> i32;
    pub fn cudaDeviceGetDefaultMemPool(pool: *mut *mut c_void, device: i32) -> i32;
    pub fn cudaMemPoolSetAttribute(pool: *mut c_void, attr: i32, value: *const c_void) -> i32;
    pub fn cudaMemPoolTrimTo(pool: *mut c_void, min_bytes_to_keep: usize) -> i32;
}

// CUDA streams
unsafe extern "C" {
    pub fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
}

/// RAII wrapper for a CUDA stream.
pub struct CudaStream {
    pub ptr: *mut c_void,
}

impl CudaStream {
    /// Create a new CUDA stream, panicking on failure.
    pub fn new() -> Self {
        Self::try_new().unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create a new CUDA stream, returning an error if CUDA fails (e.g. OOM or no device).
    pub fn try_new() -> Result<Self, String> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamCreate(&mut ptr) };
        if err != 0 {
            return Err(format!("cudaStreamCreate failed: cuda error {err}"));
        }
        Ok(Self { ptr })
    }

    pub fn sync(&self) {
        let err = unsafe { cudaStreamSynchronize(self.ptr) };
        assert!(err == 0, "cudaStreamSynchronize failed: {err}");
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaStreamDestroy(self.ptr) };
        }
    }
}

// cudaMemcpyKind
pub const MEMCPY_H2D: i32 = 1;
pub const MEMCPY_D2H: i32 = 2;
pub const MEMCPY_D2D: i32 = 3;

// cudaMemPoolAttr
pub const MEMPOOL_ATTR_RELEASE_THRESHOLD: i32 = 4;

// CUDA memory info
unsafe extern "C" {
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
}

/// Query current VRAM state. Returns (free_bytes, total_bytes).
pub fn vram_query() -> (usize, usize) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    let err = unsafe { cudaMemGetInfo(&mut free, &mut total) };
    assert!(err == 0, "cudaMemGetInfo failed: {err}");
    (free, total)
}

/// VRAM safety check via nvidia-smi. Must be called before CUDA context init
/// (before detect_wsl2_and_configure) so readings reflect external processes only.
pub fn vram_preflight_check() {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"])
        .output();

    if let Ok(out) = output {
        let s = String::from_utf8_lossy(&out.stdout);
        let parts: Vec<&str> = s.trim().split(',').collect();
        if parts.len() == 3 {
            let used_mb: usize = parts[0].trim().parse().unwrap_or(0);
            let free_mb: usize = parts[1].trim().parse().unwrap_or(0);
            let total_mb: usize = parts[2].trim().parse().unwrap_or(0);
            eprintln!("[VRAM] {used_mb} MB used / {total_mb} MB total ({free_mb} MB free)");
            if used_mb > 512 {
                eprintln!("[VRAM] WARNING: {used_mb} MB already in use by another process.");
                eprintln!("[VRAM] Another GPU workload may be running.");
                eprintln!("[VRAM] VortexSTARK needs up to 28 GB for large proofs (log_n>=27).");
                eprintln!("[VRAM] Proceeding, but large proofs may OOM. Stop other GPU processes first.");
            }
            return;
        }
    }
    // Fallback to cudaMemGetInfo if nvidia-smi unavailable (note: requires active CUDA context)
    let (free, total) = vram_query();
    let used = total - free;
    eprintln!("[VRAM] {} MB used / {} MB total ({} MB free)",
        used / (1024*1024), total / (1024*1024), free / (1024*1024));
}

/// Flush the CUDA memory pool back to the OS. Call after each job completes
/// to release VRAM that's no longer needed. Keeps `keep_bytes` in the pool
/// for fast reuse on the next job (0 = release everything).
pub fn vram_release(keep_bytes: usize) {
    if crate::device::buffer::use_sync() {
        // Sync mode: no pool to trim, just sync the device
        unsafe { cudaDeviceSynchronize(); }
        return;
    }
    unsafe {
        cudaDeviceSynchronize();
        let mut pool: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = cudaDeviceGetDefaultMemPool(&mut pool, 0);
        if err != 0 { return; }
        cudaMemPoolTrimTo(pool, keep_bytes);
    }
}

/// Initialize CUDA memory pool for async allocation. Call once at startup.
///
/// Runs a VRAM preflight check, then configures the default memory pool.
/// The pool release threshold is set to 2 GB — freed buffers stay cached
/// up to that limit for fast reuse, but anything beyond 2 GB is returned
/// to the OS between jobs so other processes can share the GPU.
pub fn init_memory_pool() {
    // Check VRAM before CUDA context is initialized (nvidia-smi sees only external processes)
    vram_preflight_check();

    // Detect WSL2 and switch to sync malloc if needed
    crate::device::buffer::detect_wsl2_and_configure();

    // Sanity check: verify GPU is accessible with a tiny allocation
    unsafe {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = cudaMalloc(&mut ptr, 64);
        if err != 0 {
            panic!("[CUDA] GPU sanity check failed: cudaMalloc(64) returned error {err}. \
                    Is the CUDA driver working? Try: nvidia-smi");
        }
        cudaFree(ptr);
        let err = cudaDeviceSynchronize();
        if err != 0 {
            panic!("[CUDA] GPU sync failed after sanity check: error {err}");
        }
        eprintln!("[CUDA] GPU sanity check passed");
    }

    // Skip pool configuration on WSL2 (not using async alloc)
    if crate::device::buffer::use_sync() {
        return;
    }

    unsafe {
        let mut pool: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = cudaDeviceGetDefaultMemPool(&mut pool, 0);
        assert!(err == 0, "cudaDeviceGetDefaultMemPool failed: {err}");
        // Keep 2 GB cached in the pool for fast reuse; release the rest.
        let threshold: u64 = 2 * 1024 * 1024 * 1024;
        let err = cudaMemPoolSetAttribute(
            pool,
            MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const u64 as *const std::ffi::c_void,
        );
        assert!(err == 0, "cudaMemPoolSetAttribute failed: {err}");
    }
}

/// Initialize with greedy pool (never releases memory). Use only for
/// back-to-back benchmarks where you know nothing else needs the GPU.
pub fn init_memory_pool_greedy() {
    // Check VRAM before CUDA context is initialized (nvidia-smi sees only external processes)
    vram_preflight_check();
    crate::device::buffer::detect_wsl2_and_configure();

    if crate::device::buffer::use_sync() {
        return;
    }

    unsafe {
        let mut pool: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = cudaDeviceGetDefaultMemPool(&mut pool, 0);
        assert!(err == 0, "cudaDeviceGetDefaultMemPool failed: {err}");
        let threshold: u64 = u64::MAX;
        let err = cudaMemPoolSetAttribute(
            pool,
            MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const u64 as *const std::ffi::c_void,
        );
        assert!(err == 0, "cudaMemPoolSetAttribute failed: {err}");
    }
}

// Field operation kernels
unsafe extern "C" {
    pub fn cuda_m31_add(a: *const u32, b: *const u32, out: *mut u32, n: u32);
    pub fn cuda_m31_mul(a: *const u32, b: *const u32, out: *mut u32, n: u32);
    pub fn cuda_device_sync();
}

// Stark252 NTT kernels (SoA layout: 4*n u64s per array, one block per limb)
unsafe extern "C" {
    /// Forward NTT over Stark252. d_data and d_tw are SoA u64 device pointers.
    /// d_data: 4*n u64s (modified in-place). d_tw: 4*(n/2) u64s of twiddles ω_N^j.
    pub fn cuda_stark252_ntt_forward(d_data: *mut u64, d_tw: *const u64, log_n: u32);

    /// Inverse NTT over Stark252 (includes 1/N scaling).
    /// d_inv_n: pointer to 4 u64s representing 1/N in standard form.
    pub fn cuda_stark252_ntt_inverse(
        d_data: *mut u64,
        d_tw: *const u64,
        log_n: u32,
        d_inv_n: *const u64,
    );
}

// Stwo-compatible Circle NTT kernels (flat twiddle format)
unsafe extern "C" {
    /// Forward NTT using stwo's flat twiddle buffer.
    pub fn cuda_stwo_ntt_evaluate(d_data: *mut u32, d_twiddles: *const u32, n: u32);
    /// Inverse NTT using stwo's flat twiddle buffer (includes 1/n scaling).
    pub fn cuda_stwo_ntt_interpolate(d_data: *mut u32, d_itwiddles: *const u32, n: u32);
}

// Circle NTT kernels (original VortexSTARK format)
unsafe extern "C" {
    pub fn cuda_circle_ntt_evaluate(
        d_data: *mut u32,
        d_twiddles: *const u32,
        d_circle_twids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
    );

    pub fn cuda_circle_ntt_interpolate(
        d_data: *mut u32,
        d_itwiddles: *const u32,
        d_circle_itwids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
    );

    pub fn cuda_circle_ntt_evaluate_batch(
        d_columns: *mut *mut u32,
        d_twiddles: *const u32,
        d_circle_twids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
        n_cols: u32,
    );

    pub fn cuda_circle_ntt_interpolate_batch(
        d_columns: *mut *mut u32,
        d_itwiddles: *const u32,
        d_circle_itwids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
        n_cols: u32,
    );

    pub fn cuda_circle_ntt_layer(
        d_data: *mut u32,
        d_twiddles: *const u32,
        layer_idx: u32,
        n: u32,
        forward: i32,
    );

    pub fn cuda_bit_reverse_m31(data: *mut u32, log_n: u32);

    /// Permute hc-natural → canonic-BRT in one GPU pass.
    /// Equivalent to `Coset::permute_hc_natural_to_canonic_brt` on CPU.
    /// Used to keep quotient-commit data GPU-resident instead of
    /// round-tripping through host for the permute.
    pub fn cuda_permute_hc_to_canonic_brt(
        src: *const u32,
        dst: *mut u32,
        n: u32,
        log_n: u32,
    );

    /// Inverse of `cuda_permute_hc_to_canonic_brt` — takes BRT-canonic input
    /// and writes hc-natural output. Used to produce constraint-kernel input
    /// on GPU from committed Merkle-order data without a host round trip.
    pub fn cuda_permute_canonic_brt_to_hc_natural(
        src: *const u32,
        dst: *mut u32,
        n: u32,
        log_n: u32,
    );

    pub fn cuda_eval_at_point(
        d_coeffs: *const u32,
        d_folding_factors: *const u32,
        h_result: *mut u32,
        n: u32,
        d_scratch1: *mut u32,
        d_scratch2: *mut u32,
    );
}

// FRI fold kernels (SoA layout)
unsafe extern "C" {
    pub fn cuda_fold_line_soa(
        in0: *const u32, in1: *const u32, in2: *const u32, in3: *const u32,
        twiddles: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha: *const u32, // [4] on host
        half_n: u32,
    );

    pub fn cuda_fold_circle_into_line_soa(
        dst0: *mut u32, dst1: *mut u32, dst2: *mut u32, dst3: *mut u32,
        src0: *const u32, src1: *const u32, src2: *const u32, src3: *const u32,
        twiddles: *const u32,
        alpha: *const u32,     // [4] on host
        alpha_sq: *const u32,  // [4] on host
        half_n: u32,
    );

    /// FORGE-emitted FRI fold_line_soa. Same semantics as
    /// `cuda_fold_line_soa` but generated from
    /// forge/analysis/vortex_ntt/fri_fold_line.fg with 193 proof
    /// obligations discharged. Inputs must be canonical M31
    /// (< P); the kernel canonicalizes anyway as a contract guard.
    pub fn cuda_fold_line_soa_forge(
        in0: *const u32, in1: *const u32, in2: *const u32, in3: *const u32,
        twiddles: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha: *const u32,
        half_n: u32,
    );

    /// FORGE-emitted FRI fold_circle_into_line_soa. Source:
    /// forge/analysis/vortex_ntt/fri_fold_circle.fg (237 obligations).
    pub fn cuda_fold_circle_into_line_soa_forge(
        dst0: *mut u32, dst1: *mut u32, dst2: *mut u32, dst3: *mut u32,
        src0: *const u32, src1: *const u32, src2: *const u32, src3: *const u32,
        twiddles: *const u32,
        alpha: *const u32,
        alpha_sq: *const u32,
        half_n: u32,
    );

    /// FORGE-emitted single-column Circle NTT butterfly layer.
    /// Equivalent to `cuda_circle_ntt_layer` from cuda/circle_ntt.cu,
    /// generated from forge/analysis/vortex_ntt/circle_ntt_layer.fg
    /// (138 proof obligations). `forward != 0` selects the forward
    /// butterfly; `forward == 0` selects the inverse.
    pub fn cuda_circle_ntt_layer_forge(
        data: *mut u32, twiddles: *const u32,
        layer_idx: u32, n: u32, forward: i32,
    );

    /// FORGE-emitted batched Circle NTT (multi-column SoA, full
    /// evaluate / interpolate). Source:
    /// forge/analysis/vortex_ntt/circle_ntt_batch.fg (148 obligations
    /// across 3 kernels: forward butterfly, inverse butterfly, scale).
    /// Same ABI as `cuda_circle_ntt_evaluate_batch`. Drop-in
    /// replacement when `forge-ntt` is on, supplanting the prior
    /// loop-of-single-col fallback.
    pub fn cuda_circle_ntt_evaluate_batch_forge(
        col_ptrs: *const *mut u32,
        d_twiddles: *const u32,
        d_circle_twids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
        n_cols: u32,
    );

    pub fn cuda_circle_ntt_interpolate_batch_forge(
        col_ptrs: *const *mut u32,
        d_itwiddles: *const u32,
        d_circle_itwids: *const u32,
        h_layer_offsets: *const u32,
        h_layer_sizes: *const u32,
        n_line_layers: u32,
        n: u32,
        n_cols: u32,
    );
}

// Twiddle computation kernels
unsafe extern "C" {
    pub fn cuda_compute_fold_twiddle_sources(
        initial_x: u32, initial_y: u32,
        step_x: u32, step_y: u32,
        output: *mut u32,
        n: u32, log_n: u32,
        extract_y: i32,
    );

    pub fn cuda_batch_inverse_m31(input: *const u32, output: *mut u32, n: u32);

    pub fn cuda_compute_fold_twiddle_sources_stream(
        initial_x: u32, initial_y: u32,
        step_x: u32, step_y: u32,
        output: *mut u32,
        n: u32, log_n: u32,
        extract_y: i32,
        stream: *mut c_void,
    );

    pub fn cuda_batch_inverse_m31_stream(input: *const u32, output: *mut u32, n: u32, stream: *mut c_void);

    pub fn cuda_compute_coset_points(
        initial_x: u32, initial_y: u32,
        step_x: u32, step_y: u32,
        output_x: *mut u32, output_y: *mut u32,
        n: u32,
    );

    pub fn cuda_squash_x(input: *const u32, output: *mut u32, out_n: u32);

    pub fn cuda_extract_and_squash(
        input: *const u32,
        twiddle_out: *mut u32,
        squash_out: *mut u32,
        half_n: u32,
    );
}

// Constraint evaluation kernels
unsafe extern "C" {
    pub fn cuda_fibonacci_quotient(
        trace: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha: *const u32, // [4] on host
        n: u32,
    );

    pub fn cuda_zero_pad(
        src: *const u32,
        dst: *mut u32,
        src_n: u32,
        dst_n: u32,
    );

    pub fn cuda_fibonacci_quotient_chunk(
        trace: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha: *const u32, // [4] on host
        offset: u32,
        chunk_n: u32,
        global_n: u32,
    );

    pub fn cuda_fibonacci_quotient_chunk_stream(
        trace: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha: *const u32,
        offset: u32, chunk_n: u32, global_n: u32,
        stream: *mut std::ffi::c_void,
    );

    // LogUp interaction kernels
    /// Chunked LogUp: process one (addr, value) pair, accumulate into running sum.
    /// Call 4 times (once per memory access) with is_first=1 for the first call.
    pub fn cuda_logup_accumulate_pair(
        col_addr: *const u32, col_val: *const u32,
        acc0: *mut u32, acc1: *mut u32, acc2: *mut u32, acc3: *mut u32,
        z: *const u32, alpha: *const u32,
        n: u32, is_first: u32,
    );

    pub fn cuda_logup_memory_fused(
        col_pc: *const u32, col_inst_lo: *const u32, col_inst_hi: *const u32,
        col_dst_addr: *const u32, col_dst: *const u32,
        col_op0_addr: *const u32, col_op0: *const u32,
        col_op1_addr: *const u32, col_op1: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        z: *const u32, alpha: *const u32,
        n: u32,
    );

    pub fn cuda_logup_rc_fused(
        col_off0: *const u32, col_off1: *const u32, col_off2: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        z_rc: *const u32, n: u32,
    );

    pub fn cuda_logup_memory_denoms(
        col_pc: *const u32, col_inst_lo: *const u32,
        col_dst_addr: *const u32, col_dst: *const u32,
        col_op0_addr: *const u32, col_op0: *const u32,
        col_op1_addr: *const u32, col_op1: *const u32,
        denom0: *mut u32, denom1: *mut u32, denom2: *mut u32, denom3: *mut u32,
        z: *const u32, alpha: *const u32,
        n: u32,
    );

    pub fn cuda_logup_memory_combine(
        col_pc: *const u32, col_inst_lo: *const u32,
        col_dst_addr: *const u32, col_dst: *const u32,
        col_op0_addr: *const u32, col_op0: *const u32,
        col_op1_addr: *const u32, col_op1: *const u32,
        inv0: *const u32, inv1: *const u32, inv2: *const u32, inv3: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        z: *const u32, alpha: *const u32,
        n: u32,
    );

    pub fn cuda_qm31_inverse(
        in0: *const u32, in1: *const u32, in2: *const u32, in3: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        n: u32,
    );

    pub fn cuda_qm31_block_scan(
        c0: *mut u32, c1: *mut u32, c2: *mut u32, c3: *mut u32,
        block_sums0: *mut u32, block_sums1: *mut u32,
        block_sums2: *mut u32, block_sums3: *mut u32,
        n: u32, block_size: u32,
    );

    pub fn cuda_qm31_add_block_prefix(
        c0: *mut u32, c1: *mut u32, c2: *mut u32, c3: *mut u32,
        prefix0: *const u32, prefix1: *const u32,
        prefix2: *const u32, prefix3: *const u32,
        n: u32, block_size: u32,
    );

    pub fn cuda_compute_vanishing_inv(
        initial_x: u32, initial_y: u32,
        step_x: u32, step_y: u32,
        out_vh_inv: *mut u32,
        log_eval: u32,
        log_n: u32,
    );

    /// Compute Z_H (vanishing polynomial, NOT its inverse) at every eval-domain position.
    /// Used for ZK blinding: column[i] += r * Z_H[i] before trace commitment.
    pub fn cuda_compute_vanishing(
        initial_x: u32, initial_y: u32,
        step_x: u32, step_y: u32,
        out_zh: *mut u32,
        log_eval: u32,
        log_n: u32,
    );

    /// Fused multiply-add: y[i] = y[i] + scalar * x[i]  (mod P).
    /// Used for ZK blinding: add r * Z_H to a trace column.
    pub fn cuda_axpy_m31(scalar: u32, x: *const u32, y: *mut u32, n: u32);

    pub fn cuda_cairo_quotient(
        trace_cols: *const *const u32,
        s_logup0: *const u32, s_logup1: *const u32, s_logup2: *const u32, s_logup3: *const u32,
        t1l0: *const u32, t1l1: *const u32, t1l2: *const u32, t1l3: *const u32,
        t2l0: *const u32, t2l1: *const u32, t2l2: *const u32, t2l3: *const u32,
        t3l0: *const u32, t3l1: *const u32, t3l2: *const u32, t3l3: *const u32,
        s_rc0: *const u32, s_rc1: *const u32, s_rc2: *const u32, s_rc3: *const u32,
        u1r0: *const u32, u1r1: *const u32, u1r2: *const u32, u1r3: *const u32,
        u2r0: *const u32, u2r1: *const u32, u2r2: *const u32, u2r3: *const u32,
        s_dict0: *const u32, s_dict1: *const u32, s_dict2: *const u32, s_dict3: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha_coeffs: *const u32,
        vh_inv: *const u32,
        trans_factor: *const u32,
        challenges: *const u32,
        n: u32,
        blowup_step: u32,
    );

    /// Slab-mode chunked variant of `cuda_cairo_quotient`. All column
    /// pointers must address per-chunk slabs spanning `chunk_n +
    /// blowup_step` rows starting at global row `chunk_offset`. The
    /// trailing `blowup_step` overlap rows let the kernel read
    /// `next_i = local_i + blowup_step` without crossing the slab end.
    /// `vh_inv`, `trans_factor`, and `out0..out3` remain full-eval-size
    /// and are written/read at `chunk_offset + local_i`. For the last
    /// chunk, the slab's trailing rows must wrap the FIRST `blowup_step`
    /// rows of the global eval domain.
    pub fn cuda_cairo_quotient_slab(
        trace_cols: *const *const u32,
        s_logup0: *const u32, s_logup1: *const u32, s_logup2: *const u32, s_logup3: *const u32,
        t1l0: *const u32, t1l1: *const u32, t1l2: *const u32, t1l3: *const u32,
        t2l0: *const u32, t2l1: *const u32, t2l2: *const u32, t2l3: *const u32,
        t3l0: *const u32, t3l1: *const u32, t3l2: *const u32, t3l3: *const u32,
        s_rc0: *const u32, s_rc1: *const u32, s_rc2: *const u32, s_rc3: *const u32,
        u1r0: *const u32, u1r1: *const u32, u1r2: *const u32, u1r3: *const u32,
        u2r0: *const u32, u2r1: *const u32, u2r2: *const u32, u2r3: *const u32,
        s_dict0: *const u32, s_dict1: *const u32, s_dict2: *const u32, s_dict3: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha_coeffs: *const u32,
        vh_inv: *const u32,
        trans_factor: *const u32,
        challenges: *const u32,
        chunk_n: u32,
        blowup_step: u32,
        chunk_offset: u32,
    );

    pub fn cuda_cairo_quotient_chunk(
        trace_cols: *const *const u32,
        s_logup0: *const u32, s_logup1: *const u32, s_logup2: *const u32, s_logup3: *const u32,
        s_rc0: *const u32, s_rc1: *const u32, s_rc2: *const u32, s_rc3: *const u32,
        s_dict0: *const u32, s_dict1: *const u32, s_dict2: *const u32, s_dict3: *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        alpha_coeffs: *const u32,
        vh_inv: *const u32,
        trans_factor: *const u32,
        challenges: *const u32,
        offset: u32, chunk_n: u32, global_n: u32,
        blowup_step: u32,
    );

    // Fp252 test
    pub fn cuda_fp252_test(results: *mut u64);

    // Pedersen GPU
    pub fn cuda_pedersen_test_double(
        px: *const u64, py: *const u64,
        out_x: *mut u64, out_y: *mut u64, out_z: *mut u64,
    );

    pub fn cuda_pedersen_upload_points(px: *const u64, py: *const u64);

    // Upload precomputed windowed tables + P0 in Montgomery form
    pub fn cuda_pedersen_upload_tables(
        table_x: *const u64, table_y: *const u64, table_z: *const u64, // [4][16] each
        p0_x: *const u64, p0_y: *const u64, p0_z: *const u64, // single point
    );
    pub fn cuda_pedersen_hash_batch(
        inputs_a: *const u64, inputs_b: *const u64,
        out_x: *mut u64, out_zz: *mut u64, n: u32,
    );

    pub fn cuda_pedersen_hash_batch_stream(
        inputs_a: *const u64, inputs_b: *const u64,
        out_x: *mut u64, out_zz: *mut u64, n: u32,
        stream: *mut c_void,
    );

    /// Decompose pre-computed Fp252 values into 27 M31 trace columns (no hashing).
    pub fn cuda_pedersen_decompose(
        vals_a: *const u64, vals_b: *const u64, vals_out: *const u64,
        trace_cols: *mut *mut u32,
        n: u32,
        stream: *mut c_void,
    );

    /// EC trace generation: outputs intermediate Jacobian points per step.
    pub fn cuda_pedersen_ec_trace(
        inputs_a: *const u64, inputs_b: *const u64,
        ec_trace: *mut u64, ec_ops: *mut u32,
        n: u32, stream: *mut c_void,
    );

    /// Decompose raw EC trace (u64 Jacobian) into M31 SoA columns.
    pub fn cuda_ec_trace_decompose(
        ec_trace: *const u64, ec_ops: *const u32,
        trace_cols: *mut *mut u32,
        n_rows: u32, stream: *mut c_void,
    );

    /// Fused Pedersen hash + trace column generation.
    /// Hashes (a, b) pairs and decomposes results into 27 M31 trace columns on GPU.
    /// trace_cols: device pointer to array of 27 device pointers (one per column).
    pub fn cuda_pedersen_trace(
        inputs_a: *const u64, inputs_b: *const u64,
        trace_cols: *mut *mut u32,
        n: u32,
        stream: *mut c_void,
    );

    pub fn cuda_poseidon_upload_round_consts(host_rc: *const u32);

    pub fn cuda_poseidon_trace(
        block_inputs: *const u32,
        trace_cols: *const *mut u32,
        n_blocks: u32,
    );

    pub fn cuda_poseidon_quotient(
        trace_cols: *const *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        round_consts: *const u32,
        alpha_coeffs: *const u32,
        n: u32,
    );

    pub fn cuda_poseidon_quotient_chunk(
        trace_cols: *const *const u32,
        out0: *mut u32, out1: *mut u32, out2: *mut u32, out3: *mut u32,
        round_consts: *const u32,
        alpha_coeffs: *const u32,
        offset: u32, chunk_n: u32, global_n: u32,
    );

    pub fn cuda_interleave_u32(
        even: *const u32,
        odd: *const u32,
        output: *mut u32,
        half_n: u32,
    );

    pub fn cuda_rpo_upload_constants(host_mds: *const u32, host_rc: *const u32);

    pub fn cuda_rpo_trace(
        block_inputs: *const u32,
        trace_cols: *const *mut u32,
        n_blocks: u32,
    );

    pub fn cuda_p2f_upload_consts(host_rc: *const u32);

    pub fn cuda_p2f_trace(
        block_inputs: *const u32,
        trace_cols: *const *mut u32,
        n_blocks: u32,
    );
}

// Blake2s PoW grinding kernel
unsafe extern "C" {
    /// Launch GPU grind kernel. Each of n_threads threads tries one nonce
    /// starting from batch_offset. Result is atomicMin'd into result[0]
    /// (must be initialized to u64::MAX by caller).
    pub fn cuda_grind_pow(
        prefixed_digest: *const u32, // device ptr, [8] words
        result: *mut u64,            // device ptr, [1] word
        pow_bits: u32,
        batch_offset: u64,
        n_threads: u32,
    );
}

// Blake2s Merkle tree kernels
unsafe extern "C" {
    pub fn cuda_merkle_hash_leaves(
        columns: *const *const u32,
        hashes: *mut u32,
        n_cols: u32,
        n_leaves: u32,
    );

    /// Shinobi-hash support: reduce every u32 in `data[0..n_words]`
    /// to the M31 range `[0, P-1]` in place. Matches `M31::reduce` on
    /// the CPU side. Post-processes hash outputs to produce the
    /// `HashValue<QM31>`-safe form the Shinobi verifier expects.
    pub fn cuda_reduce_words_to_m31(data: *mut u32, n_words: u32);

    /// Single-column FORGE-emitted Blake2s leaf hash. Equivalent to
    /// `cuda_merkle_hash_leaves(&[column], hashes, 1, n_leaves)` but
    /// generated from a verified source (analysis/vortex_ntt/
    /// merkle_hash_leaves.fg, 133 proof obligations discharged).
    /// Output is M31-reduced unconditionally (matches shinobi-hash
    /// post-reduce; for non-shinobi builds the reduction is a no-op
    /// since canonical M31 leaves are already in range).
    pub fn cuda_merkle_hash_leaves_forge_single(
        column: *const u32,
        hashes: *mut u32,
        n_leaves: u32,
    );

    /// Four-column FORGE-emitted Blake2s leaf hash. Equivalent to
    /// `cuda_merkle_hash_leaves(&[c0,c1,c2,c3], hashes, 4, n_leaves)`.
    /// FORGE source: forge/analysis/vortex_ntt/merkle_hash_leaves.fg
    /// `merkle_hash_leaves_quad` kernel (proof obligations included in
    /// the file's 145-obligation total). Same M31-reduce semantics
    /// as the single-column variant.
    pub fn cuda_merkle_hash_leaves_forge_quad(
        c0: *const u32,
        c1: *const u32,
        c2: *const u32,
        c3: *const u32,
        hashes: *mut u32,
        n_leaves: u32,
    );

    pub fn cuda_merkle_hash_nodes(
        children: *const u32,
        parents: *mut u32,
        n_parents: u32,
    );

    pub fn cuda_merkle_reduce_to_root(
        nodes: *const u32,
        root_out: *mut u32,
        n_nodes: u32,
    );

    pub fn cuda_merkle_commit_small_soa4(
        col0: *const u32, col1: *const u32,
        col2: *const u32, col3: *const u32,
        root_out: *mut u32,
        n_leaves: u32,
    );

    pub fn cuda_merkle_hash_leaves_merge_soa4(
        col0: *const u32, col1: *const u32,
        col2: *const u32, col3: *const u32,
        parents: *mut u32,
        n_leaves: u32,
    );

    pub fn cuda_merkle_tiled_soa4(
        col0: *const u32, col1: *const u32,
        col2: *const u32, col3: *const u32,
        subtree_roots: *mut u32,
        n_leaves: u32,
    );

    pub fn cuda_merkle_tiled_soa4_stream(
        col0: *const u32, col1: *const u32,
        col2: *const u32, col3: *const u32,
        subtree_roots: *mut u32,
        n_leaves: u32,
        stream: *mut std::ffi::c_void,
    );

    pub fn cuda_merkle_tiled_generic(
        columns: *const *const u32,
        subtree_roots: *mut u32,
        n_cols: u32,
        n_leaves: u32,
    );
}

// Bytecode constraint evaluation kernel
unsafe extern "C" {
    /// Execute a bytecode constraint evaluation program on the GPU.
    ///
    /// Each GPU thread interprets the same bytecode program for one row,
    /// evaluating all constraints and accumulating the results into the
    /// output columns (SoA QM31 format).
    ///
    /// - `bytecode`: device pointer to encoded u32 instruction stream
    /// - `n_words`: number of u32 words in the bytecode
    /// - `trace_cols`: device pointer to array of device pointers (one per column)
    /// - `trace_col_sizes`: device pointer to array of column sizes (unused, reserved)
    /// - `n_trace_cols`: number of columns in trace_cols
    /// - `n_rows`: number of rows in the evaluation domain
    /// - `trace_n_rows`: number of rows in the trace domain (power of 2)
    /// - `random_coeff_powers`: device pointer to [n_constraints * 4] QM31 values (reversed)
    /// - `denom_inv`: device pointer to [1 << log_expand] M31 denominator inverses
    /// - `log_expand`: eval_log_size - trace_log_size
    /// - `accum0..3`: device pointers to output accumulator columns (SoA QM31)
    pub fn cuda_bytecode_constraint_eval(
        bytecode: *const u32,
        n_words: u32,
        trace_cols: *const *const u32,
        trace_col_sizes: *const u32,
        n_trace_cols: u32,
        n_rows: u32,
        trace_n_rows: u32,
        random_coeff_powers: *const u32,
        denom_inv: *const u32,
        log_expand: u32,
        accum0: *mut u32,
        accum1: *mut u32,
        accum2: *mut u32,
        accum3: *mut u32,
        n_registers: u32,
    );

    /// Warp-cooperative bytecode constraint eval kernel.
    /// Uses one warp (32 threads) per row, distributing the register file
    /// across warp lanes. Each thread holds ceil(n_registers/32) registers.
    /// Same interface as cuda_bytecode_constraint_eval.
    pub fn cuda_warp_bytecode_constraint_eval(
        bytecode: *const u32,
        n_words: u32,
        trace_cols: *const *const u32,
        trace_col_sizes: *const u32,
        n_trace_cols: u32,
        n_rows: u32,
        trace_n_rows: u32,
        random_coeff_powers: *const u32,
        denom_inv: *const u32,
        log_expand: u32,
        accum0: *mut u32,
        accum1: *mut u32,
        accum2: *mut u32,
        accum3: *mut u32,
        n_registers: u32,
    );

    // ── GPU lifted Merkle leaf hashing ──────────────────────────────────

    /// Build Merkle leaves on GPU using the lifted hashing algorithm.
    /// Columns of different sizes are handled via the lifted row index formula.
    ///
    /// - `col_ptrs`: device pointer to array of column device pointers
    /// - `schedule`: device pointer to array of LeafHashChunk structs
    /// - `n_chunks`: number of chunks in the schedule
    /// - `lifting_log_size`: log2 of output leaf count
    /// - `output_hashes`: device pointer to [n_leaves * 8] u32s (Blake2s hashes)
    /// - `n_leaves`: 2^lifting_log_size
    pub fn cuda_build_leaves_lifted(
        col_ptrs: *const *const u32,
        schedule: *const u8,  // LeafHashChunk array (passed as raw bytes)
        n_chunks: u32,
        lifting_log_size: u32,
        output_hashes: *mut u32,
        n_leaves: u32,
    );

    // ── Poseidon252 GPU Merkle ──────────────────────────────────────────

    /// Build Poseidon252 leaf hashes for a lifted Merkle tree.
    ///
    /// - `col_ptrs`: device pointer to array of column device pointers (uint32_t**)
    /// - `col_log_sizes`: device pointer to array of log2(col_length) per column
    /// - `n_cols`: number of columns
    /// - `lifting_log_size`: log2 of output leaf count
    /// - `output_hashes`: device pointer to [n_leaves * 4] u64s (Fp252 = 4×u64)
    /// - `n_leaves`: 2^lifting_log_size
    pub fn build_leaves_poseidon252(
        col_ptrs: *const *const u32,
        col_log_sizes: *const u32,
        n_cols: u32,
        lifting_log_size: u32,
        output_hashes: *mut u64,
        n_leaves: u32,
    );

    /// Build one level of the Poseidon252 Merkle tree (parent hashes from children).
    ///
    /// - `prev_layer`: device pointer to [2*n_parents * 4] u64s
    /// - `output`: device pointer to [n_parents * 4] u64s
    /// - `n_parents`: number of parent nodes to compute
    pub fn build_next_layer_poseidon252(
        prev_layer: *const u64,
        output: *mut u64,
        n_parents: u32,
    );

    // ── Batched gather ──────────────────────────────────────────────────

    /// Gather u32 elements: dst[i] = src[idx[i]]
    /// Used for batched CudaColumn<BaseField> decommitment.
    pub fn cuda_gather_u32(src: *const u32, idx: *const u32, dst: *mut u32, n: u32);

    /// Gather 8-word (256-bit) elements: dst[i*8..+8] = src[idx[i]*8..+8]
    /// Used for batched CudaColumn<Blake2sHash> decommitment.
    pub fn cuda_gather_u256(src: *const u32, idx: *const u32, dst: *mut u32, n: u32);

    /// FORGE-emitted variant of `cuda_gather_u32`. Same semantics but
    /// takes the explicit `src_len` so the kernel can guard against
    /// out-of-range `idx[i]` values (FORGE proves the load is safe
    /// only when this guard is present).
    pub fn cuda_gather_u32_forge(
        src: *const u32, idx: *const u32, dst: *mut u32,
        n: u32, src_len: u32,
    );

    /// FORGE-emitted variant of `cuda_gather_u256`. Same semantics but
    /// takes the explicit `src_len` (in u32 units) so the kernel can
    /// bounds-check the 8-word load.
    pub fn cuda_gather_u256_forge(
        src: *const u32, idx: *const u32, dst: *mut u32,
        n: u32, src_len: u32,
    );

    /// FORGE-emitted QM31 bit-reverse. SecureField columns store QM31
    /// elements as 4 consecutive u32s. This kernel permutes the 4-u32
    /// chunks into bit-reversed index order. Equivalent to
    /// `cuda_bit_reverse_qm31` from cuda/bit_reverse_wide.cu, generated
    /// from forge/analysis/vortex_ntt/bit_reverse_qm31.fg
    /// (12 proof obligations including bounded-loop termination on
    /// the bit-reverse inner loop).
    ///
    /// Under `--features open-toolchain`, this symbol is replaced by a
    /// Rust shim (defined below) that loads an OpenCUDA+OpenPTXas-built
    /// cubin via cuModuleLoadData and launches via cuLaunchKernel — no
    /// nvcc/ptxas compilation in the build path for this kernel.
    #[cfg(not(feature = "open-toolchain"))]
    pub fn cuda_bit_reverse_qm31_forge(
        in_buf: *const u32, out_buf: *mut u32, n: u32, log_n: u32,
    );

    /// FORGE-emitted M31 batch inverse via Montgomery's trick. Same
    /// chunk size (64 elements per thread) as `cuda_batch_inverse_m31`.
    /// Generated from forge/analysis/vortex_ntt/batch_inverse.fg
    /// (160 proof obligations including m31_inv termination + the
    /// prefix-product loop invariants).
    pub fn cuda_batch_inverse_m31_forge(
        input: *const u32, output: *mut u32, n: u32,
    );

    // ── Barycentric evaluation ──────────────────────────────────────────

    /// Compute result = sum_i(evals[i] * weights[i]) using a parallel reduction.
    ///
    /// - `evals`: device pointer to n M31 values (1 u32 each)
    /// - `weights`: device pointer to n QM31 values in AoS layout (4 u32 per element)
    /// - `n`: number of elements
    /// - `out`: device pointer to output buffer of `n_blocks * 4` u32s (QM31 partial sums)
    /// - `n_blocks`: number of parallel reduction blocks (caller allocates out and
    ///   must CPU-reduce the partial sums after the call)
    pub fn cuda_barycentric_eval(
        evals: *const u32,
        weights: *const u32,
        n: u32,
        out: *mut u32,
        n_blocks: u32,
    );

    // ── FRI Quotient kernels ────────────────────────────────────────────

    /// Accumulate partial numerators for a single sample batch.
    ///
    /// For each row: result = sum_i (c_i * col[col_idx_i][row] - b_i)
    /// where (b_i, c_i) are QM31 line coefficients.
    ///
    /// - `col_ptrs`: device array of pointers to M31 column data
    /// - `col_indices`: device array of column indices to use [n_batch_cols]
    /// - `b_coeffs`: device array of QM31 b-coefficients [n_batch_cols * 4] M31 limbs
    /// - `c_coeffs`: device array of QM31 c-coefficients [n_batch_cols * 4] M31 limbs
    /// - `n_batch_cols`: number of columns in this batch
    /// - `n_rows`: number of rows
    /// - `out0..3`: output SoA QM31 accumulator [n_rows] each
    pub fn cuda_accumulate_numerators(
        col_ptrs: *const *const u32,
        col_indices: *const u32,
        b_coeffs: *const u32,
        c_coeffs: *const u32,
        n_batch_cols: u32,
        n_rows: u32,
        out0: *mut u32,
        out1: *mut u32,
        out2: *mut u32,
        out3: *mut u32,
    );

    /// Fused dual-sample accumulate_numerators: computes both the z-sample
    /// and z_next-sample accumulators in a single kernel pass, reading
    /// each M31 column value once. `zn_coeff_idx[i]` = index of
    /// col_indices_z[i] within the z_next's coefficient arrays, or
    /// UINT32_MAX if this column is not part of the z_next set.
    pub fn cuda_accumulate_numerators_dual(
        col_ptrs: *const *const u32,
        col_indices_z: *const u32,
        zn_coeff_idx: *const u32,
        b_coeffs_z: *const u32,
        c_coeffs_z: *const u32,
        b_coeffs_zn: *const u32,
        c_coeffs_zn: *const u32,
        n_cols_z: u32,
        n_rows: u32,
        out_z0: *mut u32, out_z1: *mut u32, out_z2: *mut u32, out_z3: *mut u32,
        out_zn0: *mut u32, out_zn1: *mut u32, out_zn2: *mut u32, out_zn3: *mut u32,
    );

    /// Compute FRI quotients and combine across sample points.
    ///
    /// For each row:
    ///   domain_point = (domain_xs[row], domain_ys[row])
    ///   quotient[row] = sum_j (numer_j[lifted_idx] - a_acc_j * y) * den_inv_j
    ///
    /// - `sample_points_x/y`: device arrays of QM31 sample point coords [n_accs * 4]
    /// - `first_linear_acc`: device array of QM31 a-coefficients [n_accs * 4]
    /// - `numer_ptrs0..3`: device arrays of pointers to partial numerator SoA columns
    /// - `acc_log_sizes`: device array of log2(size) per accumulation [n_accs]
    /// - `n_accs`: number of accumulations
    /// - `domain_xs/ys`: device arrays of M31 domain point coords [n_rows]
    /// - `lifting_log_size`: log2 of the lifting domain size
    /// - `n_rows`: number of output rows
    /// - `out0..3`: output SoA QM31 [n_rows] each
    pub fn cuda_compute_quotients_combine(
        sample_points_x: *const u32,
        sample_points_y: *const u32,
        first_linear_acc: *const u32,
        numer_ptrs0: *const *const u32,
        numer_ptrs1: *const *const u32,
        numer_ptrs2: *const *const u32,
        numer_ptrs3: *const *const u32,
        acc_log_sizes: *const u32,
        n_accs: u32,
        domain_xs: *const u32,
        domain_ys: *const u32,
        lifting_log_size: u32,
        n_rows: u32,
        out0: *mut u32,
        out1: *mut u32,
        out2: *mut u32,
        out3: *mut u32,
    );
}

// ── GKR sum-check GPU kernels ───────────────────────────────────���────────────
unsafe extern "C" {
    /// fix_first_variable: BaseField (M31) → SecureField (QM31)
    /// in: n M31 values (n u32s), out: n/2 QM31 (2n u32s), r_ptr: [4] QM31 on device
    pub fn cuda_gkr_fix_first_variable_base(
        inp: *const u32, out: *mut u32, r_ptr: *const u32, n: u32,
    );
    /// fix_first_variable: SecureField (QM31) → SecureField (QM31)
    /// in: n QM31 (4n u32s), out: n/2 QM31 (2n u32s), r_ptr: [4] QM31 on device
    pub fn cuda_gkr_fix_first_variable_secure(
        inp: *const u32, out: *mut u32, r_ptr: *const u32, n: u32,
    );
    /// gen_eq_evals: initialize buf[0..3] = v (single thread)
    pub fn cuda_gkr_gen_eq_evals_init(buf: *mut u32, v_ptr: *const u32);
    /// gen_eq_evals: one doubling pass (cur_size threads)
    pub fn cuda_gkr_gen_eq_evals_pass(buf: *mut u32, y_i_ptr: *const u32, cur_size: u32);
    /// next_layer for GrandProduct: out[i] = in[2i]*in[2i+1], n elements → n/2
    pub fn cuda_gkr_next_layer_grand_product(inp: *const u32, out: *mut u32, n: u32);
    /// next_layer for LogUpGeneric/Multiplicities (QM31 numerators): n → n/2
    pub fn cuda_gkr_next_layer_logup_generic(
        in_num: *const u32, in_den: *const u32,
        out_num: *mut u32, out_den: *mut u32, n: u32,
    );
    /// next_layer for LogUpMultiplicities (M31 numerators): n → n/2
    pub fn cuda_gkr_next_layer_logup_mult(
        in_num: *const u32, in_den: *const u32,
        out_num: *mut u32, out_den: *mut u32, n: u32,
    );
    /// next_layer for LogUpSingles: n → n/2
    pub fn cuda_gkr_next_layer_logup_singles(
        in_den: *const u32, out_num: *mut u32, out_den: *mut u32, n: u32,
    );
    /// sum_as_poly for GrandProduct: writes n_blocks*8 u32s, returns n_blocks
    pub fn cuda_gkr_sum_poly_grand_product(
        eq_evals: *const u32, layer: *const u32,
        partial_sums: *mut u32, n_terms: u32,
    ) -> u32;
    /// sum_as_poly for LogUpGeneric/Multiplicities (QM31 numerators)
    pub fn cuda_gkr_sum_poly_logup_generic(
        eq_evals: *const u32, in_num: *const u32, in_den: *const u32,
        lambda_ptr: *const u32, partial_sums: *mut u32, n_terms: u32,
    ) -> u32;
    /// sum_as_poly for LogUpMultiplicities (M31 numerators)
    pub fn cuda_gkr_sum_poly_logup_mult(
        eq_evals: *const u32, in_num: *const u32, in_den: *const u32,
        lambda_ptr: *const u32, partial_sums: *mut u32, n_terms: u32,
    ) -> u32;
    /// sum_as_poly for LogUpSingles
    pub fn cuda_gkr_sum_poly_logup_singles(
        eq_evals: *const u32, in_den: *const u32,
        lambda_ptr: *const u32, partial_sums: *mut u32, n_terms: u32,
    ) -> u32;
}

// ── SecureField bit-reverse, lift_and_accumulate, pack_leaves ────────────────
unsafe extern "C" {
    /// Bit-reverse n QM31 elements out-of-place (in→out, both 4n u32s)
    pub fn cuda_bit_reverse_qm31(inp: *const u32, out: *mut u32, n: u32, log_n: u32);
    /// In-place lift_and_accumulate pass: col[i] += curr[src_idx(i)] for i < col_n
    pub fn cuda_accumulate_lift(col: *mut u32, curr: *const u32, col_n: u32, log_ratio: u32);
    /// Pack 4 M31 input columns (size N) into 64 output columns (size N/16)
    pub fn cuda_pack_leaves(
        input_col_ptrs: *const *const u32,
        output_col_ptrs: *const *mut u32,
        n: u32,
    );
}

// ── PoW grinding variants ─────────────────────────────────────────────────────
unsafe extern "C" {
    /// Blake2s grind with M31 reduction applied to output words before trailing-zero check.
    pub fn cuda_grind_pow_m31_output(
        prefixed_digest: *const u32,
        result: *mut u64,
        pow_bits: u32,
        batch_offset: u64,
        n_threads: u32,
    );
    /// Poseidon252 PoW grinding. prefixed_digest_mont: [4] u64 in Montgomery form.
    pub fn cuda_grind_pow_poseidon(
        prefixed_digest_mont: *const u64,
        result: *mut u64,
        pow_bits: u32,
        batch_offset: u64,
        n_threads: u32,
    );
}

// ─── FB-1 open toolchain ─────────────────────────────────────────────────
// CUDA driver-API bindings + Rust shim that replaces the FORGE
// `cuda_bit_reverse_qm31_forge` symbol when the `open-toolchain` feature
// is enabled.  The kernel is built at compile time via OpenCUDA +
// OpenPTXas (see build.rs) and embedded as a static byte array.  At
// first call, we cuModuleLoadData against the runtime API's primary
// context and cache the CUmodule + CUfunction handles for life-of-process.
//
// Driver-API symbols are linked from libcuda (the CUDA driver, not the
// runtime) — same library the existing cudaMalloc / cudaMemcpy resolve
// through indirectly.  No new crate dependencies.

#[cfg(feature = "open-toolchain")]
unsafe extern "C" {
    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32;
    fn cuModuleGetFunction(
        hfunc: *mut *mut c_void, hmod: *mut c_void, name: *const std::ffi::c_char,
    ) -> i32;
    fn cuLaunchKernel(
        f: *mut c_void,
        grid_x: u32, grid_y: u32, grid_z: u32,
        block_x: u32, block_y: u32, block_z: u32,
        shared_mem_bytes: u32,
        stream: *mut c_void,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuDevicePrimaryCtxRetain(ctx: *mut *mut c_void, dev: i32) -> i32;
    fn cuCtxSetCurrent(ctx: *mut c_void) -> i32;
}

#[cfg(feature = "open-toolchain")]
pub mod open_toolchain {
    //! Rust-side launcher for FORGE kernels built via OpenCUDA + OpenPTXas.
    //!
    //! The cubin is embedded at compile time from `OPEN_TOOLCHAIN_CUBIN_DIR`
    //! (set by build.rs).  First call to any kernel here triggers a one-time
    //! `cuModuleLoadData` against the CUDA runtime's primary context, so
    //! GPU pointers from `cudaMalloc` are valid here too.
    use std::ffi::c_void;
    use std::sync::OnceLock;

    static CUBIN_BIT_REVERSE_QM31: &[u8] = include_bytes!(concat!(
        env!("OPEN_TOOLCHAIN_CUBIN_DIR"),
        "/bit_reverse_qm31.cubin"
    ));

    /// Cached (CUmodule, CUfunction) for `bit_reverse_qm31` — initialized
    /// on first call.
    static BIT_REVERSE_HANDLES: OnceLock<(usize, usize)> = OnceLock::new();

    fn ensure_bit_reverse_loaded() -> (*mut c_void, *mut c_void) {
        let (mod_addr, func_addr) = *BIT_REVERSE_HANDLES.get_or_init(|| unsafe {
            // Bind the runtime API's primary context to the calling thread.
            // GPU memory allocated via cudaMalloc lives in this context;
            // launching against a different context would see invalid ptrs.
            let mut ctx: *mut c_void = std::ptr::null_mut();
            let rc = super::cuDevicePrimaryCtxRetain(&mut ctx, 0);
            assert!(rc == 0,
                    "cuDevicePrimaryCtxRetain failed: {rc}");
            let rc = super::cuCtxSetCurrent(ctx);
            assert!(rc == 0, "cuCtxSetCurrent failed: {rc}");

            let mut module: *mut c_void = std::ptr::null_mut();
            let rc = super::cuModuleLoadData(
                &mut module,
                CUBIN_BIT_REVERSE_QM31.as_ptr() as *const c_void,
            );
            assert!(rc == 0,
                    "cuModuleLoadData(bit_reverse_qm31) failed: {rc}");

            let mut func: *mut c_void = std::ptr::null_mut();
            let rc = super::cuModuleGetFunction(
                &mut func, module,
                b"bit_reverse_qm31\0".as_ptr() as *const std::ffi::c_char,
            );
            assert!(rc == 0,
                    "cuModuleGetFunction(bit_reverse_qm31) failed: {rc}");
            (module as usize, func as usize)
        });
        (mod_addr as *mut c_void, func_addr as *mut c_void)
    }

    /// Open-toolchain replacement for the nvcc-compiled
    /// `cuda_bit_reverse_qm31_forge`.  Same signature, same semantics —
    /// just routed through OpenCUDA → OpenPTXas → cuModuleLoadData →
    /// cuLaunchKernel.  Mirrors the host shim at
    /// cuda/bit_reverse_qm31_forge.cu lines 12-24 (grid/block, span ABI,
    /// argument order).
    ///
    /// Argument layout (post struct-by-value flattening per OpenCUDA's
    /// param ABI fix in opencuda commit 0e1e621):
    ///   .param .u64 in_buf_data, .param .u64 in_buf_len,   // u32 count
    ///   .param .u64 out_buf_data, .param .u64 out_buf_len, // u32 count
    ///   .param .u64 n,            .param .u32 log_n
    ///
    /// SAFETY: caller-passed pointers must be valid GPU device addresses
    /// in the current CUDA primary context (allocated via cudaMalloc).
    /// `n` must equal the QM31 element count (each QM31 = 4 u32).
    pub unsafe fn cuda_bit_reverse_qm31_forge(
        in_buf: *const u32, out_buf: *mut u32, n: u32, log_n: u32,
    ) {
        if n == 0 {
            return;
        }
        let (_module, func) = ensure_bit_reverse_loaded();

        // Flatten the two `forge_span_u32_t` structs into 4 separate u64
        // parameters.  `len` is the u32 element count (= n * 4 since each
        // QM31 has 4 u32s), matching the host shim at
        // cuda/bit_reverse_qm31_forge.cu line 21: `(uintptr_t)n * 4u`.
        let in_buf_data: u64 = in_buf as u64;
        let in_buf_len: u64 = (n as u64) * 4;
        let out_buf_data: u64 = out_buf as u64;
        let out_buf_len: u64 = (n as u64) * 4;
        let n_u64: u64 = n as u64;
        let log_n_u32: u32 = log_n;

        let mut args: [*mut c_void; 6] = [
            &in_buf_data as *const u64 as *mut c_void,
            &in_buf_len as *const u64 as *mut c_void,
            &out_buf_data as *const u64 as *mut c_void,
            &out_buf_len as *const u64 as *mut c_void,
            &n_u64 as *const u64 as *mut c_void,
            &log_n_u32 as *const u32 as *mut c_void,
        ];

        let threads: u32 = 256;
        let blocks: u32 = (n + threads - 1) / threads;

        unsafe {
            let rc = super::cuLaunchKernel(
                func,
                blocks, 1, 1,
                threads, 1, 1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            assert!(rc == 0,
                    "cuLaunchKernel(bit_reverse_qm31) failed: {rc}");
        }
    }
}

/// Open-toolchain shim for `cuda_bit_reverse_qm31_forge` — same signature
/// as the nvcc-compiled extern, dispatches to the cuModuleLoad-based
/// launcher.  Callers in src/blake2s_m31.rs etc. don't need to change.
#[cfg(feature = "open-toolchain")]
pub unsafe fn cuda_bit_reverse_qm31_forge(
    in_buf: *const u32, out_buf: *mut u32, n: u32, log_n: u32,
) {
    unsafe {
        open_toolchain::cuda_bit_reverse_qm31_forge(in_buf, out_buf, n, log_n)
    }
}
