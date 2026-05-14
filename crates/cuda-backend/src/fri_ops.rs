//! FriOps: FRI folding on GPU.
//!
//! Fold twiddles computed on CPU from stwo's domain iteration (matching
//! the CPU FRI exactly), uploaded as flat arrays for GPU kernels.

use stwo::core::fields::qm31::SecureField;
use stwo::core::utils::bit_reverse_index;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::fri::FriOps;
use stwo::prover::poly::circle::SecureEvaluation;
use stwo::prover::line::LineEvaluation;
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;
use super::column::CudaColumn;


/// Fold-twiddle caches keyed by source-domain log_size.
///
/// fold_circle_into_line and fold_line each iterate the canonical input
/// domain on CPU to compute inverse y/x coordinates, then upload the result
/// to GPU. Both inputs are deterministic per log_size for canonical domains
/// (the only kind FRI uses), so we cache the GPU buffer and reuse it across
/// proves in the same process. Same pattern as the quotient_ops domain-point
/// cache (commit 7f59fa7).
fn circle_fold_twiddles_cache() -> &'static Mutex<HashMap<u32, Arc<DeviceBuffer<u32>>>> {
    static CACHE: OnceLock<Mutex<HashMap<u32, Arc<DeviceBuffer<u32>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn line_fold_twiddles_cache() -> &'static Mutex<HashMap<u32, Arc<DeviceBuffer<u32>>>> {
    static CACHE: OnceLock<Mutex<HashMap<u32, Arc<DeviceBuffer<u32>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_compute_circle_fold_twiddles(
    domain: stwo::core::poly::circle::CircleDomain,
) -> Arc<DeviceBuffer<u32>> {
    let key = domain.log_size();
    {
        let cache = circle_fold_twiddles_cache().lock().unwrap();
        if let Some(arc) = cache.get(&key) {
            return arc.clone();
        }
    }
    let half_n = (1usize << key) / 2;
    let mut inv_y_vals = Vec::with_capacity(half_n);
    for i in 0..half_n {
        let p = domain.at(bit_reverse_index(i << 1, key));
        inv_y_vals.push(p.y.inverse().0);
    }
    let new_arc = Arc::new(DeviceBuffer::from_host(&inv_y_vals));
    let mut cache = circle_fold_twiddles_cache().lock().unwrap();
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }
    cache.insert(key, new_arc.clone());
    new_arc
}

fn get_or_compute_line_fold_twiddles(
    domain: stwo::core::poly::line::LineDomain,
) -> Arc<DeviceBuffer<u32>> {
    let key = domain.log_size();
    {
        let cache = line_fold_twiddles_cache().lock().unwrap();
        if let Some(arc) = cache.get(&key) {
            return arc.clone();
        }
    }
    let half_n = (1usize << key) / 2;
    let mut inv_x_vals = Vec::with_capacity(half_n);
    for i in 0..half_n {
        let x = domain.at(bit_reverse_index(i << 1, key));
        inv_x_vals.push(x.inverse().0);
    }
    let new_arc = Arc::new(DeviceBuffer::from_host(&inv_x_vals));
    let mut cache = line_fold_twiddles_cache().lock().unwrap();
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }
    cache.insert(key, new_arc.clone());
    new_arc
}

// Per-call counters for FriOps GPU path.
pub static FOLD_LINE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static FOLD_LINE_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static FOLD_CIRCLE_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static FOLD_CIRCLE_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub fn fri_stats_take() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        FOLD_LINE_CALLS.swap(0, Relaxed),
        FOLD_LINE_NANOS.swap(0, Relaxed),
        FOLD_CIRCLE_CALLS.swap(0, Relaxed),
        FOLD_CIRCLE_NANOS.swap(0, Relaxed),
    )
}

struct FriStatsGuard {
    t0: std::time::Instant,
    calls: &'static std::sync::atomic::AtomicU64,
    nanos: &'static std::sync::atomic::AtomicU64,
}
impl Drop for FriStatsGuard {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.calls.fetch_add(1, Relaxed);
        self.nanos.fetch_add(self.t0.elapsed().as_nanos() as u64, Relaxed);
    }
}

impl FriOps for CudaBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alphas: &[SecureField],
        _twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let _g = FriStatsGuard { t0: std::time::Instant::now(), calls: &FOLD_LINE_CALLS, nanos: &FOLD_LINE_NANOS };
        assert!(!alphas.is_empty(), "fold_line requires at least one alpha");

        let mut res = gpu_fold_line_single(eval, alphas[0]);
        for &alpha in &alphas[1..] {
            res = gpu_fold_line_single(&res, alpha);
        }
        res
    }

    fn fold_circle_into_line(
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        _twiddles: &TwiddleTree<Self>,
    ) -> LineEvaluation<Self> {
        let _g = FriStatsGuard { t0: std::time::Instant::now(), calls: &FOLD_CIRCLE_CALLS, nanos: &FOLD_CIRCLE_NANOS };
        let n = src.len();
        let half_n = n / 2;
        let domain = src.domain;

        // Cached fold twiddles: 1/y for each pair's domain point.
        let d_twiddles = get_or_compute_circle_fold_twiddles(domain);

        let src_cols = &src.values.columns;
        // Allocate destination buffers (replaces the old &mut dst arg —
        // upstream stwo's fold_circle_into_line now owns the dst).
        let mut o0 = DeviceBuffer::<u32>::alloc(half_n);
        let mut o1 = DeviceBuffer::<u32>::alloc(half_n);
        let mut o2 = DeviceBuffer::<u32>::alloc(half_n);
        let mut o3 = DeviceBuffer::<u32>::alloc(half_n);

        let alpha_arr = qm31_to_arr(alpha);
        let alpha_sq = alpha * alpha;
        let alpha_sq_arr = qm31_to_arr(alpha_sq);

        unsafe {
            // Pre-zero the new outputs so the SoA kernel's `dst = dst*alpha^2 + f'`
            // accumulation reduces to plain `dst = f'`. (Upstream takes ownership
            // of an uninitialized buffer; we match that semantics by zero-init.)
            ffi::cudaMemset(o0.as_mut_ptr() as *mut _, 0, (half_n * 4) as usize);
            ffi::cudaMemset(o1.as_mut_ptr() as *mut _, 0, (half_n * 4) as usize);
            ffi::cudaMemset(o2.as_mut_ptr() as *mut _, 0, (half_n * 4) as usize);
            ffi::cudaMemset(o3.as_mut_ptr() as *mut _, 0, (half_n * 4) as usize);
            ffi::cuda_fold_circle_into_line_soa(
                o0.as_mut_ptr(), o1.as_mut_ptr(),
                o2.as_mut_ptr(), o3.as_mut_ptr(),
                src_cols[0].buf.as_ptr(), src_cols[1].buf.as_ptr(),
                src_cols[2].buf.as_ptr(), src_cols[3].buf.as_ptr(),
                d_twiddles.as_ptr(),
                alpha_arr.as_ptr(),
                alpha_sq_arr.as_ptr(),
                half_n as u32,
            );
            ffi::cuda_device_sync();
        }

        let dst_values = SecureColumnByCoords {
            columns: [
                CudaColumn::from_device_buffer(o0, half_n),
                CudaColumn::from_device_buffer(o1, half_n),
                CudaColumn::from_device_buffer(o2, half_n),
                CudaColumn::from_device_buffer(o3, half_n),
            ],
        };
        let dst_domain = stwo::core::poly::line::LineDomain::new(
            stwo::core::circle::Coset::half_odds(domain.log_size() - 1),
        );
        LineEvaluation::new(dst_domain, dst_values)
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        // CPU — small computation
        let cpu_eval = secure_eval_to_cpu(eval);
        let (cpu_result, lambda) = CpuBackend::decompose(&cpu_eval);
        (secure_eval_from_cpu(&cpu_result), lambda)
    }
}

/// GPU fold_line: compute twiddles from domain iteration (matching CPU exactly).
fn gpu_fold_line_single(
    eval: &LineEvaluation<CudaBackend>,
    alpha: SecureField,
) -> LineEvaluation<CudaBackend> {
    let n = eval.len();
    assert!(n >= 2);
    let half_n = n / 2;
    let domain = eval.domain();

    // Cached fold twiddles: 1/x for each pair's domain point.
    let d_twiddles = get_or_compute_line_fold_twiddles(domain);

    let cols = &eval.values.columns;
    let alpha_arr = qm31_to_arr(alpha);

    let mut o0 = DeviceBuffer::<u32>::alloc(half_n);
    let mut o1 = DeviceBuffer::<u32>::alloc(half_n);
    let mut o2 = DeviceBuffer::<u32>::alloc(half_n);
    let mut o3 = DeviceBuffer::<u32>::alloc(half_n);

    unsafe {
        ffi::cuda_fold_line_soa(
            cols[0].buf.as_ptr(), cols[1].buf.as_ptr(),
            cols[2].buf.as_ptr(), cols[3].buf.as_ptr(),
            d_twiddles.as_ptr(),
            o0.as_mut_ptr(), o1.as_mut_ptr(), o2.as_mut_ptr(), o3.as_mut_ptr(),
            alpha_arr.as_ptr(),
            half_n as u32,
        );
        ffi::cuda_device_sync();
    }

    let result = SecureColumnByCoords {
        columns: [
            CudaColumn::from_device_buffer(o0, half_n),
            CudaColumn::from_device_buffer(o1, half_n),
            CudaColumn::from_device_buffer(o2, half_n),
            CudaColumn::from_device_buffer(o3, half_n),
        ],
    };
    LineEvaluation::new(domain.double(), result)
}

fn secure_eval_to_cpu(eval: &SecureEvaluation<CudaBackend, BitReversedOrder>) -> SecureEvaluation<CpuBackend, BitReversedOrder> {
    let cpu_coords = SecureColumnByCoords {
        columns: std::array::from_fn(|i| eval.values.columns[i].to_cpu()),
    };
    SecureEvaluation::new(eval.domain, cpu_coords)
}

fn secure_eval_from_cpu(eval: &SecureEvaluation<CpuBackend, BitReversedOrder>) -> SecureEvaluation<CudaBackend, BitReversedOrder> {
    let gpu_coords = SecureColumnByCoords {
        columns: std::array::from_fn(|i| eval.values.columns[i].iter().copied().collect()),
    };
    SecureEvaluation::new(eval.domain, gpu_coords)
}

fn qm31_to_arr(v: SecureField) -> [u32; 4] {
    let arr = v.to_m31_array();
    [arr[0].0, arr[1].0, arr[2].0, arr[3].0]
}
