//! QuotientOps: DEEP quotient accumulation on GPU.
//!
//! `accumulate_numerators`: runs a CUDA kernel per sample batch that computes
//!   partial_numerators[row] = sum_i (c_i * col[col_idx_i][row] - b_i)
//! where (b_i, c_i) are precomputed QM31 line coefficients.
//!
//! `compute_quotients_and_combine`: runs a CUDA kernel that computes per-row
//!   quotient[row] = sum_j (numer_j[lifted_idx] - a_acc_j * y) * den_inv_j
//! where den_inv_j is computed from (sample_point_j, domain_point) on GPU.

use std::iter::zip;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use tracing::{span, Level};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::quotients::{column_line_coeffs, ColumnSampleBatch};
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::utils::bit_reverse_index;
use stwo::prover::backend::{Column, CpuBackend};
use stwo::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use stwo::prover::poly::twiddles::TwiddleTree;
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::secure_column::SecureColumnByCoords;
use stwo::prover::{AccumulatedNumerators, QuotientOps};
use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;

use super::CudaBackend;

// Per-call counters for the QuotientOps GPU path.
pub static ACC_NUM_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static ACC_NUM_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static QC_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static QC_NANOS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub fn quotient_stats_take() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        ACC_NUM_CALLS.swap(0, Relaxed),
        ACC_NUM_NANOS.swap(0, Relaxed),
        QC_CALLS.swap(0, Relaxed),
        QC_NANOS.swap(0, Relaxed),
    )
}


fn secure_to_raw(v: SecureField) -> [u32; 4] {
    let arr = v.to_m31_array();
    [arr[0].0, arr[1].0, arr[2].0, arr[3].0]
}


/// Cache for bit-reversed subdomain (x, y) coordinate buffers used by
/// compute_quotients_and_combine. Keyed by (subdomain_log_size,
/// log_blowup_factor) since the canonical subdomain is deterministic for
/// that pair. Built once per process; later proves with the same key
/// reuse the GPU upload instead of running the 8M-row CPU precompute loop.
struct DomainPointBuffers {
    xs: DeviceBuffer<u32>,
    ys: DeviceBuffer<u32>,
}

fn domain_points_cache() -> &'static Mutex<HashMap<(u32, u32), Arc<DomainPointBuffers>>> {
    static CACHE: OnceLock<Mutex<HashMap<(u32, u32), Arc<DomainPointBuffers>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_compute_domain_points(
    domain: &stwo::core::poly::circle::CircleDomain,
    subdomain_log_size: u32,
    log_blowup_factor: u32,
) -> Arc<DomainPointBuffers> {
    let key = (subdomain_log_size, log_blowup_factor);
    {
        let cache = domain_points_cache().lock().unwrap();
        if let Some(arc) = cache.get(&key) {
            return arc.clone();
        }
    }
    let n_rows = 1usize << subdomain_log_size;
    let mut xs = Vec::with_capacity(n_rows);
    let mut ys = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let pt = domain.at(bit_reverse_index(row, subdomain_log_size));
        xs.push(pt.x.0);
        ys.push(pt.y.0);
    }
    let new_arc = Arc::new(DomainPointBuffers {
        xs: DeviceBuffer::from_host(&xs),
        ys: DeviceBuffer::from_host(&ys),
    });
    let mut cache = domain_points_cache().lock().unwrap();
    if let Some(existing) = cache.get(&key) {
        return existing.clone();
    }
    cache.insert(key, new_arc.clone());
    new_arc
}

struct StatsGuard {
    t0: std::time::Instant,
    calls: &'static std::sync::atomic::AtomicU64,
    nanos: &'static std::sync::atomic::AtomicU64,
}
impl Drop for StatsGuard {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.calls.fetch_add(1, Relaxed);
        self.nanos.fetch_add(self.t0.elapsed().as_nanos() as u64, Relaxed);
    }
}

impl QuotientOps for CudaBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
        log_blowup_factor: u32,
    ) {
        let _g = StatsGuard { t0: std::time::Instant::now(), calls: &ACC_NUM_CALLS, nanos: &ACC_NUM_NANOS };
        let _span =
            span!(Level::INFO, "GPU accumulate_numerators", n_batches = sample_batches.len())
                .entered();

        // Upstream stwo accumulates over only the first
        // `column_size >> log_blowup_factor` subdomain rows (bit-reversed).
        // The bit-reversed-prefix property guarantees those are the first
        // `subdomain_size` slots of the column, so we just pass
        // `subdomain_size` as `n_rows` to the existing kernel.
        let size = columns[0].len();
        let subdomain_size = size >> log_blowup_factor;

        // Precompute line coefficients on CPU (small: one (a,b,c) per column per batch).
        let line_coeffs = column_line_coeffs(sample_batches);

        // Build device pointer array for all columns.
        let col_ptrs: Vec<*const u32> = columns.iter().map(|c| c.values.device_ptr()).collect();
        let d_col_ptrs = DeviceBuffer::from_host(&col_ptrs);

        for (batch, coeffs) in zip(sample_batches, line_coeffs) {
            let n_batch_cols = batch.cols_vals_randpows.len();

            // Flatten column indices, b-coefficients, c-coefficients for this batch.
            let mut col_indices = Vec::with_capacity(n_batch_cols);
            let mut b_flat = Vec::with_capacity(n_batch_cols * 4);
            let mut c_flat = Vec::with_capacity(n_batch_cols * 4);

            for (nd, (_, b, c)) in zip(&batch.cols_vals_randpows, &coeffs) {
                col_indices.push(nd.column_index as u32);
                let b_raw = secure_to_raw(*b);
                let c_raw = secure_to_raw(*c);
                b_flat.extend_from_slice(&b_raw);
                c_flat.extend_from_slice(&c_raw);
            }

            // Upload batch parameters to GPU.
            let d_col_indices = DeviceBuffer::from_host(&col_indices);
            let d_b = DeviceBuffer::from_host(&b_flat);
            let d_c = DeviceBuffer::from_host(&c_flat);

            // Allocate output SoA columns of subdomain size.
            let mut out0 = DeviceBuffer::<u32>::alloc(subdomain_size);
            let mut out1 = DeviceBuffer::<u32>::alloc(subdomain_size);
            let mut out2 = DeviceBuffer::<u32>::alloc(subdomain_size);
            let mut out3 = DeviceBuffer::<u32>::alloc(subdomain_size);

            unsafe {
                ffi::cuda_accumulate_numerators(
                    d_col_ptrs.as_ptr() as *const *const u32,
                    d_col_indices.as_ptr(),
                    d_b.as_ptr(),
                    d_c.as_ptr(),
                    n_batch_cols as u32,
                    subdomain_size as u32,
                    out0.as_mut_ptr(),
                    out1.as_mut_ptr(),
                    out2.as_mut_ptr(),
                    out3.as_mut_ptr(),
                );
                ffi::cuda_device_sync();
            }

            // Build GPU-resident SecureColumnByCoords from the 4 output buffers.
            use super::column::CudaColumn;
            let gpu_coords = SecureColumnByCoords {
                columns: [
                    CudaColumn::from_device_buffer(out0, subdomain_size),
                    CudaColumn::from_device_buffer(out1, subdomain_size),
                    CudaColumn::from_device_buffer(out2, subdomain_size),
                    CudaColumn::from_device_buffer(out3, subdomain_size),
                ],
            };

            let first_linear_term_acc: SecureField = coeffs.iter().map(|(a, ..)| a).sum();

            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                partial_numerators_acc: gpu_coords,
                first_linear_term_acc,
            });
        }
    }

    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
        log_blowup_factor: u32,
        _twiddles: &TwiddleTree<Self>,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let _g = StatsGuard { t0: std::time::Instant::now(), calls: &QC_CALLS, nanos: &QC_NANOS };
        let _span = span!(
            Level::INFO,
            "GPU compute_quotients_combine",
            n_accs = accs.len(),
            lifting_log_size = lifting_log_size,
        )
        .entered();

        // Upstream stwo computes the quotient on the subdomain (size
        // 2^(lifting_log_size - log_blowup_factor)) and then lifts via
        // interpolation + evaluation through the supplied twiddle tree.
        // We do the same: compute_quotients_combine_kernel produces
        // subdomain-sized output (subdomain_log_size as the kernel'''s
        // "lifting_log_size" arg), then we IFFT each channel and FFT to
        // the full eval domain.
        let eval_domain = CanonicCoset::new(lifting_log_size).circle_domain();
        let (eval_subdomain, _) = eval_domain.split(log_blowup_factor);
        let subdomain_log_size = eval_subdomain.log_size();
        let n_rows = 1u32 << subdomain_log_size;
        let n_accs = accs.len();
        let domain = eval_subdomain;

        // Cached bit-reversed subdomain points.
        let domain_pts = get_or_compute_domain_points(&domain, subdomain_log_size, log_blowup_factor);

        // Pack sample point data.
        let mut sp_x_flat = Vec::with_capacity(n_accs * 4);
        let mut sp_y_flat = Vec::with_capacity(n_accs * 4);
        let mut fla_flat = Vec::with_capacity(n_accs * 4);
        let mut acc_log_sizes = Vec::with_capacity(n_accs);

        // Collect device pointers to partial numerator columns (SoA: 4 per acc).
        let mut numer_host_ptrs0: Vec<*const u32> = Vec::with_capacity(n_accs);
        let mut numer_host_ptrs1: Vec<*const u32> = Vec::with_capacity(n_accs);
        let mut numer_host_ptrs2: Vec<*const u32> = Vec::with_capacity(n_accs);
        let mut numer_host_ptrs3: Vec<*const u32> = Vec::with_capacity(n_accs);

        for acc in &accs {
            let sp = acc.sample_point;
            // CirclePoint<SecureField> has x: SecureField, y: SecureField.
            // secure_to_raw converts SecureField -> [u32; 4] via to_m31_array.
            sp_x_flat.extend_from_slice(&secure_to_raw(sp.x));
            sp_y_flat.extend_from_slice(&secure_to_raw(sp.y));
            fla_flat.extend_from_slice(&secure_to_raw(acc.first_linear_term_acc));

            let acc_size = acc.partial_numerators_acc.columns[0].len();
            acc_log_sizes.push(acc_size.ilog2());

            numer_host_ptrs0.push(acc.partial_numerators_acc.columns[0].device_ptr());
            numer_host_ptrs1.push(acc.partial_numerators_acc.columns[1].device_ptr());
            numer_host_ptrs2.push(acc.partial_numerators_acc.columns[2].device_ptr());
            numer_host_ptrs3.push(acc.partial_numerators_acc.columns[3].device_ptr());
        }

        let d_sp_x = DeviceBuffer::from_host(&sp_x_flat);
        let d_sp_y = DeviceBuffer::from_host(&sp_y_flat);
        let d_fla = DeviceBuffer::from_host(&fla_flat);
        let d_acc_log_sizes = DeviceBuffer::from_host(&acc_log_sizes);
        let d_numer0 = DeviceBuffer::from_host(&numer_host_ptrs0);
        let d_numer1 = DeviceBuffer::from_host(&numer_host_ptrs1);
        let d_numer2 = DeviceBuffer::from_host(&numer_host_ptrs2);
        let d_numer3 = DeviceBuffer::from_host(&numer_host_ptrs3);

        // Allocate output.
        let mut out0 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut out1 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut out2 = DeviceBuffer::<u32>::alloc(n_rows as usize);
        let mut out3 = DeviceBuffer::<u32>::alloc(n_rows as usize);

        unsafe {
            ffi::cuda_compute_quotients_combine(
                d_sp_x.as_ptr(),
                d_sp_y.as_ptr(),
                d_fla.as_ptr(),
                d_numer0.as_ptr() as *const *const u32,
                d_numer1.as_ptr() as *const *const u32,
                d_numer2.as_ptr() as *const *const u32,
                d_numer3.as_ptr() as *const *const u32,
                d_acc_log_sizes.as_ptr(),
                n_accs as u32,
                domain_pts.xs.as_ptr(),
                domain_pts.ys.as_ptr(),
                subdomain_log_size,
                n_rows,
                out0.as_mut_ptr(),
                out1.as_mut_ptr(),
                out2.as_mut_ptr(),
                out3.as_mut_ptr(),
            );
            ffi::cuda_device_sync();
        }

        // Lift: for each of the 4 SoA channels, interpolate over eval_subdomain
        // (IFFT) and evaluate over eval_domain (FFT). CudaBackend::interpolate
        // and evaluate look up GPU twiddles via the internal coset cache, so the
        // passed `_twiddles` argument is unused.
        use super::column::CudaColumn;
        use stwo::prover::poly::circle::PolyOps;

        let dummy_twiddles = CudaBackend::precompute_twiddles(eval_subdomain.half_coset);
        let lift = |sub_buf: DeviceBuffer<u32>| -> CudaColumn<BaseField> {
            let col = CudaColumn::<BaseField>::from_device_buffer(sub_buf, n_rows as usize);
            let circle_eval = CircleEvaluation::<CudaBackend, BaseField, BitReversedOrder>::new(
                eval_subdomain,
                col,
            );
            let poly = CudaBackend::interpolate(circle_eval, &dummy_twiddles);
            let evaluated = CudaBackend::evaluate(&poly, eval_domain, &dummy_twiddles);
            evaluated.values
        };

        let lifted_cols = SecureColumnByCoords {
            columns: [lift(out0), lift(out1), lift(out2), lift(out3)],
        };
        SecureEvaluation::new(eval_domain, lifted_cols)
    }
}
