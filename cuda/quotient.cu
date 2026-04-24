#include "include/qm31.cuh"

// ═══════════════════════════════════════════════════════════════════════════
//  accumulate_numerators kernel
// ═══════════════════════════════════════════════════════════════════════════
//
// For a single sample batch, compute per-row:
//   result[row] = sum_i (c_i * f_i[row] - b_i)
// where:
//   - (b_i, c_i) are QM31 line coefficients for the i-th column in this batch
//   - f_i[row] is the M31 column value at `row`
//
// The result is stored as SoA QM31 (4 separate M31 arrays).
// All columns have the same size.
//
// Parameters:
//   col_ptrs:    device array of pointers to M31 column data
//   col_indices: which columns from col_ptrs to use (length = n_batch_cols)
//   b_coeffs:    QM31 b coefficients [n_batch_cols * 4] (flat M31 limbs)
//   c_coeffs:    QM31 c coefficients [n_batch_cols * 4] (flat M31 limbs)
//   n_batch_cols: number of columns in this batch
//   n_rows:      number of rows
//   out0..3:     output SoA QM31 accumulator

__global__ void accumulate_numerators_kernel(
    const uint32_t* const* __restrict__ col_ptrs,
    const uint32_t* __restrict__ col_indices,
    const uint32_t* __restrict__ b_coeffs,   // [n_batch_cols * 4]
    const uint32_t* __restrict__ c_coeffs,   // [n_batch_cols * 4]
    uint32_t n_batch_cols,
    uint32_t n_rows,
    uint32_t* __restrict__ out0,
    uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2,
    uint32_t* __restrict__ out3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    QM31 acc = qm31_zero();
    for (uint32_t i = 0; i < n_batch_cols; i++) {
        uint32_t col_idx = col_indices[i];
        uint32_t f_val = col_ptrs[col_idx][row];

        // Load c coefficient (QM31)
        QM31 c;
        c.v[0] = c_coeffs[i * 4 + 0];
        c.v[1] = c_coeffs[i * 4 + 1];
        c.v[2] = c_coeffs[i * 4 + 2];
        c.v[3] = c_coeffs[i * 4 + 3];

        // Load b coefficient (QM31)
        QM31 b;
        b.v[0] = b_coeffs[i * 4 + 0];
        b.v[1] = b_coeffs[i * 4 + 1];
        b.v[2] = b_coeffs[i * 4 + 2];
        b.v[3] = b_coeffs[i * 4 + 3];

        // c * f_val - b
        QM31 term = qm31_sub(qm31_mul_m31(c, f_val), b);
        acc = qm31_add(acc, term);
    }

    out0[row] = acc.v[0];
    out1[row] = acc.v[1];
    out2[row] = acc.v[2];
    out3[row] = acc.v[3];
}

// ═══════════════════════════════════════════════════════════════════════════
//  accumulate_numerators_dual kernel
// ═══════════════════════════════════════════════════════════════════════════
//
// Fused version of accumulate_numerators that computes BOTH the z-sample and
// the z_next-sample accumulators in a single pass, reading each M31 column
// value ONCE instead of twice. The two accumulations share ~90% of their
// columns (trace + interaction cols) at different (b, c) coefficients, so
// fusing them cuts device-memory traffic almost in half.
//
// Layout:
//   cols indexed by col_indices_z[0..n_cols_z] and col_indices_zn[0..n_cols_zn]
//   For every col appearing in BOTH lists (which is any trace/interaction
//   col — EVERY column of the z_next list), we have a (b, c) pair in the z
//   coefficients and a corresponding (b', c') pair in the z_next coefficients.
//   Columns in z but not z_next are the 4 AIR quotient cols (indices
//   N_COLS..N_COLS+4).
//
// Since the two lists share a prefix structure on the caller side
// (col_idx_z = 0..N_COLS+4+12, col_idx_zn = trace + interaction skipping
// quotient), we instead take a SIMPLE approach: iterate over the union by
// iterating col_indices_z; for each col, also check if it's in z_next's set
// and add its term. To avoid a second lookup, the caller pre-computes a
// zn_coeff_idx array of length n_cols_z where zn_coeff_idx[i] = the index
// of col_indices_z[i] within col_indices_zn, or UINT32_MAX if it's not
// present (only the quotient cols will have UINT32_MAX).

// Upper bound on columns per batch. The VortexSTARK main prove runs
// with n_cols_z = N_COLS + 4 + 12 = 50 at most. 128 gives headroom
// for future additions without changing the shmem layout.
#define ACC_MAX_COLS 128

__global__ void accumulate_numerators_dual_kernel(
    const uint32_t* const* __restrict__ col_ptrs,
    const uint32_t* __restrict__ col_indices_z,
    const uint32_t* __restrict__ zn_coeff_idx,
    const uint32_t* __restrict__ b_coeffs_z,
    const uint32_t* __restrict__ c_coeffs_z,
    const uint32_t* __restrict__ b_coeffs_zn,
    const uint32_t* __restrict__ c_coeffs_zn,
    uint32_t n_cols_z,
    uint32_t n_rows,
    uint32_t* __restrict__ out_z0, uint32_t* __restrict__ out_z1,
    uint32_t* __restrict__ out_z2, uint32_t* __restrict__ out_z3,
    uint32_t* __restrict__ out_zn0, uint32_t* __restrict__ out_zn1,
    uint32_t* __restrict__ out_zn2, uint32_t* __restrict__ out_zn3
) {
    // Stage per-iteration metadata in shared memory. Every thread in
    // the block uses the SAME col_indices / zn_coeff_idx / (b, c)
    // values for iteration i; loading them once from global memory
    // and reading from shared on the hot path saves ~96 cached loads
    // per thread per row. At 128M rows that's a 12B-load reduction.
    __shared__ const uint32_t* s_col_ptrs[ACC_MAX_COLS];
    __shared__ uint32_t s_col_indices[ACC_MAX_COLS];
    __shared__ uint32_t s_zn_coeff_idx[ACC_MAX_COLS];
    __shared__ uint32_t s_bc_z[ACC_MAX_COLS * 8];   // interleaved (b0,b1,b2,b3,c0,c1,c2,c3)
    __shared__ uint32_t s_bc_zn[ACC_MAX_COLS * 8];

    // Cooperative load: threads in the block split the work.
    uint32_t tid = threadIdx.x;
    if (tid < n_cols_z) {
        uint32_t ci = col_indices_z[tid];
        s_col_indices[tid] = ci;
        s_col_ptrs[tid] = col_ptrs[ci];
        s_zn_coeff_idx[tid] = zn_coeff_idx[tid];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            s_bc_z[tid * 8 + k]     = b_coeffs_z[tid * 4 + k];
            s_bc_z[tid * 8 + 4 + k] = c_coeffs_z[tid * 4 + k];
        }
        uint32_t zi = s_zn_coeff_idx[tid];
        if (zi != 0xFFFFFFFFu) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                s_bc_zn[tid * 8 + k]     = b_coeffs_zn[zi * 4 + k];
                s_bc_zn[tid * 8 + 4 + k] = c_coeffs_zn[zi * 4 + k];
            }
        }
    }
    __syncthreads();

    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    QM31 acc_z  = qm31_zero();
    QM31 acc_zn = qm31_zero();

    for (uint32_t i = 0; i < n_cols_z; i++) {
        uint32_t f_val = s_col_ptrs[i][row];

        // z contribution
        QM31 b_z = {{s_bc_z[i*8+0], s_bc_z[i*8+1], s_bc_z[i*8+2], s_bc_z[i*8+3]}};
        QM31 c_z = {{s_bc_z[i*8+4], s_bc_z[i*8+5], s_bc_z[i*8+6], s_bc_z[i*8+7]}};
        acc_z = qm31_add(acc_z, qm31_sub(qm31_mul_m31(c_z, f_val), b_z));

        // z_next contribution — branch predictable (same for all threads)
        uint32_t zn_i = s_zn_coeff_idx[i];
        if (zn_i != 0xFFFFFFFFu) {
            QM31 b_zn = {{s_bc_zn[i*8+0], s_bc_zn[i*8+1], s_bc_zn[i*8+2], s_bc_zn[i*8+3]}};
            QM31 c_zn = {{s_bc_zn[i*8+4], s_bc_zn[i*8+5], s_bc_zn[i*8+6], s_bc_zn[i*8+7]}};
            acc_zn = qm31_add(acc_zn, qm31_sub(qm31_mul_m31(c_zn, f_val), b_zn));
        }
    }

    out_z0[row]  = acc_z.v[0];
    out_z1[row]  = acc_z.v[1];
    out_z2[row]  = acc_z.v[2];
    out_z3[row]  = acc_z.v[3];
    out_zn0[row] = acc_zn.v[0];
    out_zn1[row] = acc_zn.v[1];
    out_zn2[row] = acc_zn.v[2];
    out_zn3[row] = acc_zn.v[3];
}

// ═══════════════════════════════════════════════════════════════════════════
//  compute_quotients_and_combine kernel
// ═══════════════════════════════════════════════════════════════════════════
//
// For each row in the lifting domain:
//   domain_point = domain.at(bit_reverse_index(row, log_size))
//   quotient[row] = sum_j (numer_j[lifted_idx] - a_acc_j * y) * den_inv_j
//
// where den_inv_j is computed from (sample_point_j, domain_point).
//
// Domain points on the circle: initial_x, initial_y define the canonic coset,
// and we bit-reverse-index into it.

// Helper: bit-reverse an index
__device__ __forceinline__ uint32_t bit_reverse(uint32_t val, uint32_t log_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log_n; i++) {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    return result;
}

// Circle domain point computation:
// CanonicCoset::new(log_size).circle_domain() generates points on the circle.
// The i-th point (natural order) of circle_domain is:
//   G^(2*i+1) where G is the circle group generator of order 2^(log_size+1)
// For bit-reversed order, we bit-reverse the index first.
//
// But the caller precomputes and passes the domain x,y coordinates.

__global__ void compute_quotients_combine_kernel(
    // Per-accumulation data (flattened, n_accs accumulations)
    const uint32_t* __restrict__ sample_points_x,  // [n_accs * 4] QM31
    const uint32_t* __restrict__ sample_points_y,  // [n_accs * 4] QM31
    const uint32_t* __restrict__ first_linear_acc,  // [n_accs * 4] QM31
    // Partial numerators for each acc: SoA, each of size = (1 << acc_log_sizes[j])
    const uint32_t* const* __restrict__ numer_ptrs0,  // [n_accs] pointers
    const uint32_t* const* __restrict__ numer_ptrs1,
    const uint32_t* const* __restrict__ numer_ptrs2,
    const uint32_t* const* __restrict__ numer_ptrs3,
    const uint32_t* __restrict__ acc_log_sizes,  // [n_accs] log2 of each acc's size
    uint32_t n_accs,
    // Domain info
    const uint32_t* __restrict__ domain_xs,  // [n_rows] M31 x-coordinates
    const uint32_t* __restrict__ domain_ys,  // [n_rows] M31 y-coordinates
    uint32_t lifting_log_size,
    uint32_t n_rows,
    // Output SoA QM31
    uint32_t* __restrict__ out0,
    uint32_t* __restrict__ out1,
    uint32_t* __restrict__ out2,
    uint32_t* __restrict__ out3
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    uint32_t dx = domain_xs[row];
    uint32_t dy = domain_ys[row];

    QM31 quotient = qm31_zero();

    for (uint32_t j = 0; j < n_accs; j++) {
        // Load sample point
        QM31 sp_x, sp_y;
        sp_x.v[0] = sample_points_x[j * 4 + 0];
        sp_x.v[1] = sample_points_x[j * 4 + 1];
        sp_x.v[2] = sample_points_x[j * 4 + 2];
        sp_x.v[3] = sample_points_x[j * 4 + 3];
        sp_y.v[0] = sample_points_y[j * 4 + 0];
        sp_y.v[1] = sample_points_y[j * 4 + 1];
        sp_y.v[2] = sample_points_y[j * 4 + 2];
        sp_y.v[3] = sample_points_y[j * 4 + 3];

        // Denominator: (Re(px) - dx)*Im(py) - (Re(py) - dy)*Im(px)
        // where Re = CM31.a (.v[0]) and Im = CM31.b (.v[1])
        // sp_x = CM31(v[0], v[1]) + CM31(v[2], v[3])*u
        // Re(sp_x) = CM31(v[0], v[1]), Im(sp_x) = CM31(v[2], v[3])
        CM31 prx = {sp_x.v[0], sp_x.v[1]};  // Re(sp_x)
        CM31 pry = {sp_y.v[0], sp_y.v[1]};  // Re(sp_y)
        CM31 pix = {sp_x.v[2], sp_x.v[3]};  // Im(sp_x)
        CM31 piy = {sp_y.v[2], sp_y.v[3]};  // Im(sp_y)

        // (prx - dx) * piy - (pry - dy) * pix
        CM31 term1 = cm31_mul(cm31_sub(prx, {dx, 0}), piy);
        CM31 term2 = cm31_mul(cm31_sub(pry, {dy, 0}), pix);
        CM31 denom = cm31_sub(term1, term2);
        CM31 den_inv = cm31_inv(denom);

        // Lifted index
        uint32_t acc_log_sz = acc_log_sizes[j];
        uint32_t log_ratio = lifting_log_size - acc_log_sz;
        uint32_t lifted_idx = (row >> (log_ratio + 1) << 1) + (row & 1);

        // Load partial numerator at lifted_idx
        QM31 partial_numer;
        partial_numer.v[0] = numer_ptrs0[j][lifted_idx];
        partial_numer.v[1] = numer_ptrs1[j][lifted_idx];
        partial_numer.v[2] = numer_ptrs2[j][lifted_idx];
        partial_numer.v[3] = numer_ptrs3[j][lifted_idx];

        // Load first_linear_term_acc
        QM31 a_acc;
        a_acc.v[0] = first_linear_acc[j * 4 + 0];
        a_acc.v[1] = first_linear_acc[j * 4 + 1];
        a_acc.v[2] = first_linear_acc[j * 4 + 2];
        a_acc.v[3] = first_linear_acc[j * 4 + 3];

        // full_numerator = partial_numer - a_acc * domain_point.y
        // domain_point.y is M31
        QM31 full_numer = qm31_sub(partial_numer, qm31_mul_m31(a_acc, dy));

        // Multiply by den_inv (CM31): QM31 * CM31
        // (a + bu) * c = ac + bc*u
        CM31 fa = qm31_a(full_numer);
        CM31 fb = qm31_b(full_numer);
        CM31 ra = cm31_mul(fa, den_inv);
        CM31 rb = cm31_mul(fb, den_inv);
        QM31 term_result = qm31_from(ra, rb);

        quotient = qm31_add(quotient, term_result);
    }

    out0[row] = quotient.v[0];
    out1[row] = quotient.v[1];
    out2[row] = quotient.v[2];
    out3[row] = quotient.v[3];
}

extern "C" {

void cuda_accumulate_numerators(
    const uint32_t* const* col_ptrs,
    const uint32_t* col_indices,
    const uint32_t* b_coeffs,
    const uint32_t* c_coeffs,
    uint32_t n_batch_cols,
    uint32_t n_rows,
    uint32_t* out0,
    uint32_t* out1,
    uint32_t* out2,
    uint32_t* out3
) {
    if (n_rows == 0) return;
    uint32_t threads = 256;
    uint32_t blocks = (n_rows + threads - 1) / threads;
    accumulate_numerators_kernel<<<blocks, threads>>>(
        col_ptrs, col_indices, b_coeffs, c_coeffs,
        n_batch_cols, n_rows, out0, out1, out2, out3
    );
}

void cuda_accumulate_numerators_dual(
    const uint32_t* const* col_ptrs,
    const uint32_t* col_indices_z,
    const uint32_t* zn_coeff_idx,
    const uint32_t* b_coeffs_z,
    const uint32_t* c_coeffs_z,
    const uint32_t* b_coeffs_zn,
    const uint32_t* c_coeffs_zn,
    uint32_t n_cols_z,
    uint32_t n_rows,
    uint32_t* out_z0, uint32_t* out_z1, uint32_t* out_z2, uint32_t* out_z3,
    uint32_t* out_zn0, uint32_t* out_zn1, uint32_t* out_zn2, uint32_t* out_zn3
) {
    if (n_rows == 0) return;
    uint32_t threads = 256;
    uint32_t blocks = (n_rows + threads - 1) / threads;
    accumulate_numerators_dual_kernel<<<blocks, threads>>>(
        col_ptrs, col_indices_z, zn_coeff_idx,
        b_coeffs_z, c_coeffs_z, b_coeffs_zn, c_coeffs_zn,
        n_cols_z, n_rows,
        out_z0, out_z1, out_z2, out_z3,
        out_zn0, out_zn1, out_zn2, out_zn3
    );
}

void cuda_compute_quotients_combine(
    const uint32_t* sample_points_x,
    const uint32_t* sample_points_y,
    const uint32_t* first_linear_acc,
    const uint32_t* const* numer_ptrs0,
    const uint32_t* const* numer_ptrs1,
    const uint32_t* const* numer_ptrs2,
    const uint32_t* const* numer_ptrs3,
    const uint32_t* acc_log_sizes,
    uint32_t n_accs,
    const uint32_t* domain_xs,
    const uint32_t* domain_ys,
    uint32_t lifting_log_size,
    uint32_t n_rows,
    uint32_t* out0,
    uint32_t* out1,
    uint32_t* out2,
    uint32_t* out3
) {
    if (n_rows == 0) return;
    uint32_t threads = 256;
    uint32_t blocks = (n_rows + threads - 1) / threads;
    compute_quotients_combine_kernel<<<blocks, threads>>>(
        sample_points_x, sample_points_y, first_linear_acc,
        numer_ptrs0, numer_ptrs1, numer_ptrs2, numer_ptrs3,
        acc_log_sizes, n_accs,
        domain_xs, domain_ys, lifting_log_size, n_rows,
        out0, out1, out2, out3
    );
}

} // extern "C"
