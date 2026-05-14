// Multi-column batched stwo-format NTT. Adapts circle_ntt_stwo.cu's generic
// per-layer kernel to process N polynomials of the same size in a single
// kernel launch. Eliminates per-polynomial launch overhead (~20us × log_n
// per poly) when stwo's evaluate_polynomials drives many same-size polys.
//
// v1 limitation: uses ONLY the generic per-layer kernel for every layer.
// Skips the radix-4 and fused-shared-memory optimizations that the single-
// poly path uses. Correct, just not as fast per-poly. Wins on batched
// launch-overhead amortization when N is large enough.

#include "include/m31.cuh"

// Butterfly (must match circle_ntt_stwo.cu)
__device__ __forceinline__ void butterfly_batch(uint32_t& v0, uint32_t& v1, uint32_t t) {
    uint32_t tmp = m31_mul(v1, t);
    v1 = m31_sub(v0, tmp);
    v0 = m31_add(v0, tmp);
}

__device__ __forceinline__ void ibutterfly_batch(uint32_t& v0, uint32_t& v1, uint32_t t) {
    uint32_t tmp = v0;
    v0 = m31_add(tmp, v1);
    v1 = m31_mul(m31_sub(tmp, v1), t);
}

// Batched line-layer kernel. Each thread handles one butterfly slot of one
// column. tid = col_idx * half_n + lane.
__global__ void stwo_ntt_batch_layer_kernel(
    uint32_t* const* __restrict__ col_ptrs,
    const uint32_t* __restrict__ twiddle_ptr,
    uint32_t layer_idx,
    uint32_t half_n,
    uint32_t n_cols,
    int forward
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = half_n * n_cols;
    if (tid >= total) return;

    uint32_t col_idx = tid / half_n;
    uint32_t lane = tid - col_idx * half_n;  // tid % half_n

    uint32_t stride = 1u << layer_idx;
    uint32_t h = lane >> layer_idx;
    uint32_t l = lane & (stride - 1);
    uint32_t idx0 = (h << (layer_idx + 1)) + l;
    uint32_t idx1 = idx0 + stride;

    uint32_t* data = col_ptrs[col_idx];

    uint32_t v0 = data[idx0];
    uint32_t v1 = data[idx1];
    uint32_t t = twiddle_ptr[h];

    if (forward) {
        butterfly_batch(v0, v1, t);
    } else {
        ibutterfly_batch(v0, v1, t);
    }

    data[idx0] = v0;
    data[idx1] = v1;
}

// Batched circle-layer kernel (final layer of forward / first layer of inverse).
// Twiddle comes from first_line_layer pairs in stwo's flat buffer.
__global__ void stwo_circle_batch_layer_kernel(
    uint32_t* const* __restrict__ col_ptrs,
    const uint32_t* __restrict__ first_line_layer,
    uint32_t half_n,
    uint32_t n_cols,
    int forward
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = half_n * n_cols;
    if (tid >= total) return;

    uint32_t col_idx = tid / half_n;
    uint32_t lane = tid - col_idx * half_n;  // tid % half_n

    uint32_t pair_idx = lane / 4;
    uint32_t sub_idx = lane & 3u;

    uint32_t x = first_line_layer[pair_idx * 2];
    uint32_t y = first_line_layer[pair_idx * 2 + 1];

    uint32_t t;
    switch (sub_idx) {
        case 0: t = y; break;
        case 1: t = m31_neg(y); break;
        case 2: t = m31_neg(x); break;
        case 3: t = x; break;
    }

    uint32_t idx0 = lane * 2;
    uint32_t idx1 = idx0 + 1;

    uint32_t* data = col_ptrs[col_idx];

    uint32_t v0 = data[idx0];
    uint32_t v1 = data[idx1];

    if (forward) {
        butterfly_batch(v0, v1, t);
    } else {
        ibutterfly_batch(v0, v1, t);
    }

    data[idx0] = v0;
    data[idx1] = v1;
}

// Batched scale kernel. Multiply each element of each column by `scale`.
// Used after inverse NTT for 1/n normalization.
__global__ void stwo_batch_scale_kernel(
    uint32_t* const* __restrict__ col_ptrs,
    uint32_t scale,
    uint32_t n,
    uint32_t n_cols
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n * n_cols;
    if (tid >= total) return;
    uint32_t col_idx = tid / n;
    uint32_t lane = tid - col_idx * n;
    uint32_t* data = col_ptrs[col_idx];
    data[lane] = m31_mul(data[lane], scale);
}

namespace {
// Upload a host array of N column pointers to device. Caller cudaFrees.
uint32_t** upload_col_ptrs(uint32_t* const* col_ptrs, uint32_t n_cols) {
    uint32_t** d_ptrs = nullptr;
    cudaMalloc(&d_ptrs, n_cols * sizeof(uint32_t*));
    cudaMemcpy(d_ptrs, col_ptrs, n_cols * sizeof(uint32_t*), cudaMemcpyHostToDevice);
    return d_ptrs;
}
}  // namespace

extern "C" {

// Forward NTT (evaluate): coefficients -> values in bit-reversed order.
// All N columns must have the same size `n` and use the same twiddle buffer.
void cuda_stwo_ntt_batch_evaluate(
    uint32_t* const* col_ptrs,
    const uint32_t* d_twiddles,
    uint32_t n,
    uint32_t n_cols
) {
    if (n_cols == 0 || n < 2) return;
    uint32_t half_n = n / 2;
    uint32_t log_n = 0;
    for (uint32_t tmp = n; tmp > 1; tmp >>= 1) log_n++;
    uint32_t n_line_layers = log_n - 1;

    uint32_t total_threads = half_n * n_cols;
    uint32_t threads = 256;
    uint32_t blocks = (total_threads + threads - 1) / threads;

    uint32_t** d_ptrs = upload_col_ptrs(col_ptrs, n_cols);

    // Forward: line layers from highest k to lowest k. Generic per-layer kernel
    // for every layer (no radix-4, no fused). Each layer's twiddles start at
    // offset (half_n - (1 << (n_line_layers - k))) in the flat buffer.
    for (int k = (int)n_line_layers - 1; k >= 0; k--) {
        uint32_t twid_offset = half_n - (1u << (n_line_layers - k));
        stwo_ntt_batch_layer_kernel<<<blocks, threads>>>(
            d_ptrs, d_twiddles + twid_offset, (uint32_t)(k + 1),
            half_n, n_cols, 1
        );
    }

    // Final circle layer.
    stwo_circle_batch_layer_kernel<<<blocks, threads>>>(
        d_ptrs, d_twiddles, half_n, n_cols, 1
    );

    cudaFree(d_ptrs);
}

// Inverse NTT (interpolate): values (bit-reversed) -> coefficients with 1/n scaling.
void cuda_stwo_ntt_batch_interpolate(
    uint32_t* const* col_ptrs,
    const uint32_t* d_itwiddles,
    uint32_t n,
    uint32_t n_cols
) {
    if (n_cols == 0 || n < 2) return;
    uint32_t half_n = n / 2;
    uint32_t log_n = 0;
    for (uint32_t tmp = n; tmp > 1; tmp >>= 1) log_n++;
    uint32_t n_line_layers = log_n - 1;

    uint32_t total_threads = half_n * n_cols;
    uint32_t threads = 256;
    uint32_t blocks = (total_threads + threads - 1) / threads;

    uint32_t** d_ptrs = upload_col_ptrs(col_ptrs, n_cols);

    // Inverse: circle layer first.
    stwo_circle_batch_layer_kernel<<<blocks, threads>>>(
        d_ptrs, d_itwiddles, half_n, n_cols, 0
    );

    // Then line layers from lowest k to highest k.
    for (int k = 0; k < (int)n_line_layers; k++) {
        uint32_t twid_offset = half_n - (1u << (n_line_layers - k));
        stwo_ntt_batch_layer_kernel<<<blocks, threads>>>(
            d_ptrs, d_itwiddles + twid_offset, (uint32_t)(k + 1),
            half_n, n_cols, 0
        );
    }

    // Final 1/n scaling.
    uint32_t scale_total = n * n_cols;
    uint32_t scale_blocks = (scale_total + threads - 1) / threads;
    // 1/n in M31. n is a power of 2 so use mod-inverse: inv_n = 2^(31 - log_n) % (2^31 - 1)
    // For M31 (prime p = 2^31 - 1), n^(-1) mod p; for n = 2^k, inv = 2^(31-k) mod p.
    uint32_t inv_n = (1u << (31 - log_n));
    stwo_batch_scale_kernel<<<scale_blocks, threads>>>(
        d_ptrs, inv_n, n, n_cols
    );

    cudaFree(d_ptrs);
}

}  // extern "C"
