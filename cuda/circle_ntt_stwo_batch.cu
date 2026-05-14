// Multi-column batched stwo-format NTT (v2).
//
// v2 vs v1: adapts ALL the single-poly path kernels (generic per-layer, radix-4,
// fused-shared-memory, circle, scale) to multi-column form via blockIdx.y for
// column dispatch. Host driver mirrors the single-poly cuda_stwo_ntt_evaluate /
// _interpolate phases exactly so behaviour is byte-identical, just batched
// over N columns at the same log_size.
//
// Each block now handles one (tile_or_group, column) pair: blockIdx.x picks
// the tile/group within a column, blockIdx.y picks the column. Threads within
// a block work the same as the single-poly kernel for their column.

#include "include/m31.cuh"

#define NTT_FUSED_LAYERS 10
#define NTT_FUSED_TILE (2u << NTT_FUSED_LAYERS)
#define NTT_FUSED_THREADS 512

// Butterflies (same as circle_ntt_stwo.cu)
__device__ __forceinline__ void bf_b(uint32_t& v0, uint32_t& v1, uint32_t t) {
    uint32_t tmp = m31_mul(v1, t);
    v1 = m31_sub(v0, tmp);
    v0 = m31_add(v0, tmp);
}
__device__ __forceinline__ void ibf_b(uint32_t& v0, uint32_t& v1, uint32_t t) {
    uint32_t tmp = v0;
    v0 = m31_add(tmp, v1);
    v1 = m31_mul(m31_sub(tmp, v1), t);
}

// ─── Batched generic per-layer kernel ──────────────────────────────────
__global__ void stwo_ntt_batch_layer_kernel(
    uint32_t* const* __restrict__ col_ptrs,
    const uint32_t* __restrict__ twiddle_ptr,
    uint32_t layer_idx,
    uint32_t half_n,
    int forward
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_n) return;

    uint32_t* data = col_ptrs[blockIdx.y];

    uint32_t stride = 1u << layer_idx;
    uint32_t h = tid >> layer_idx;
    uint32_t l = tid & (stride - 1);
    uint32_t idx0 = (h << (layer_idx + 1)) + l;
    uint32_t idx1 = idx0 + stride;

    uint32_t v0 = data[idx0];
    uint32_t v1 = data[idx1];
    uint32_t t = twiddle_ptr[h];

    if (forward) bf_b(v0, v1, t);
    else         ibf_b(v0, v1, t);

    data[idx0] = v0;
    data[idx1] = v1;
}

// ─── Batched radix-4 kernel (pairs of layers) ──────────────────────────
__global__ void stwo_ntt_batch_radix4_kernel(
    uint32_t* const* __restrict__ col_ptrs,
    const uint32_t* __restrict__ twid_hi,
    const uint32_t* __restrict__ twid_lo,
    uint32_t layer_idx_hi,
    uint32_t n_groups,
    int forward
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_groups) return;

    uint32_t* data = col_ptrs[blockIdx.y];

    uint32_t S = 1u << layer_idx_hi;
    uint32_t S2 = S >> 1;

    uint32_t super_idx = gid / S2;
    uint32_t intra = gid % S2;
    uint32_t base = super_idx * (2u * S) + intra;

    uint32_t v0 = data[base];
    uint32_t v1 = data[base + S2];
    uint32_t v2 = data[base + S];
    uint32_t v3 = data[base + S + S2];

    uint32_t t_hi  = twid_hi[super_idx];
    uint32_t t_lo0 = twid_lo[super_idx * 2];
    uint32_t t_lo1 = twid_lo[super_idx * 2 + 1];

    if (forward) {
        bf_b(v0, v2, t_hi);
        bf_b(v1, v3, t_hi);
        bf_b(v0, v1, t_lo0);
        bf_b(v2, v3, t_lo1);
    } else {
        ibf_b(v0, v1, t_lo0);
        ibf_b(v2, v3, t_lo1);
        ibf_b(v0, v2, t_hi);
        ibf_b(v1, v3, t_hi);
    }

    data[base] = v0;
    data[base + S2] = v1;
    data[base + S] = v2;
    data[base + S + S2] = v3;
}

// ─── Batched fused shared-memory kernel ────────────────────────────────
__global__ void stwo_ntt_batch_fused_kernel(
    uint32_t* const* __restrict__ col_ptrs,
    const uint32_t* __restrict__ d_twiddles,
    uint32_t half_n,
    uint32_t n_line_layers,
    int forward,
    uint32_t first_layer_idx,
    uint32_t last_layer_idx
) {
    extern __shared__ uint32_t smem[];

    const uint32_t tile_size = 2u << last_layer_idx;
    const uint32_t n = half_n * 2;
    const uint32_t tile_base = blockIdx.x * tile_size;
    const uint32_t n_bflies = tile_size / 2;

    uint32_t* data = col_ptrs[blockIdx.y];

    for (uint32_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
        uint32_t gi = tile_base + i;
        smem[i] = (gi < n) ? data[gi] : 0u;
    }
    __syncthreads();

    if (forward) {
        for (uint32_t layer_idx = last_layer_idx; ; layer_idx--) {
            uint32_t stride = 1u << layer_idx;
            uint32_t k = layer_idx - 1;
            const uint32_t* twid_ptr = d_twiddles + (half_n - (1u << (n_line_layers - k)));
            uint32_t base_h = tile_base >> (layer_idx + 1);

            for (uint32_t i = threadIdx.x; i < n_bflies; i += blockDim.x) {
                uint32_t h = i >> layer_idx;
                uint32_t l = i & (stride - 1);
                uint32_t idx0 = (h << (layer_idx + 1)) + l;
                uint32_t idx1 = idx0 + stride;
                uint32_t t = twid_ptr[base_h + h];
                bf_b(smem[idx0], smem[idx1], t);
            }
            __syncthreads();
            if (layer_idx == first_layer_idx) break;
        }
    } else {
        for (uint32_t layer_idx = first_layer_idx; layer_idx <= last_layer_idx; layer_idx++) {
            uint32_t stride = 1u << layer_idx;
            uint32_t k = layer_idx - 1;
            const uint32_t* twid_ptr = d_twiddles + (half_n - (1u << (n_line_layers - k)));
            uint32_t base_h = tile_base >> (layer_idx + 1);

            for (uint32_t i = threadIdx.x; i < n_bflies; i += blockDim.x) {
                uint32_t h = i >> layer_idx;
                uint32_t l = i & (stride - 1);
                uint32_t idx0 = (h << (layer_idx + 1)) + l;
                uint32_t idx1 = idx0 + stride;
                uint32_t t = twid_ptr[base_h + h];
                ibf_b(smem[idx0], smem[idx1], t);
            }
            __syncthreads();
        }
    }

    for (uint32_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
        uint32_t gi = tile_base + i;
        if (gi < n) data[gi] = smem[i];
    }
}

// ─── Batched circle-layer kernel ───────────────────────────────────────
__global__ void stwo_circle_batch_layer_kernel(
    uint32_t* const* __restrict__ col_ptrs,
    const uint32_t* __restrict__ first_line_layer,
    uint32_t half_n,
    int forward
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half_n) return;

    uint32_t* data = col_ptrs[blockIdx.y];

    uint32_t pair_idx = tid / 4;
    uint32_t sub_idx = tid & 3u;
    uint32_t x = first_line_layer[pair_idx * 2];
    uint32_t y = first_line_layer[pair_idx * 2 + 1];

    uint32_t t;
    switch (sub_idx) {
        case 0: t = y; break;
        case 1: t = m31_neg(y); break;
        case 2: t = m31_neg(x); break;
        case 3: t = x; break;
    }

    uint32_t idx0 = tid * 2;
    uint32_t idx1 = idx0 + 1;

    uint32_t v0 = data[idx0];
    uint32_t v1 = data[idx1];

    if (forward) bf_b(v0, v1, t);
    else         ibf_b(v0, v1, t);

    data[idx0] = v0;
    data[idx1] = v1;
}

// ─── Batched scale kernel ──────────────────────────────────────────────
__global__ void stwo_batch_scale_kernel(
    uint32_t* const* __restrict__ col_ptrs,
    uint32_t scale,
    uint32_t n
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t* data = col_ptrs[blockIdx.y];
    data[tid] = m31_mul(data[tid], scale);
}

namespace {
uint32_t** upload_col_ptrs(uint32_t* const* col_ptrs, uint32_t n_cols) {
    uint32_t** d_ptrs = nullptr;
    cudaMalloc(&d_ptrs, n_cols * sizeof(uint32_t*));
    cudaMemcpy(d_ptrs, col_ptrs, n_cols * sizeof(uint32_t*), cudaMemcpyHostToDevice);
    return d_ptrs;
}
}  // namespace

extern "C" {

// Forward NTT (evaluate). Mirrors cuda_stwo_ntt_evaluate's phase ordering.
void cuda_stwo_ntt_batch_evaluate(
    uint32_t* const* col_ptrs,
    const uint32_t* d_twiddles,
    uint32_t n,
    uint32_t n_cols
) {
    if (n_cols == 0 || n < 2) return;
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks_x = (half_n + threads - 1) / threads;
    uint32_t log_n = 0;
    for (uint32_t tmp = n; tmp > 1; tmp >>= 1) log_n++;
    uint32_t n_line_layers = log_n - 1;

    uint32_t** d_ptrs = upload_col_ptrs(col_ptrs, n_cols);
    dim3 grid_layer(blocks_x, n_cols);

    {
        int k = (int)n_line_layers - 1;
        int n_unfused = k - (int)NTT_FUSED_LAYERS + 1;
        if (n_unfused % 2 == 1) {
            uint32_t twid_offset = half_n - (1u << (n_line_layers - k));
            stwo_ntt_batch_layer_kernel<<<grid_layer, threads>>>(
                d_ptrs, d_twiddles + twid_offset, (uint32_t)(k + 1), half_n, 1
            );
            k--;
        }
        uint32_t n_groups = half_n / 2;
        uint32_t r4_blocks = (n_groups + threads - 1) / threads;
        dim3 grid_r4(r4_blocks, n_cols);
        while (k >= (int)NTT_FUSED_LAYERS + 1) {
            const uint32_t* twid_hi = d_twiddles + (half_n - (1u << (n_line_layers - k)));
            const uint32_t* twid_lo = d_twiddles + (half_n - (1u << (n_line_layers - (k - 1))));
            stwo_ntt_batch_radix4_kernel<<<grid_r4, threads>>>(
                d_ptrs, twid_hi, twid_lo, (uint32_t)(k + 1), n_groups, 1
            );
            k -= 2;
        }
        if (k >= (int)NTT_FUSED_LAYERS) {
            uint32_t twid_offset = half_n - (1u << (n_line_layers - k));
            stwo_ntt_batch_layer_kernel<<<grid_layer, threads>>>(
                d_ptrs, d_twiddles + twid_offset, (uint32_t)(k + 1), half_n, 1
            );
        }
    }

    if (n_line_layers > 0 && NTT_FUSED_LAYERS > 0) {
        uint32_t fused_count = (n_line_layers < NTT_FUSED_LAYERS) ? n_line_layers : NTT_FUSED_LAYERS;
        uint32_t last_li = fused_count;
        uint32_t first_li = 1;
        uint32_t tile_size = 2u << last_li;
        uint32_t fused_blocks = n / tile_size;
        uint32_t smem_bytes = tile_size * sizeof(uint32_t);
        dim3 grid_fused(fused_blocks, n_cols);
        stwo_ntt_batch_fused_kernel<<<grid_fused, NTT_FUSED_THREADS, smem_bytes>>>(
            d_ptrs, d_twiddles, half_n, n_line_layers, 1, first_li, last_li
        );
    }

    stwo_circle_batch_layer_kernel<<<grid_layer, threads>>>(
        d_ptrs, d_twiddles, half_n, 1
    );

    cudaDeviceSynchronize();
    cudaFree(d_ptrs);
}

// Inverse NTT (interpolate). Mirrors cuda_stwo_ntt_interpolate's phase ordering
// including the (30 * log_n) % 31 inverse-n exponent (NOT (31 - log_n)).
void cuda_stwo_ntt_batch_interpolate(
    uint32_t* const* col_ptrs,
    const uint32_t* d_itwiddles,
    uint32_t n,
    uint32_t n_cols
) {
    if (n_cols == 0 || n < 2) return;
    uint32_t half_n = n / 2;
    uint32_t threads = 256;
    uint32_t blocks_x = (half_n + threads - 1) / threads;
    uint32_t log_n = 0;
    for (uint32_t tmp = n; tmp > 1; tmp >>= 1) log_n++;
    uint32_t n_line_layers = log_n - 1;

    uint32_t** d_ptrs = upload_col_ptrs(col_ptrs, n_cols);
    dim3 grid_layer(blocks_x, n_cols);

    stwo_circle_batch_layer_kernel<<<grid_layer, threads>>>(
        d_ptrs, d_itwiddles, half_n, 0
    );

    if (n_line_layers > 0 && NTT_FUSED_LAYERS > 0) {
        uint32_t fused_count = (n_line_layers < NTT_FUSED_LAYERS) ? n_line_layers : NTT_FUSED_LAYERS;
        uint32_t last_li = fused_count;
        uint32_t first_li = 1;
        uint32_t tile_size = 2u << last_li;
        uint32_t fused_blocks = n / tile_size;
        uint32_t smem_bytes = tile_size * sizeof(uint32_t);
        dim3 grid_fused(fused_blocks, n_cols);
        stwo_ntt_batch_fused_kernel<<<grid_fused, NTT_FUSED_THREADS, smem_bytes>>>(
            d_ptrs, d_itwiddles, half_n, n_line_layers, 0, first_li, last_li
        );
    }

    {
        uint32_t k = NTT_FUSED_LAYERS;
        uint32_t n_groups = half_n / 2;
        uint32_t r4_blocks = (n_groups + threads - 1) / threads;
        dim3 grid_r4(r4_blocks, n_cols);
        while (k + 1 < n_line_layers) {
            const uint32_t* twid_lo = d_itwiddles + (half_n - (1u << (n_line_layers - k)));
            const uint32_t* twid_hi = d_itwiddles + (half_n - (1u << (n_line_layers - (k + 1))));
            stwo_ntt_batch_radix4_kernel<<<grid_r4, threads>>>(
                d_ptrs, twid_hi, twid_lo, k + 2, n_groups, 0
            );
            k += 2;
        }
        if (k < n_line_layers) {
            uint32_t twid_offset = half_n - (1u << (n_line_layers - k));
            stwo_ntt_batch_layer_kernel<<<grid_layer, threads>>>(
                d_ptrs, d_itwiddles + twid_offset, k + 1, half_n, 0
            );
        }
    }

    // Scale by 1/n. inv_n exponent is (30 * log_n) % 31, NOT (31 - log_n).
    uint32_t exp = (30u * log_n) % 31u;
    uint32_t inv_n = (exp == 0) ? 1u : (1u << exp);
    uint32_t scale_blocks = (n + threads - 1) / threads;
    dim3 grid_scale(scale_blocks, n_cols);
    stwo_batch_scale_kernel<<<grid_scale, threads>>>(d_ptrs, inv_n, n);

    cudaDeviceSynchronize();
    cudaFree(d_ptrs);
}

}  // extern "C"
