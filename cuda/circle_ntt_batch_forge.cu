// FORGE-emitted batched Circle NTT (multi-column SoA) + host shims.
// Source: forge/analysis/vortex_ntt/circle_ntt_batch.fg (145 obligations).
// 3 kernels: forward butterfly, inverse butterfly, m31_batch_scale.
//
// Replaces `cuda_circle_ntt_evaluate_batch` and `_interpolate_batch`
// from cuda/circle_ntt.cu when the `forge-ntt-batch` feature is on.
//
// The FORGE kernel takes `forge_span_span_u32_t columns` =
// `{ forge_span_u32_t* data; uintptr_t len }`. Each inner span is a
// fat-pointer struct living on device. Host code allocates a small
// device buffer of `forge_span_u32_t[n_cols]`, fills each entry with
// the column's device pointer + len, then passes the array pointer
// + n_cols as the outer span's data + len.
//
// Phase 4 (bitwise column demux): the kernels now take `log_half_n`
// (resp. `log_n` for scale) so the column demux uses shifts and bit-
// ands instead of div/mod. Host computes log via __builtin_ctz on the
// power-of-two NTT size.

#include <vector>
#include "forge/circle_ntt_batch_forge.cu"

namespace {
// Build a device-resident array of `forge_span_u32_t` from host
// pointer/len pairs, returns the device pointer (caller cudaFrees).
forge_span_u32_t* upload_col_spans(uint32_t* const* col_ptrs,
                                    uint32_t n_cols, uint32_t n) {
    std::vector<forge_span_u32_t> host_spans(n_cols);
    for (uint32_t c = 0; c < n_cols; c++) {
        host_spans[c].data = col_ptrs[c];
        host_spans[c].len  = (uintptr_t)n;
    }
    forge_span_u32_t* d_spans = nullptr;
    cudaMalloc(&d_spans, n_cols * sizeof(forge_span_u32_t));
    cudaMemcpy(d_spans, host_spans.data(),
               n_cols * sizeof(forge_span_u32_t),
               cudaMemcpyHostToDevice);
    return d_spans;
}
}  // namespace

extern "C" void cuda_circle_ntt_batch_layer_forge(
    uint32_t* const* col_ptrs,
    const uint32_t* twiddles,
    uint32_t layer_idx,
    uint32_t n,
    uint32_t n_cols,
    int forward
) {
    if (n_cols == 0 || n < 2) return;
    uint32_t half_n = n / 2;
    uint64_t log_half_n = (uint64_t)__builtin_ctz(half_n);
    forge_span_u32_t* d_spans = upload_col_spans(col_ptrs, n_cols, n);

    forge_span_span_u32_t outer = { d_spans, (uintptr_t)n_cols };
    // Twiddle len: for circle layer (layer_idx=0) it's n_circle; for
    // line layers it's at most half_n. The kernel guards via
    // `if h < twiddles.len`; we pass half_n which covers all cases.
    forge_span_u32_t s_tw = { const_cast<uint32_t*>(twiddles), (uintptr_t)half_n };

    uint32_t total = half_n * n_cols;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;
    if (forward) {
        circle_ntt_batch_layer_forward<<<blocks, threads>>>(
            outer, s_tw, layer_idx, (uint64_t)half_n, log_half_n, (uint64_t)n_cols);
    } else {
        circle_ntt_batch_layer_inverse<<<blocks, threads>>>(
            outer, s_tw, layer_idx, (uint64_t)half_n, log_half_n, (uint64_t)n_cols);
    }
    cudaFree(d_spans);
}

extern "C" void cuda_m31_batch_scale_forge(
    uint32_t* const* col_ptrs,
    uint32_t scale,
    uint32_t n,
    uint32_t n_cols
) {
    if (n_cols == 0 || n == 0) return;
    uint64_t log_n = (uint64_t)__builtin_ctz(n);
    forge_span_u32_t* d_spans = upload_col_spans(col_ptrs, n_cols, n);
    forge_span_span_u32_t outer = { d_spans, (uintptr_t)n_cols };
    uint32_t total = n * n_cols;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;
    m31_batch_scale<<<blocks, threads>>>(
        outer, scale, (uint64_t)n, log_n, (uint64_t)n_cols);
    cudaFree(d_spans);
}

// Aggregate: full forward NTT across all line layers + the circle
// layer. Same input/output ABI as `cuda_circle_ntt_evaluate_batch`
// (cuda/circle_ntt.cu) — drop-in replacement when forge-ntt is on.
extern "C" void cuda_circle_ntt_evaluate_batch_forge(
    uint32_t* const* col_ptrs,
    const uint32_t* d_twiddles,
    const uint32_t* d_circle_twids,
    const uint32_t* h_layer_offsets,
    const uint32_t* /*h_layer_sizes*/,
    uint32_t n_line_layers,
    uint32_t n,
    uint32_t n_cols
) {
    if (n_cols == 0 || n < 2) return;
    uint32_t half_n = n / 2;
    uint64_t log_half_n = (uint64_t)__builtin_ctz(half_n);
    forge_span_u32_t* d_spans = upload_col_spans(col_ptrs, n_cols, n);
    forge_span_span_u32_t outer = { d_spans, (uintptr_t)n_cols };

    uint32_t total = half_n * n_cols;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;

    // Line layers (highest to lowest)
    for (int layer = (int)n_line_layers - 1; layer >= 0; layer--) {
        forge_span_u32_t s_tw = {
            const_cast<uint32_t*>(d_twiddles + h_layer_offsets[layer]),
            (uintptr_t)half_n
        };
        circle_ntt_batch_layer_forward<<<blocks, threads>>>(
            outer, s_tw, (uint32_t)(layer + 1),
            (uint64_t)half_n, log_half_n, (uint64_t)n_cols);
    }
    // Circle layer (layer_idx = 0)
    forge_span_u32_t s_tw_circle = {
        const_cast<uint32_t*>(d_circle_twids), (uintptr_t)half_n
    };
    circle_ntt_batch_layer_forward<<<blocks, threads>>>(
        outer, s_tw_circle, 0u, (uint64_t)half_n, log_half_n, (uint64_t)n_cols);

    cudaDeviceSynchronize();
    cudaFree(d_spans);
}

extern "C" void cuda_circle_ntt_interpolate_batch_forge(
    uint32_t* const* col_ptrs,
    const uint32_t* d_itwiddles,
    const uint32_t* d_circle_itwids,
    const uint32_t* h_layer_offsets,
    const uint32_t* /*h_layer_sizes*/,
    uint32_t n_line_layers,
    uint32_t n,
    uint32_t n_cols
) {
    if (n_cols == 0 || n < 2) return;
    uint32_t half_n = n / 2;
    uint64_t log_half_n = (uint64_t)__builtin_ctz(half_n);
    forge_span_u32_t* d_spans = upload_col_spans(col_ptrs, n_cols, n);
    forge_span_span_u32_t outer = { d_spans, (uintptr_t)n_cols };

    uint32_t total = half_n * n_cols;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;

    // Circle layer first (layer_idx = 0)
    forge_span_u32_t s_tw_circle = {
        const_cast<uint32_t*>(d_circle_itwids), (uintptr_t)half_n
    };
    circle_ntt_batch_layer_inverse<<<blocks, threads>>>(
        outer, s_tw_circle, 0u, (uint64_t)half_n, log_half_n, (uint64_t)n_cols);

    // Line layers (lowest to highest)
    for (uint32_t layer = 0; layer < n_line_layers; layer++) {
        forge_span_u32_t s_tw = {
            const_cast<uint32_t*>(d_itwiddles + h_layer_offsets[layer]),
            (uintptr_t)half_n
        };
        circle_ntt_batch_layer_inverse<<<blocks, threads>>>(
            outer, s_tw, layer + 1u,
            (uint64_t)half_n, log_half_n, (uint64_t)n_cols);
    }

    // Scale by 1/n: inv_n = 2^(30*log_n mod 31) in M31.
    uint32_t log_n = (uint32_t)__builtin_ctz(n);
    uint32_t exp = (30u * log_n) % 31u;
    uint32_t inv_n = (exp == 0) ? 1u : (1u << exp);
    uint32_t scale_total = n * n_cols;
    uint32_t scale_blocks = (scale_total + threads - 1) / threads;
    m31_batch_scale<<<scale_blocks, threads>>>(
        outer, inv_n, (uint64_t)n, (uint64_t)log_n, (uint64_t)n_cols);

    cudaDeviceSynchronize();
    cudaFree(d_spans);
}
