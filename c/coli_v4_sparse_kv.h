#ifndef COLI_V4_SPARSE_KV_H
#define COLI_V4_SPARSE_KV_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    const float *window;
    int window_count;
    const float *compressed;
    int compressed_count;
    int row_dim;
} ColiV4SparseKVView;

static inline int coli_v4_sparse_kv_count(const ColiV4SparseKVView *view) {
    if (!view || view->window_count < 0 || view->compressed_count < 0 ||
        view->row_dim <= 0 ||
        (view->window_count && !view->window) ||
        (view->compressed_count && !view->compressed) ||
        view->window_count > INT32_MAX - view->compressed_count)
        return -1;
    return view->window_count + view->compressed_count;
}

/* Resolve the same logical row ordering as the old materialized
 * [window][compressed] buffer without copying either backing store. */
static inline const float *coli_v4_sparse_kv_row(
        const ColiV4SparseKVView *view, int logical_index) {
    int total = coli_v4_sparse_kv_count(view);
    if (total < 0 || logical_index < 0 || logical_index >= total) return NULL;
    if (logical_index < view->window_count)
        return view->window + (size_t)logical_index * (size_t)view->row_dim;
    logical_index -= view->window_count;
    return view->compressed + (size_t)logical_index * (size_t)view->row_dim;
}

/* Test/oracle helper: reconstruct the legacy contiguous representation. This
 * is intentionally not the inference API; it exists to prove logical ordering
 * while the hot path moves to row-wise sparse access. */
static inline int coli_v4_sparse_kv_materialize_oracle(
        const ColiV4SparseKVView *view, float *dst, size_t dst_floats) {
    int total = coli_v4_sparse_kv_count(view);
    if (total < 0 || !dst) return -1;
    size_t rows = (size_t)total;
    size_t dim = (size_t)view->row_dim;
    if (rows && dim > SIZE_MAX / rows) return -1;
    size_t need = rows * dim;
    if (dst_floats < need) return -1;
    if (view->window_count)
        memcpy(dst, view->window,
               (size_t)view->window_count * dim * sizeof(float));
    if (view->compressed_count)
        memcpy(dst + (size_t)view->window_count * dim, view->compressed,
               (size_t)view->compressed_count * dim * sizeof(float));
    return 0;
}

/* Gather only sparse-selected rows. This is useful as a transitional oracle:
 * it changes copy complexity from O(total context) to O(k_top) while callers
 * are migrated to a fully row-addressed sparse-attention kernel. */
static inline int coli_v4_sparse_kv_gather_selected(
        const ColiV4SparseKVView *view,
        const int *logical_indices, int count,
        float *dst, size_t dst_floats) {
    if (!view || !logical_indices || count < 0 || (!dst && count)) return -1;
    size_t dim = (size_t)view->row_dim;
    if ((size_t)count && dim > SIZE_MAX / (size_t)count) return -1;
    if (dst_floats < (size_t)count * dim) return -1;
    for (int i = 0; i < count; ++i) {
        const float *row = coli_v4_sparse_kv_row(view, logical_indices[i]);
        if (!row) return -1;
        memcpy(dst + (size_t)i * dim, row, dim * sizeof(float));
    }
    return 0;
}

#endif
