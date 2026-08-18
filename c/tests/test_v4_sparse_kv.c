#include "../coli_v4_sparse_kv.h"

#include <stdio.h>
#include <string.h>

static int row_equal(const float *a, const float *b, int dim) {
    for (int i = 0; i < dim; ++i)
        if (a[i] != b[i]) return 0;
    return 1;
}

int main(void) {
    enum { DIM = 4, WINDOW = 3, COMPRESSED = 2 };
    const float window[WINDOW * DIM] = {
        0, 1, 2, 3,
        10, 11, 12, 13,
        20, 21, 22, 23,
    };
    const float compressed[COMPRESSED * DIM] = {
        100, 101, 102, 103,
        110, 111, 112, 113,
    };
    const float legacy[(WINDOW + COMPRESSED) * DIM] = {
        0, 1, 2, 3,
        10, 11, 12, 13,
        20, 21, 22, 23,
        100, 101, 102, 103,
        110, 111, 112, 113,
    };
    ColiV4SparseKVView view = {
        window, WINDOW, compressed, COMPRESSED, DIM
    };

    if (coli_v4_sparse_kv_count(&view) != WINDOW + COMPRESSED) return 1;
    for (int logical = 0; logical < WINDOW + COMPRESSED; ++logical) {
        const float *row = coli_v4_sparse_kv_row(&view, logical);
        if (!row || !row_equal(row, legacy + logical * DIM, DIM)) return 2;
    }
    if (coli_v4_sparse_kv_row(&view, -1) ||
        coli_v4_sparse_kv_row(&view, WINDOW + COMPRESSED))
        return 3;

    float materialized[(WINDOW + COMPRESSED) * DIM];
    if (coli_v4_sparse_kv_materialize_oracle(
            &view, materialized, sizeof(materialized) / sizeof(*materialized)) != 0 ||
        memcmp(materialized, legacy, sizeof(legacy)) != 0)
        return 4;

    const int selected[] = {4, 0, 3, 2};
    float gathered[4 * DIM];
    if (coli_v4_sparse_kv_gather_selected(
            &view, selected, 4, gathered,
            sizeof(gathered) / sizeof(*gathered)) != 0)
        return 5;
    for (int i = 0; i < 4; ++i)
        if (!row_equal(gathered + i * DIM, legacy + selected[i] * DIM, DIM))
            return 6;

    /* Window-only and compressed-only are both valid views. */
    ColiV4SparseKVView window_only = {window, WINDOW, NULL, 0, DIM};
    ColiV4SparseKVView compressed_only = {NULL, 0, compressed, COMPRESSED, DIM};
    if (coli_v4_sparse_kv_count(&window_only) != WINDOW ||
        coli_v4_sparse_kv_count(&compressed_only) != COMPRESSED ||
        !row_equal(coli_v4_sparse_kv_row(&compressed_only, 1), compressed + DIM, DIM))
        return 7;

    /* Bad geometry and selected indices fail closed instead of forming a bad
     * pointer into either backing store. */
    ColiV4SparseKVView bad = view;
    bad.row_dim = 0;
    if (coli_v4_sparse_kv_count(&bad) >= 0) return 8;
    const int bad_selected[] = {0, 99};
    if (coli_v4_sparse_kv_gather_selected(
            &view, bad_selected, 2, gathered,
            sizeof(gathered) / sizeof(*gathered)) == 0)
        return 9;

    puts("DeepSeek-V4 segmented sparse KV ordering: ok");
    return 0;
}
