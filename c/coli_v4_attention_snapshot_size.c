#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Existing transaction implementation + private-layout byte accounting. Keep
 * the amalgamation's include order intact so any unit-local symbol remapping is
 * applied to deepseek_v4_internal.h declarations before its include guard. */
#include "deepseek_v4.c"
#include "coli_v4_prefix_cache.h"

static size_t add_snapshot_bytes(size_t total, size_t amount) {
    return total > SIZE_MAX - amount ? SIZE_MAX : total + amount;
}

size_t coli_v4_attention_snapshot_bytes(const ColiV4AttentionSnapshot *snapshot) {
    if (!snapshot) return 0;
    if (snapshot->window_size < 0 || snapshot->head_dim < 0 ||
        snapshot->compressed_count < 0)
        return SIZE_MAX;

    size_t total = sizeof(*snapshot);
    size_t head_dim = (size_t)snapshot->head_dim;
    size_t window = (size_t)snapshot->window_size;
    size_t compressed = (size_t)snapshot->compressed_count;
    if (head_dim && (window > SIZE_MAX / head_dim ||
                     compressed > SIZE_MAX / head_dim))
        return SIZE_MAX;
    size_t window_values = window * head_dim;
    size_t compressed_values = compressed * head_dim;
    if (window_values > SIZE_MAX / sizeof(float) ||
        compressed_values > SIZE_MAX / sizeof(float))
        return SIZE_MAX;
    total = add_snapshot_bytes(total, window_values * sizeof(float));
    total = add_snapshot_bytes(total, compressed_values * sizeof(float));
    total = add_snapshot_bytes(
        total, coli_v4_compressor_snapshot_bytes(snapshot->compressor));
    total = add_snapshot_bytes(
        total, coli_v4_indexer_snapshot_bytes(snapshot->indexer));
    return total;
}
