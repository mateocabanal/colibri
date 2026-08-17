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

static size_t attention_geometry_bytes(int window_size, int head_dim,
                                       int compressed_count,
                                       size_t compressor, size_t indexer) {
    if (window_size < 0 || head_dim < 0 || compressed_count < 0)
        return SIZE_MAX;

    size_t total = sizeof(ColiV4AttentionSnapshot);
    size_t head = (size_t)head_dim;
    size_t window = (size_t)window_size;
    size_t compressed = (size_t)compressed_count;
    if (head && (window > SIZE_MAX / head || compressed > SIZE_MAX / head))
        return SIZE_MAX;
    size_t window_values = window * head;
    size_t compressed_values = compressed * head;
    if (window_values > SIZE_MAX / sizeof(float) ||
        compressed_values > SIZE_MAX / sizeof(float))
        return SIZE_MAX;
    total = add_snapshot_bytes(total, window_values * sizeof(float));
    total = add_snapshot_bytes(total, compressed_values * sizeof(float));
    total = add_snapshot_bytes(total, compressor);
    total = add_snapshot_bytes(total, indexer);
    return total;
}

size_t coli_v4_attention_snapshot_bytes(const ColiV4AttentionSnapshot *snapshot) {
    if (!snapshot) return 0;
    return attention_geometry_bytes(
        snapshot->window_size, snapshot->head_dim, snapshot->compressed_count,
        coli_v4_compressor_snapshot_bytes(snapshot->compressor),
        coli_v4_indexer_snapshot_bytes(snapshot->indexer));
}

size_t coli_v4_attention_state_snapshot_bytes(
    const ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return 0;
    return attention_geometry_bytes(
        state->window_size, state->head_dim, state->compressed_count,
        coli_v4_compressor_state_snapshot_bytes(state->compressor),
        coli_v4_indexer_state_snapshot_bytes(state->indexer));
}
