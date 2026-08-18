#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Existing transaction implementation + private-layout byte accounting. Keep
 * the amalgamation's include order intact so any unit-local symbol remapping is
 * applied to deepseek_v4_internal.h declarations before its include guard. */
#include "deepseek_v4.c"
#include "coli_v4_prefix_cache.h"

#include <limits.h>

int coli_v4_compressor_snapshot_restore_unbound(
    ColiDeepSeekV4CompressorState **output,
    const ColiV4CompressorSnapshot *snapshot,
    const ColiDeepSeekV4Config *config,
    int layer, int compression_ratio,
    const ColiDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size);
int coli_v4_indexer_snapshot_restore_unbound(
    ColiDeepSeekV4Indexer **output,
    const ColiV4IndexerSnapshot *snapshot,
    const ColiDeepSeekV4Config *config,
    int layer, int compression_ratio, int max_context,
    char *error, size_t error_size);

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

static void discard_fresh_compressed_layout(
    ColiDeepSeekV4WindowAttentionState *state) {
    if (!state) return;
    coli_v4_indexer_destroy(state->indexer);
    coli_v4_compressor_destroy(state->compressor);
    free(state->compressed);
    state->indexer = NULL;
    state->compressor = NULL;
    state->compressed = NULL;
    state->compressed_count = 0;
    state->compressed_capacity = 0;
    state->ratio = 0;
    state->layer = -1;
}

/* The ordinary transaction restore expects compressor/indexer objects to have
 * been lazily created by a previous layer pass. Cross-session cache hits land
 * in a virgin session, so hydrate those recurrent objects from the snapshot
 * without loading dense weights. On the first fresh-tail token the normal
 * prepare_compressed_state() path binds the real layer weights before use. */
int coli_v4_attention_snapshot_restore_fresh(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiV4AttentionSnapshot *snapshot,
    const ColiDeepSeekV4Config *config, int layer) {
    if (!state || !snapshot || !config || layer < 0 ||
        layer >= config->num_hidden_layers ||
        layer >= config->compress_ratio_count)
        return -1;

    int ratio = config->compress_ratios[layer];
    if (ratio <= 0)
        return coli_v4_attention_snapshot_restore(state, snapshot);

    if (state->layer >= 0) {
        if (state->layer != layer || state->ratio != ratio) return -1;
        return coli_v4_attention_snapshot_restore(state, snapshot);
    }

    if (state->compressor || state->indexer || state->compressed ||
        state->compressed_capacity || state->compressed_count ||
        !snapshot->compressor ||
        ((ratio == 4) != (snapshot->indexer != NULL)))
        return -1;

    int capacity = 16;
    while (capacity < snapshot->compressed_count) {
        if (capacity > INT_MAX / 2) return -1;
        capacity *= 2;
    }
    if (state->head_dim <= 0 ||
        (size_t)capacity > SIZE_MAX / (size_t)state->head_dim ||
        (size_t)capacity * (size_t)state->head_dim >
            SIZE_MAX / sizeof(*state->compressed))
        return -1;

    state->layer = layer;
    state->ratio = ratio;
    state->compressed_capacity = capacity;
    state->compressed = calloc(
        (size_t)capacity * (size_t)state->head_dim,
        sizeof(*state->compressed));
    if (!state->compressed) {
        discard_fresh_compressed_layout(state);
        return -1;
    }

    ColiDeepSeekV4CompressorOptions options = {
        "attn.compressor", config->head_dim, 0
    };
    char error[256] = {0};
    if (coli_v4_compressor_snapshot_restore_unbound(
            &state->compressor, snapshot->compressor, config, layer, ratio,
            &options, error, sizeof(error))) {
        discard_fresh_compressed_layout(state);
        return -1;
    }
    if (snapshot->indexer && coli_v4_indexer_snapshot_restore_unbound(
            &state->indexer, snapshot->indexer, config, layer, ratio,
            config->max_position_embeddings, error, sizeof(error))) {
        discard_fresh_compressed_layout(state);
        return -1;
    }

    if (coli_v4_attention_snapshot_restore(state, snapshot)) {
        discard_fresh_compressed_layout(state);
        return -1;
    }
    return 0;
}
