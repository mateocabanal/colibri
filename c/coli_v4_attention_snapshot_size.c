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

/* Persistent payload per layer:
 *   i32 window_size, head_dim, compressed_count, has_compressor, has_indexer
 *   window KV [window_size * head_dim] f32
 *   compressed KV [compressed_count * head_dim] f32
 *   optional compressor wire payload
 *   optional indexer wire payload
 * The outer global object owns version/endian/model/layout/state-ABI framing. */
size_t coli_v4_attention_snapshot_wire_bytes(
    const ColiV4AttentionSnapshot *snapshot) {
    if (!snapshot || snapshot->window_size < 0 || snapshot->head_dim <= 0 ||
        snapshot->compressed_count < 0)
        return snapshot ? SIZE_MAX : 0;
    size_t head = (size_t)snapshot->head_dim;
    size_t window = (size_t)snapshot->window_size;
    size_t compressed = (size_t)snapshot->compressed_count;
    if (head && (window > SIZE_MAX / head || compressed > SIZE_MAX / head))
        return SIZE_MAX;
    size_t window_values = window * head;
    size_t compressed_values = compressed * head;
    size_t header = 5 * sizeof(int32_t);
    if (window_values > (SIZE_MAX - header) / sizeof(float)) return SIZE_MAX;
    size_t total = header + window_values * sizeof(float);
    if (compressed_values > (SIZE_MAX - total) / sizeof(float)) return SIZE_MAX;
    total += compressed_values * sizeof(float);
    if (snapshot->compressor) {
        size_t bytes = coli_v4_compressor_snapshot_wire_bytes(snapshot->compressor);
        if (bytes == SIZE_MAX || total > SIZE_MAX - bytes) return SIZE_MAX;
        total += bytes;
    }
    if (snapshot->indexer) {
        size_t bytes = coli_v4_indexer_snapshot_wire_bytes(snapshot->indexer);
        if (bytes == SIZE_MAX || total > SIZE_MAX - bytes) return SIZE_MAX;
        total += bytes;
    }
    return total;
}

int coli_v4_attention_snapshot_wire_emit(
    const ColiV4AttentionSnapshot *snapshot,
    ColiV4PrefixWireSink sink, void *user_data) {
    if (!snapshot || !sink || snapshot->window_size < 0 ||
        snapshot->head_dim <= 0 || snapshot->compressed_count < 0 ||
        coli_v4_attention_snapshot_wire_bytes(snapshot) == SIZE_MAX)
        return -1;
    int32_t fields[5] = {
        snapshot->window_size, snapshot->head_dim, snapshot->compressed_count,
        snapshot->compressor ? 1 : 0, snapshot->indexer ? 1 : 0,
    };
    if (sink(user_data, fields, sizeof(fields))) return -1;
    size_t window_bytes = (size_t)snapshot->window_size *
                          (size_t)snapshot->head_dim * sizeof(float);
    if (window_bytes && !snapshot->kv) return -1;
    if (window_bytes && sink(user_data, snapshot->kv, window_bytes)) return -1;
    size_t compressed_bytes = (size_t)snapshot->compressed_count *
                              (size_t)snapshot->head_dim * sizeof(float);
    if (compressed_bytes && !snapshot->compressed) return -1;
    if (compressed_bytes && sink(user_data, snapshot->compressed, compressed_bytes))
        return -1;
    if (snapshot->compressor && coli_v4_compressor_snapshot_wire_emit(
            snapshot->compressor, sink, user_data))
        return -1;
    if (snapshot->indexer && coli_v4_indexer_snapshot_wire_emit(
            snapshot->indexer, sink, user_data))
        return -1;
    return 0;
}

int coli_v4_attention_snapshot_wire_write(
    const ColiV4AttentionSnapshot *snapshot,
    unsigned char *output, size_t output_size, size_t *written) {
    if (written) *written = 0;
    if (!snapshot || !output || !written || snapshot->window_size < 0 ||
        snapshot->head_dim <= 0 || snapshot->compressed_count < 0)
        return -1;
    size_t need = coli_v4_attention_snapshot_wire_bytes(snapshot);
    if (need == SIZE_MAX || need > output_size) return -1;

    int32_t fields[5] = {
        snapshot->window_size, snapshot->head_dim, snapshot->compressed_count,
        snapshot->compressor ? 1 : 0, snapshot->indexer ? 1 : 0,
    };
    memcpy(output, fields, sizeof(fields));
    size_t offset = sizeof(fields);
    size_t window_values = (size_t)snapshot->window_size *
                           (size_t)snapshot->head_dim;
    size_t window_bytes = window_values * sizeof(float);
    if (window_bytes) {
        if (!snapshot->kv) return -1;
        memcpy(output + offset, snapshot->kv, window_bytes);
        offset += window_bytes;
    }
    size_t compressed_values = (size_t)snapshot->compressed_count *
                               (size_t)snapshot->head_dim;
    size_t compressed_bytes = compressed_values * sizeof(float);
    if (compressed_bytes) {
        if (!snapshot->compressed) return -1;
        memcpy(output + offset, snapshot->compressed, compressed_bytes);
        offset += compressed_bytes;
    }
    if (snapshot->compressor) {
        size_t nested = 0;
        if (coli_v4_compressor_snapshot_wire_write(
                snapshot->compressor, output + offset, output_size - offset,
                &nested))
            return -1;
        offset += nested;
    }
    if (snapshot->indexer) {
        size_t nested = 0;
        if (coli_v4_indexer_snapshot_wire_write(
                snapshot->indexer, output + offset, output_size - offset,
                &nested))
            return -1;
        offset += nested;
    }
    *written = offset;
    return offset == need ? 0 : -1;
}

int coli_v4_attention_snapshot_wire_read(
    ColiV4AttentionSnapshot **output,
    const unsigned char *input, size_t input_size, size_t *consumed) {
    if (consumed) *consumed = 0;
    if (!output || !input || !consumed || input_size < 5 * sizeof(int32_t))
        return -1;
    *output = NULL;
    int32_t fields[5];
    memcpy(fields, input, sizeof(fields));
    int window_size = fields[0], head_dim = fields[1], compressed_count = fields[2];
    int has_compressor = fields[3], has_indexer = fields[4];
    if (window_size < 0 || head_dim <= 0 || compressed_count < 0 ||
        (has_compressor != 0 && has_compressor != 1) ||
        (has_indexer != 0 && has_indexer != 1) ||
        (has_indexer && !has_compressor))
        return -1;
    size_t head = (size_t)head_dim, window = (size_t)window_size;
    size_t compressed = (size_t)compressed_count;
    if (head && (window > SIZE_MAX / head || compressed > SIZE_MAX / head))
        return -1;
    size_t window_values = window * head;
    size_t compressed_values = compressed * head;
    size_t offset = sizeof(fields);
    if (window_values > (SIZE_MAX - offset) / sizeof(float)) return -1;
    size_t window_bytes = window_values * sizeof(float);
    offset += window_bytes;
    if (compressed_values > (SIZE_MAX - offset) / sizeof(float)) return -1;
    size_t compressed_bytes = compressed_values * sizeof(float);
    offset += compressed_bytes;
    if (offset > input_size) return -1;

    ColiV4AttentionSnapshot *snapshot = calloc(1, sizeof(*snapshot));
    if (!snapshot) return -1;
    snapshot->window_size = window_size;
    snapshot->head_dim = head_dim;
    snapshot->compressed_count = compressed_count;
    if (window_bytes) {
        snapshot->kv = malloc(window_bytes);
        if (!snapshot->kv) goto failed;
        memcpy(snapshot->kv, input + sizeof(fields), window_bytes);
    }
    if (compressed_bytes) {
        snapshot->compressed = malloc(compressed_bytes);
        if (!snapshot->compressed) goto failed;
        memcpy(snapshot->compressed,
               input + sizeof(fields) + window_bytes, compressed_bytes);
    }
    if (has_compressor) {
        size_t nested = 0;
        if (coli_v4_compressor_snapshot_wire_read(
                &snapshot->compressor, input + offset, input_size - offset,
                &nested))
            goto failed;
        if (nested > input_size - offset) goto failed;
        offset += nested;
    }
    if (has_indexer) {
        size_t nested = 0;
        if (coli_v4_indexer_snapshot_wire_read(
                &snapshot->indexer, input + offset, input_size - offset,
                &nested))
            goto failed;
        if (nested > input_size - offset) goto failed;
        offset += nested;
    }
    *output = snapshot;
    *consumed = offset;
    return 0;

failed:
    coli_v4_attention_snapshot_destroy(snapshot);
    return -1;
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
