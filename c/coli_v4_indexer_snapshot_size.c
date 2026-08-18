#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Keep the unit's symbol-remapping include order intact; private snapshot and
 * live indexer layouts are visible after the amalgamated unit has compiled. */
#include "deepseek_v4.c"
#include "coli_v4_prefix_cache.h"

#include <stdarg.h>

int coli_v4_compressor_snapshot_restore_unbound(
    ColiDeepSeekV4CompressorState **output,
    const ColiV4CompressorSnapshot *snapshot,
    const ColiDeepSeekV4Config *config,
    int layer, int compression_ratio,
    const ColiDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size);

static size_t indexer_geometry_bytes(size_t count, size_t head_dim,
                                     size_t compressor) {
    size_t total = sizeof(ColiV4IndexerSnapshot);
    if (head_dim && count > SIZE_MAX / head_dim) return SIZE_MAX;
    size_t values = count * head_dim;
    if (values > (SIZE_MAX - total) / sizeof(float)) return SIZE_MAX;
    total += values * sizeof(float);
    if (compressor == SIZE_MAX || total > SIZE_MAX - compressor) return SIZE_MAX;
    return total + compressor;
}

size_t coli_v4_indexer_snapshot_bytes(const ColiV4IndexerSnapshot *snapshot) {
    if (!snapshot) return 0;
    if (snapshot->count < 0 || snapshot->head_dim < 0) return SIZE_MAX;
    return indexer_geometry_bytes(
        (size_t)snapshot->count, (size_t)snapshot->head_dim,
        coli_v4_compressor_snapshot_bytes(snapshot->compressor));
}

size_t coli_v4_indexer_state_snapshot_bytes(
    const ColiDeepSeekV4Indexer *state) {
    if (!state) return 0;
    if (!state->config || state->count < 0 || state->config->index_head_dim < 0)
        return SIZE_MAX;
    return indexer_geometry_bytes(
        (size_t)state->count, (size_t)state->config->index_head_dim,
        coli_v4_compressor_state_snapshot_bytes(state->compressor));
}

static int indexer_hydrate_error(char *error, size_t error_size,
                                 const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

/* Restore an indexer into a weightless state. The normal attention path binds
 * its real layer payload immediately before using it, so this does not add a
 * second dense-layer read to a cache hit. Capacity grows to the saved prefix
 * when a long snapshot exceeds the normal 128-entry initial allocation. */
int coli_v4_indexer_snapshot_restore_unbound(
    ColiDeepSeekV4Indexer **output,
    const ColiV4IndexerSnapshot *snapshot,
    const ColiDeepSeekV4Config *config,
    int layer, int compression_ratio, int max_context,
    char *error, size_t error_size) {
    if (!output || !snapshot || !snapshot->compressor || !config ||
        layer < 0 || compression_ratio < 1 || max_context < 4 ||
        config->index_head_dim < 1 || config->index_n_heads < 1 ||
        snapshot->count < 0 || snapshot->head_dim != config->index_head_dim)
        return indexer_hydrate_error(
            error, error_size, "invalid unbound indexer snapshot geometry");

    *output = NULL;
    ColiDeepSeekV4Indexer *state = calloc(1, sizeof(*state));
    if (!state)
        return indexer_hydrate_error(
            error, error_size, "out of memory hydrating indexer snapshot");
    state->weights = NULL;
    state->config = config;
    state->layer = layer;

    int capacity = (max_context + 3) / 4;
    if (capacity > 128) capacity = 128;
    if (capacity < snapshot->count) capacity = snapshot->count;
    if (capacity < 1 ||
        (size_t)capacity > SIZE_MAX / (size_t)config->index_head_dim ||
        (size_t)capacity * (size_t)config->index_head_dim >
            SIZE_MAX / sizeof(*state->compressed)) {
        free(state);
        return indexer_hydrate_error(
            error, error_size, "indexer snapshot capacity overflow");
    }
    state->capacity = capacity;
    state->compressed = calloc(
        (size_t)capacity * (size_t)config->index_head_dim,
        sizeof(*state->compressed));

    ColiDeepSeekV4CompressorOptions options = {
        "attn.indexer.compressor", config->index_head_dim, 1
    };
    if (!state->compressed || coli_v4_compressor_snapshot_restore_unbound(
            &state->compressor, snapshot->compressor, config, layer,
            compression_ratio, &options, error, error_size)) {
        if (state->compressor) coli_v4_compressor_destroy(state->compressor);
        free(state->compressed);
        free(state);
        return indexer_hydrate_error(
            error, error_size, "cannot hydrate indexer compressor snapshot");
    }

    state->count = snapshot->count;
    if (snapshot->count)
        memcpy(state->compressed, snapshot->compressed,
               (size_t)snapshot->count * (size_t)snapshot->head_dim *
                   sizeof(*state->compressed));
    *output = state;
    return 0;
}
