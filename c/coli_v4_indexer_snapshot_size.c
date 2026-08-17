#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Keep the unit's symbol-remapping include order intact; private snapshot and
 * live indexer layouts are visible after the amalgamated unit has compiled. */
#include "deepseek_v4.c"
#include "coli_v4_prefix_cache.h"

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
