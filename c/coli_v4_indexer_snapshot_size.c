#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Keep the unit's symbol-remapping include order intact; private snapshot
 * layout is visible after the amalgamated unit has compiled. */
#include "deepseek_v4.c"
#include "coli_v4_prefix_cache.h"

size_t coli_v4_indexer_snapshot_bytes(const ColiV4IndexerSnapshot *snapshot) {
    if (!snapshot) return 0;
    size_t total = sizeof(*snapshot);
    if (snapshot->count < 0 || snapshot->head_dim < 0) return SIZE_MAX;
    size_t count = (size_t)snapshot->count;
    size_t head_dim = (size_t)snapshot->head_dim;
    if (head_dim && count > SIZE_MAX / head_dim) return SIZE_MAX;
    size_t values = count * head_dim;
    if (values > (SIZE_MAX - total) / sizeof(float)) return SIZE_MAX;
    total += values * sizeof(float);
    size_t compressor =
        coli_v4_compressor_snapshot_bytes(snapshot->compressor);
    if (compressor == SIZE_MAX || total > SIZE_MAX - compressor) return SIZE_MAX;
    return total + compressor;
}
