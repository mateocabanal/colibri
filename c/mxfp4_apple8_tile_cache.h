#ifndef COLIBRI_MXFP4_APPLE8_TILE_CACHE_H
#define COLIBRI_MXFP4_APPLE8_TILE_CACHE_H

#include "coli_executor.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t lookup;
    uint64_t hit;
    uint64_t miss;
    uint64_t stale;
    uint64_t corrupt;
    uint64_t read_bytes;
    uint64_t read_ns;
    uint64_t write_bytes;
    uint64_t write_ns;
    uint64_t write_dropped;
    uint64_t prepare_ns_avoided;
    uint64_t cold_prepares;
    uint64_t cold_prepare_ns;
    uint64_t installs;
    uint64_t install_failures;
} ColiMxfp4Apple8DerivedCacheStats;

/* Separate milestone-2 gate. V4_METAL_TILE=1 alone keeps milestone-1 behavior.
 * V4_METAL_TILE_DERIVED_CACHE=1 enables #137 persistence. */
int coli_mxfp4_apple8_derived_cache_enabled(void);

/* Returns 1 when a #137 hit or an exact cold transform was installed into the
 * bounded Metal tile slots. Returns 0 on disabled/miss failure so the caller
 * may retain the milestone-1 direct row->tile repack fallback. */
int coli_mxfp4_apple8_derived_prepare_expert(
    const ColiExecutor *executor,
    int32_t layer, int32_t expert,
    const void *resident_slot, size_t resident_bytes,
    uint64_t source_generation);

void coli_mxfp4_apple8_derived_cache_stats(
    ColiMxfp4Apple8DerivedCacheStats *stats);

#ifdef __cplusplus
}
#endif

#endif
