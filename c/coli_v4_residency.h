#ifndef COLI_V4_RESIDENCY_H
#define COLI_V4_RESIDENCY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Small cross-module contract for the current COLI V4 residency tiers.
 *
 * The expert store owns the hard cache budget supplied by the engine planner.
 * It reserves only the bytes required for its transient/persistent expert
 * slots, then offers the remainder to the deterministic dense/static cache.
 * This keeps total optional residency within the planner's existing budget
 * while #3 evolves toward a first-class tensor-granular planner.
 *
 * Runtime A/B: the expert store defaults to the new balanced split;
 * V4_RESIDENCY_POLICY=legacy restores the old per-layer expert-cache geometry
 * and leaves no reclaimed budget for this dense/static cache.
 */
typedef struct {
    uint64_t budget_bytes;
    uint64_t resident_bytes;
    uint64_t entries;
    uint64_t hits;
    uint64_t misses;
    uint64_t admissions;
    uint64_t rejected_bytes;
    uint64_t bytes_avoided;
} ColiV4DenseCacheStats;

/* Reconfigure the process-local COLI dense/static cache. This is called during
 * V4 engine setup before inference starts. Reconfiguration drops any previous
 * cache generation; normal single-engine use therefore has deterministic
 * ownership and teardown. */
void coli_v4_dense_cache_configure(uint64_t budget_bytes);

/* Release all persistent dense/static cache allocations. */
void coli_v4_dense_cache_reset(void);

/* Snapshot counters for diagnostics/tests. */
void coli_v4_dense_cache_stats(ColiV4DenseCacheStats *out);

#ifdef __cplusplus
}
#endif

#endif /* COLI_V4_RESIDENCY_H */
