#ifndef COLI_V4_RESIDENCY_H
#define COLI_V4_RESIDENCY_H

#include "resource_planner.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t budget_bytes;
    uint64_t resident_bytes;
    uint64_t pinned_bytes;
    uint64_t entries;
    uint64_t hits;
    uint64_t misses;
    uint64_t admissions;
    uint64_t evictions;
    uint64_t evicted_bytes;
    uint64_t rejected_bytes;
    uint64_t bytes_avoided;
    uint64_t copy_bytes;
    uint64_t copy_ns;
} ColiV4DenseCacheStats;

/* Compatibility vocabulary. V4 was the proving ground for benefit-per-byte
 * residency; the mechanism now lives in resource_planner.h so Qwen, prompt
 * cache, sequence state and accelerator tiers can consume the same policy. */
typedef ColiResourceValueMode ColiV4ResidencyValueMode;
#define COLI_V4_RESIDENCY_VALUE_BYTES      COLI_RESOURCE_VALUE_BYTES
#define COLI_V4_RESIDENCY_VALUE_EXPOSED_NS COLI_RESOURCE_VALUE_EXPOSED_NS

typedef ColiResourceKind ColiV4ResidencyKind;
#define COLI_V4_RESIDENCY_DENSE_TENSOR      COLI_RESOURCE_DENSE_TENSOR
#define COLI_V4_RESIDENCY_HEAD              COLI_RESOURCE_HEAD
#define COLI_V4_RESIDENCY_PERSISTENT_EXPERT COLI_RESOURCE_PERSISTENT_EXPERT
#define COLI_V4_RESIDENCY_OTHER             COLI_RESOURCE_OTHER

typedef ColiResourceCandidate ColiV4ResidencyCandidate;
typedef ColiResourceSelection ColiV4ResidencySelection;

static inline int coli_v4_expert_slot_bytes(uint64_t record_bytes,
                                            uint64_t *slot_bytes) {
    if (!slot_bytes || !record_bytes || record_bytes > UINT64_MAX - 16383u)
        return -1;
    *slot_bytes = (record_bytes + 16383u) & ~UINT64_C(16383);
    return *slot_bytes ? 0 : -1;
}

static inline int coli_v4_residency_ratio_compare(
        uint64_t a_num, uint64_t a_den,
        uint64_t b_num, uint64_t b_den) {
    return coli_resource_ratio_compare(a_num, a_den, b_num, b_den);
}

static inline uint64_t coli_v4_residency_candidate_value(
        const ColiV4ResidencyCandidate *candidate,
        ColiV4ResidencyValueMode mode) {
    return coli_resource_candidate_value(candidate, mode);
}

static inline uint64_t coli_v4_residency_saturating_add(uint64_t a,
                                                        uint64_t b) {
    return coli_resource_saturating_add(a, b);
}

static inline int coli_v4_residency_select(
        const ColiV4ResidencyCandidate *candidates, size_t count,
        uint64_t budget_bytes, ColiV4ResidencyValueMode mode,
        unsigned char *selected, ColiV4ResidencySelection *out) {
    return coli_resource_select(candidates, count, budget_bytes, mode,
                                selected, out);
}

void coli_v4_dense_cache_configure(uint64_t budget_bytes);
<<<<<<< HEAD
/* Change the dense admission ceiling and reclaim idle entries immediately.
 * Active borrowed entries are pinned until their layer/session releases them;
 * lowering below pinned_bytes therefore clamps only to that live-borrow floor. */
uint64_t coli_v4_dense_cache_set_budget(uint64_t budget_bytes);
=======
>>>>>>> origin/feat/sequence-state-core
void coli_v4_dense_cache_reset(void);
void coli_v4_dense_cache_stats(ColiV4DenseCacheStats *out);

#ifdef __cplusplus
}
#endif

#endif /* COLI_V4_RESIDENCY_H */
