#ifndef COLIBRI_PREFIX_HOT_RESOURCE_H
#define COLIBRI_PREFIX_HOT_RESOURCE_H

#include "resource_planner.h"

#include <stdint.h>

/*
 * Model-neutral inventory for one process-resident resumable prefix boundary.
 *
 * The cache owns entry lifetime/state bytes. The global policy owns forecasting
 * future reuse. Keeping these raw observations separate is important: a Qwen
 * cache hit and a V4 cache hit must not silently use different planning
 * horizons merely because their engine-local LRU clocks happen to differ.
 */
typedef struct {
    uint32_t id;
    uint64_t resident_bytes;
    uint64_t token_count;
    uint64_t hit_count;
    uint64_t last_used;
    uint32_t references;
} ColiPrefixHotEntryInfo;

/*
 * Model-neutral bridge from a valued resumable prefix-cache entry into the
 * global optional-residency planner.
 *
 * The cache/state adapter owns the semantics of a reusable boundary and the
 * policy that estimates reuse over its planning horizon. This helper only
 * translates those observations into the common planner vocabulary. Keeping
 * that policy outside resource_planner.h avoids teaching the allocator about
 * tokenization, Qwen GDN state, V4 compressor state, or model names.
 *
 * All candidates competing in one selection must use a compatible reuse_weight
 * scale/horizon, exactly like ColiResourceBenefitEstimate. bytes_per_hit should
 * describe recurring work/traffic avoided by one RAM-hot hit when measured; if
 * that telemetry is not available yet, callers may conservatively use the
 * state bytes that would otherwise have to be restored/reconstructed. Do not
 * invent exposed-nanosecond values merely to make a prefix win selection.
 */
typedef struct {
    uint32_t id;
    uint64_t resident_bytes;
    uint64_t reuse_weight;
    uint64_t bytes_per_hit;
    uint64_t exposed_ns_per_hit;
} ColiPrefixHotResourceEstimate;

static inline int coli_prefix_hot_resource_estimate(
        const ColiPrefixHotResourceEstimate *prefix,
        ColiResourceBenefitEstimate *estimate) {
    if (!prefix || !estimate || !prefix->resident_bytes) return -1;
    *estimate = (ColiResourceBenefitEstimate){
        .kind = COLI_RESOURCE_PREFIX_HOT,
        .id = prefix->id,
        .resident_bytes = prefix->resident_bytes,
        .reuse_weight = prefix->reuse_weight,
        .bytes_per_miss = prefix->bytes_per_hit,
        .exposed_ns_per_miss = prefix->exposed_ns_per_hit,
    };
    return 0;
}

static inline int coli_prefix_hot_resource_candidate(
        const ColiPrefixHotResourceEstimate *prefix,
        ColiResourceCandidate *candidate) {
    ColiResourceBenefitEstimate estimate;
    if (coli_prefix_hot_resource_estimate(prefix, &estimate) != 0)
        return -1;
    return coli_resource_candidate_from_benefit(&estimate, candidate);
}

#endif /* COLIBRI_PREFIX_HOT_RESOURCE_H */
