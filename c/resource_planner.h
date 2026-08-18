#ifndef COLIBRI_RESOURCE_PLANNER_H
#define COLIBRI_RESOURCE_PLANNER_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef enum {
    COLI_RESOURCE_VALUE_BYTES = 0,
    COLI_RESOURCE_VALUE_EXPOSED_NS = 1,
} ColiResourceValueMode;

/* Keep the first four values aligned with the completed V4 planner so V4 can
 * become a compatibility adapter without changing deterministic tie order. */
typedef enum {
    COLI_RESOURCE_DENSE_TENSOR = 0,
    COLI_RESOURCE_HEAD = 1,
    COLI_RESOURCE_PERSISTENT_EXPERT = 2,
    COLI_RESOURCE_OTHER = 3,
    COLI_RESOURCE_PREFIX_HOT = 4,
    COLI_RESOURCE_SEQUENCE_STATE = 5,
    COLI_RESOURCE_SCRATCH = 6,
    COLI_RESOURCE_IO_STAGING = 7,
} ColiResourceKind;

typedef enum {
    COLI_RESOURCE_TIER_HOST = 0,
    COLI_RESOURCE_TIER_UMA = 1,
    COLI_RESOURCE_TIER_PINNED_HOST = 2,
    COLI_RESOURCE_TIER_DEVICE = 3,
    COLI_RESOURCE_TIER_COUNT = 4,
} ColiResourceTier;

/* This prefix intentionally matches the old V4 candidate exactly. */
typedef struct {
    ColiResourceKind kind;
    uint32_t id;
    uint64_t resident_bytes;
    uint64_t expected_bytes_avoided;
    uint64_t expected_exposed_ns_avoided;
} ColiResourceCandidate;

typedef struct {
    uint64_t budget_bytes;
    uint64_t selected_resident_bytes;
    uint64_t expected_bytes_avoided;
    uint64_t expected_exposed_ns_avoided;
    size_t selected_count;
} ColiResourceSelection;

typedef struct {
    uint64_t budget_bytes;
    uint64_t mandatory_bytes;
    uint64_t optional_budget_bytes;
    uint64_t selected_optional_bytes;
    uint64_t free_bytes;
} ColiResourceTierPlan;

typedef struct {
    ColiResourceTierPlan tier[COLI_RESOURCE_TIER_COUNT];
} ColiResourcePlan;

static inline int coli_resource_u64_add(uint64_t a, uint64_t b,
                                        uint64_t *out) {
    if (!out || UINT64_MAX - a < b) return -1;
    *out = a + b;
    return 0;
}

static inline uint64_t coli_resource_saturating_add(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

/* Exact a_num/a_den vs b_num/b_den comparison without multiplication overflow.
 * This preserves the deterministic integer policy proven by the V4 planner. */
static inline int coli_resource_ratio_compare(uint64_t a_num, uint64_t a_den,
                                              uint64_t b_num, uint64_t b_den) {
    int direction = 1;
    if (!a_den || !b_den) return 0;
    for (;;) {
        uint64_t aq = a_num / a_den;
        uint64_t bq = b_num / b_den;
        if (aq != bq) return aq > bq ? direction : -direction;
        uint64_t ar = a_num % a_den;
        uint64_t br = b_num % b_den;
        if (!ar || !br) {
            if (!ar && !br) return 0;
            return ar ? direction : -direction;
        }
        a_num = a_den; a_den = ar;
        b_num = b_den; b_den = br;
        direction = -direction;
    }
}

static inline uint64_t coli_resource_candidate_value(
        const ColiResourceCandidate *candidate, ColiResourceValueMode mode) {
    return mode == COLI_RESOURCE_VALUE_EXPOSED_NS
        ? candidate->expected_exposed_ns_avoided
        : candidate->expected_bytes_avoided;
}

/* Backend-neutral greedy optional-residency allocator. Mandatory memory is not
 * represented as a fake high-score candidate: the caller reserves it first via
 * ColiResourcePlan and only passes the remaining tier budget here. */
static inline int coli_resource_select(
        const ColiResourceCandidate *candidates, size_t count,
        uint64_t budget_bytes, ColiResourceValueMode mode,
        unsigned char *selected, ColiResourceSelection *out) {
    if ((!candidates && count) || (!selected && count) || !out ||
        (mode != COLI_RESOURCE_VALUE_BYTES &&
         mode != COLI_RESOURCE_VALUE_EXPOSED_NS))
        return -1;

    if (count) memset(selected, 0, count);
    memset(out, 0, sizeof(*out));
    out->budget_bytes = budget_bytes;

    uint64_t remaining = budget_bytes;
    for (;;) {
        size_t best = count;
        uint64_t best_value = 0;
        for (size_t index = 0; index < count; ++index) {
            const ColiResourceCandidate *candidate = &candidates[index];
            if (selected[index] || !candidate->resident_bytes ||
                candidate->resident_bytes > remaining)
                continue;
            uint64_t value = coli_resource_candidate_value(candidate, mode);
            if (!value) continue;

            int better = 0;
            if (best == count) {
                better = 1;
            } else {
                const ColiResourceCandidate *incumbent = &candidates[best];
                int ratio = coli_resource_ratio_compare(
                    value, candidate->resident_bytes,
                    best_value, incumbent->resident_bytes);
                if (ratio > 0) {
                    better = 1;
                } else if (ratio == 0) {
                    if (candidate->kind < incumbent->kind ||
                        (candidate->kind == incumbent->kind &&
                         (candidate->id < incumbent->id ||
                          (candidate->id == incumbent->id && index < best))))
                        better = 1;
                }
            }
            if (better) {
                best = index;
                best_value = value;
            }
        }
        if (best == count) break;
        const ColiResourceCandidate *chosen = &candidates[best];
        selected[best] = 1;
        remaining -= chosen->resident_bytes;
        out->selected_resident_bytes += chosen->resident_bytes;
        out->expected_bytes_avoided = coli_resource_saturating_add(
            out->expected_bytes_avoided, chosen->expected_bytes_avoided);
        out->expected_exposed_ns_avoided = coli_resource_saturating_add(
            out->expected_exposed_ns_avoided,
            chosen->expected_exposed_ns_avoided);
        out->selected_count++;
    }
    return 0;
}

static inline void coli_resource_plan_init(ColiResourcePlan *plan) {
    if (plan) memset(plan, 0, sizeof(*plan));
}

static inline int coli_resource_plan_set_budget(ColiResourcePlan *plan,
                                                ColiResourceTier tier,
                                                uint64_t bytes) {
    if (!plan || tier < 0 || tier >= COLI_RESOURCE_TIER_COUNT) return -1;
    ColiResourceTierPlan *p = &plan->tier[tier];
    if (p->mandatory_bytes > bytes || p->selected_optional_bytes >
        bytes - p->mandatory_bytes)
        return -1;
    p->budget_bytes = bytes;
    p->optional_budget_bytes = bytes - p->mandatory_bytes;
    p->free_bytes = p->optional_budget_bytes - p->selected_optional_bytes;
    return 0;
}

static inline int coli_resource_plan_reserve_mandatory(
        ColiResourcePlan *plan, ColiResourceTier tier, uint64_t bytes) {
    if (!plan || tier < 0 || tier >= COLI_RESOURCE_TIER_COUNT) return -1;
    ColiResourceTierPlan *p = &plan->tier[tier];
    uint64_t mandatory;
    if (coli_resource_u64_add(p->mandatory_bytes, bytes, &mandatory) != 0 ||
        mandatory > p->budget_bytes || p->selected_optional_bytes >
        p->budget_bytes - mandatory)
        return -1;
    p->mandatory_bytes = mandatory;
    p->optional_budget_bytes = p->budget_bytes - mandatory;
    p->free_bytes = p->optional_budget_bytes - p->selected_optional_bytes;
    return 0;
}

static inline int coli_resource_plan_commit_optional(
        ColiResourcePlan *plan, ColiResourceTier tier, uint64_t bytes) {
    if (!plan || tier < 0 || tier >= COLI_RESOURCE_TIER_COUNT) return -1;
    ColiResourceTierPlan *p = &plan->tier[tier];
    if (bytes > p->free_bytes) return -1;
    p->selected_optional_bytes += bytes;
    p->free_bytes -= bytes;
    return 0;
}

static inline int coli_resource_plan_release_optional(
        ColiResourcePlan *plan, ColiResourceTier tier, uint64_t bytes) {
    if (!plan || tier < 0 || tier >= COLI_RESOURCE_TIER_COUNT) return -1;
    ColiResourceTierPlan *p = &plan->tier[tier];
    if (bytes > p->selected_optional_bytes) return -1;
    p->selected_optional_bytes -= bytes;
    p->free_bytes += bytes;
    return 0;
}

/* UMA is one physical envelope. Callers should put a CPU+Metal shared object in
 * COLI_RESOURCE_TIER_UMA once, not duplicate it into HOST and DEVICE. Explicit
 * CUDA mirrors belong in their distinct HOST/PINNED_HOST/DEVICE envelopes. */
static inline uint64_t coli_resource_plan_committed(
        const ColiResourcePlan *plan, ColiResourceTier tier) {
    if (!plan || tier < 0 || tier >= COLI_RESOURCE_TIER_COUNT) return 0;
    const ColiResourceTierPlan *p = &plan->tier[tier];
    return p->mandatory_bytes + p->selected_optional_bytes;
}

#endif
