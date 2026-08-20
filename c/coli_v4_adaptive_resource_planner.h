#ifndef COLI_V4_ADAPTIVE_RESOURCE_PLANNER_H
#define COLI_V4_ADAPTIVE_RESOURCE_PLANNER_H

#include "expert_residency_policy.h"
#include "resource_planner.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * V4 adapter from shared activation telemetry to the global resource planner.
 *
 * The physical expert store still owns generations, leases, slot storage and
 * I/O. This adapter owns only the optional-byte decision. In balanced mode the
 * per-layer persistent array becomes a lazy logical admission surface: every
 * expert has a metadata slot, but backing storage is allocated only when the
 * global planner selects that expert. Unselected experts continue through the
 * global transient pool.
 *
 * Dense package tensors are immutable borrowed allocations with explicit borrow
 * refs. Live borrows are a mandatory floor for the current replan; idle dense
 * entries are reclaimable and compete with persistent experts in the same UMA
 * budget. This preserves a hard RAM envelope without giving dense first-touch
 * residency a permanent advantage over subsequently observed hot experts.
 */
typedef struct {
    int enabled;
    size_t key_count;
    size_t candidate_capacity;
    ColiResourceCandidate *candidates;
    unsigned char *candidate_selected;
    unsigned char *expert_selected;
    uint64_t last_replan_epoch;
    uint64_t replan_count;
    uint64_t bootstrap_expert_bytes;
    uint64_t selected_dense_bytes;
    uint64_t selected_expert_bytes;
} ColiV4AdaptiveResourcePlanner;

static inline ColiResourceTier coli_v4_adaptive_resource_tier(void) {
#ifdef __APPLE__
    return COLI_RESOURCE_TIER_UMA;
#else
    return COLI_RESOURCE_TIER_HOST;
#endif
}

static inline int coli_v4_adaptive_resource_key_index(
    const State *inner, ColiExpertKey key, size_t *index) {
    if (!inner || !index || key.layer < 0 || key.layer >= inner->layers ||
        key.expert < 0 || key.expert >= inner->experts)
        return 0;
    *index = (size_t)key.layer * (size_t)inner->experts +
             (size_t)key.expert;
    return 1;
}

static inline int coli_v4_adaptive_resource_slot_resident(
    const State *inner, ColiExpertKey key) {
    if (!inner) return 0;
    Slot *slots = persistent_for((State *)inner, key.layer);
    for (int i = 0; slots && i < inner->persistent_slots_per_layer; i++) {
        const Slot *slot = &slots[i];
        if (slot->state == SLOT_RESIDENT && same_key(slot, key)) return 1;
    }
    return 0;
}

static inline void coli_v4_adaptive_resource_reset_slot_locked(
    State *inner, Slot *slot) {
    if (!inner || !slot || slot->refs || slot->state == SLOT_LOADING) return;
    if (slot->data) {
#ifdef COLI_METAL
        coli_metal_unregister(slot->data);
#endif
        compat_aligned_free(slot->data);
        slot->data = NULL;
        if (inner->stats.resident_bytes >= inner->slot_bytes)
            inner->stats.resident_bytes -= inner->slot_bytes;
        else
            inner->stats.resident_bytes = 0;
    }
    slot->generation++;
    if (!slot->generation) slot->generation = 1;
    slot->layer = -1;
    slot->expert = -1;
    /* Deliberately not EMPTY: an unselected rank-0 expert must not claim a
     * persistent slot merely because metadata is unused. The base chooser sees
     * this as a zero-value victim and admits only a planner-selected rank > 0. */
    slot->state = SLOT_RESIDENT;
    slot->last_use = 0;
}

static inline uint64_t coli_v4_adaptive_resource_forced_bytes_locked(
    ColiV4AdaptiveResourcePlanner *planner, State *inner) {
    uint64_t forced = 0;
    if (!planner || !inner) return 0;
    for (int layer = 0; layer < inner->layers; layer++) {
        Slot *slots = persistent_for(inner, layer);
        for (int i = 0; slots && i < inner->persistent_slots_per_layer; i++) {
            Slot *slot = &slots[i];
            if (!slot->data || (!slot->refs && slot->state != SLOT_LOADING))
                continue;
            size_t index = 0;
            if (coli_v4_adaptive_resource_key_index(
                    inner, (ColiExpertKey){slot->layer, slot->expert}, &index))
                planner->expert_selected[index] = 1;
            forced = coli_resource_saturating_add(forced, inner->slot_bytes);
        }
    }
    return forced;
}

static inline void coli_v4_adaptive_resource_reclaim_locked(
    ColiV4AdaptiveResourcePlanner *planner, State *inner) {
    if (!planner || !inner) return;
    for (int layer = 0; layer < inner->layers; layer++) {
        Slot *slots = persistent_for(inner, layer);
        for (int i = 0; slots && i < inner->persistent_slots_per_layer; i++) {
            Slot *slot = &slots[i];
            size_t index = 0;
            int selected = coli_v4_adaptive_resource_key_index(
                inner, (ColiExpertKey){slot->layer, slot->expert}, &index)
                ? planner->expert_selected[index] != 0 : 0;
            if (!selected)
                coli_v4_adaptive_resource_reset_slot_locked(inner, slot);
        }
    }
}

/* Dense capacity is fungible at this layer. Powers-of-two budget chunks provide
 * an exact ratio-1 baseline with at most 64 candidates, so arbitrary expert
 * slot sizes can displace dense bytes without exploding planner work. */
static inline size_t coli_v4_adaptive_resource_append_dense(
    ColiV4AdaptiveResourcePlanner *planner, size_t count,
    uint64_t optional_budget) {
    if (!planner || !optional_budget) return count;
    uint32_t id = 0;
    for (int bit = 63; bit >= 0; bit--) {
        uint64_t bytes = UINT64_C(1) << (unsigned)bit;
        if (bytes > optional_budget) continue;
        if (count >= planner->candidate_capacity) return count;
        ColiResourceBenefitEstimate estimate = {
            .kind = COLI_RESOURCE_DENSE_TENSOR,
            .id = id++,
            .resident_bytes = bytes,
            .reuse_weight = 1,
            .bytes_per_miss = bytes,
            .exposed_ns_per_miss = 0,
        };
        (void)coli_resource_candidate_from_benefit(
            &estimate, &planner->candidates[count++]);
    }
    return count;
}

/* Before the activation tracker has enough evidence to rank experts, dense
 * first-touch admissions must not consume the entire optional pool. Reserve a
 * small fungible bootstrap window: roughly one expert record per layer, capped
 * at 25% of optional memory. Nothing is allocated here; the bytes simply stay
 * out of the dense admission ceiling until the first global replan can assign
 * them using observed expert reuse. This avoids a permanent first-mover
 * advantage for immutable dense borrows without imposing a fixed long-term
 * dense/expert split. */
static inline uint64_t coli_v4_adaptive_resource_bootstrap_bytes(
    const State *inner, uint64_t optional_bytes) {
    if (!inner || inner->layers < 1 || !inner->slot_bytes ||
        optional_bytes < inner->slot_bytes)
        return 0;
    uint64_t layer_target = optional_bytes;
    if ((uint64_t)inner->layers <= UINT64_MAX / inner->slot_bytes)
        layer_target = (uint64_t)inner->layers * inner->slot_bytes;
    uint64_t reserve = layer_target;
    uint64_t cap = optional_bytes / 4;
    if (reserve > cap) reserve = cap;
    reserve -= reserve % inner->slot_bytes;
    return reserve;
}

static inline int coli_v4_adaptive_resource_init(
    ColiV4AdaptiveResourcePlanner *planner, State *inner) {
    if (!planner) return -1;
    memset(planner, 0, sizeof(*planner));
    if (!inner || inner->legacy_layout || inner->layers < 1 ||
        inner->experts < 1 || !inner->slot_bytes ||
        (size_t)inner->layers > SIZE_MAX / (size_t)inner->experts)
        return 0;

    size_t key_count = (size_t)inner->layers * (size_t)inner->experts;
    if (key_count > UINT32_MAX || key_count > SIZE_MAX - 64)
        return -1;
    size_t candidate_capacity = key_count + 64;
    if (candidate_capacity > SIZE_MAX / sizeof(ColiResourceCandidate))
        return -1;

    ColiResourceCandidate *candidates = calloc(
        candidate_capacity, sizeof(*candidates));
    unsigned char *candidate_selected = calloc(candidate_capacity, 1);
    unsigned char *expert_selected = calloc(key_count, 1);
    if (!candidates || !candidate_selected || !expert_selected) {
        free(expert_selected);
        free(candidate_selected);
        free(candidates);
        return -1;
    }

    if (key_count > (size_t)(INT_MAX - inner->transient_slots) ||
        key_count > SIZE_MAX / sizeof(Slot)) {
        free(expert_selected);
        free(candidate_selected);
        free(candidates);
        return -1;
    }
    size_t total_slots = key_count + (size_t)inner->transient_slots;
    if (total_slots > SIZE_MAX / sizeof(Slot)) {
        free(expert_selected);
        free(candidate_selected);
        free(candidates);
        return -1;
    }
    Slot *slots = calloc(total_slots, sizeof(*slots));
    if (!slots) {
        free(expert_selected);
        free(candidate_selected);
        free(candidates);
        return -1;
    }

    /* No lookup can occur before the outer store is returned, so the base
     * metadata allocated during open has no backing expert storage or leases. */
    free(inner->slots);
    inner->slots = slots;
    inner->persistent_slots_per_layer = inner->experts;
    inner->total_slots = (int)total_slots;
    for (int layer = 0; layer < inner->layers; layer++) {
        Slot *persistent = persistent_for(inner, layer);
        for (int expert = 0; expert < inner->experts; expert++) {
            persistent[expert].layer = -1;
            persistent[expert].expert = -1;
            persistent[expert].home_layer = layer;
            persistent[expert].tier = SLOT_TIER_PERSISTENT;
            persistent[expert].state = SLOT_RESIDENT;
        }
    }
    Slot *transient = transient_base(inner);
    for (int i = 0; i < inner->transient_slots; i++) {
        transient[i].layer = -1;
        transient[i].expert = -1;
        transient[i].home_layer = -1;
        transient[i].tier = SLOT_TIER_TRANSIENT;
        transient[i].state = SLOT_EMPTY;
    }

    uint64_t transient_bytes = (uint64_t)inner->transient_slots *
                               inner->slot_bytes;
    if (transient_bytes > inner->offered_cache_bytes) {
        free(inner->slots);
        inner->slots = NULL;
        free(expert_selected);
        free(candidate_selected);
        free(candidates);
        return -1;
    }
    uint64_t optional_bytes = inner->offered_cache_bytes - transient_bytes;
    uint64_t bootstrap_expert_bytes =
        coli_v4_adaptive_resource_bootstrap_bytes(inner, optional_bytes);
    inner->stats.capacity_bytes = coli_resource_saturating_add(
        transient_bytes, bootstrap_expert_bytes);
    inner->dense_cache_budget_bytes = optional_bytes - bootstrap_expert_bytes;
    (void)coli_v4_dense_cache_set_budget(inner->dense_cache_budget_bytes);

    planner->key_count = key_count;
    planner->candidate_capacity = candidate_capacity;
    planner->candidates = candidates;
    planner->candidate_selected = candidate_selected;
    planner->expert_selected = expert_selected;
    planner->bootstrap_expert_bytes = bootstrap_expert_bytes;
    planner->selected_dense_bytes = inner->dense_cache_budget_bytes;
    planner->enabled = 1;
    return 0;
}

static inline void coli_v4_adaptive_resource_destroy(
    ColiV4AdaptiveResourcePlanner *planner) {
    if (!planner) return;
    free(planner->expert_selected);
    free(planner->candidate_selected);
    free(planner->candidates);
    memset(planner, 0, sizeof(*planner));
}

static inline int coli_v4_adaptive_resource_selected(
    const ColiV4AdaptiveResourcePlanner *planner, size_t key_index) {
    return !planner || !planner->enabled ||
        (key_index < planner->key_count && planner->expert_selected[key_index]);
}

/* A resource decision must be based on one coherent logical-token span. The V4
 * activation overlay publishes one layer at a time but assigns every layer in
 * the same routed span the same epoch. Replanning before all layers have reached
 * that epoch lets layer 0 make a global UMA decision with no evidence for the
 * other 42 layers. Detect completion from the tracker itself so the planner
 * remains independent of callback ordering and loader concurrency. */
static inline int coli_v4_adaptive_resource_epoch_complete(
    const ColiExpertActivationTracker *tracker, const State *inner,
    uint64_t current_epoch) {
    if (!tracker || !inner || inner->layers < 1 || !current_epoch) return 0;

    if (inner->layers <= 64) {
        uint64_t seen = 0;
        for (size_t slot = 0; slot < tracker->capacity; slot++) {
            const ColiExpertActivationEntry *entry = &tracker->entries[slot];
            if (!entry->hash_tag || entry->last_epoch != current_epoch ||
                entry->key.layer < 0 || entry->key.layer >= inner->layers)
                continue;
            seen |= UINT64_C(1) << (unsigned)entry->key.layer;
        }
        uint64_t expected = inner->layers == 64
            ? UINT64_MAX
            : (UINT64_C(1) << (unsigned)inner->layers) - UINT64_C(1);
        return seen == expected;
    }

    /* V4 is currently 43 layers, so the mask path above is the hot path. Keep a
     * portable fallback for future deeper variants without heap allocation. */
    for (int layer = 0; layer < inner->layers; layer++) {
        int found = 0;
        for (size_t slot = 0; slot < tracker->capacity; slot++) {
            const ColiExpertActivationEntry *entry = &tracker->entries[slot];
            if (entry->hash_tag && entry->last_epoch == current_epoch &&
                entry->key.layer == layer) {
                found = 1;
                break;
            }
        }
        if (!found) return 0;
    }
    return 1;
}

static inline int coli_v4_adaptive_resource_should_replan(
    const ColiV4AdaptiveResourcePlanner *planner,
    const ColiExpertActivationTracker *tracker,
    uint64_t current_epoch,
    const ColiExpertResidencyPolicyConfig *policy) {
    if (!planner || !planner->enabled || !tracker || !policy) return 0;
    if (!planner->replan_count)
        return tracker->total_logical_activations >=
               policy->planner_confidence_mass;
    /* recency_quantum_epochs is the activity half-life quantum, not an allocator
     * cadence. Once a complete new token span is available, use it immediately. */
    return current_epoch > planner->last_replan_epoch;
}

/* Among the optional experts tentatively selected by the shared allocator,
 * identify the least valuable one using the inverse of the allocator's exact
 * deterministic ordering. This is used when a dense floor or transfer limit
 * leaves less room for experts than the unconstrained global selection. */
static inline size_t coli_v4_adaptive_resource_weakest_selected_expert(
    const ColiV4AdaptiveResourcePlanner *planner,
    size_t dense_count, size_t count) {
    size_t weakest = count;
    if (!planner) return weakest;
    for (size_t i = dense_count; i < count; i++) {
        if (!planner->candidate_selected[i]) continue;
        const ColiResourceCandidate *candidate = &planner->candidates[i];
        if (candidate->kind != COLI_RESOURCE_PERSISTENT_EXPERT) continue;
        if (weakest == count) {
            weakest = i;
            continue;
        }
        const ColiResourceCandidate *incumbent = &planner->candidates[weakest];
        int ratio = coli_resource_ratio_compare(
            candidate->expected_bytes_avoided, candidate->resident_bytes,
            incumbent->expected_bytes_avoided, incumbent->resident_bytes);
        if (ratio < 0 ||
            (ratio == 0 &&
             (candidate->id > incumbent->id ||
              (candidate->id == incumbent->id && i > weakest))))
            weakest = i;
    }
    return weakest;
}

/* If a dense-growth rate limit leaves spare UMA, keep it productive by filling
 * with the strongest expert candidates that fit. Otherwise a controller meant
 * to reduce churn would accidentally create an unallocated-memory trough on
 * every move back toward dense residency. */
static inline size_t coli_v4_adaptive_resource_strongest_unselected_expert(
    const ColiV4AdaptiveResourcePlanner *planner,
    size_t dense_count, size_t count, uint64_t available) {
    size_t strongest = count;
    if (!planner || !available) return strongest;
    for (size_t i = dense_count; i < count; i++) {
        if (planner->candidate_selected[i]) continue;
        const ColiResourceCandidate *candidate = &planner->candidates[i];
        if (candidate->kind != COLI_RESOURCE_PERSISTENT_EXPERT ||
            !candidate->resident_bytes || candidate->resident_bytes > available)
            continue;
        if (strongest == count) {
            strongest = i;
            continue;
        }
        const ColiResourceCandidate *incumbent = &planner->candidates[strongest];
        int ratio = coli_resource_ratio_compare(
            candidate->expected_bytes_avoided, candidate->resident_bytes,
            incumbent->expected_bytes_avoided, incumbent->resident_bytes);
        if (ratio > 0 ||
            (ratio == 0 &&
             (candidate->id < incumbent->id ||
              (candidate->id == incumbent->id && i < strongest))))
            strongest = i;
    }
    return strongest;
}

/* A noisy one-token benefit estimate must not migrate the entire UMA pool from
 * dense to experts (or back) in one replan. Use the topology-derived bootstrap
 * reserve -- approximately one expert record per target layer -- as the maximum
 * class transfer per complete logical-token sweep. Mandatory live dense pins or
 * in-flight expert leases may still force a larger move; safety beats damping. */
static inline uint64_t coli_v4_adaptive_resource_limit_dense_transfer(
    const ColiV4AdaptiveResourcePlanner *planner,
    uint64_t desired, uint64_t floor, uint64_t ceiling) {
    if (floor > ceiling) return ceiling;
    if (desired < floor) desired = floor;
    if (desired > ceiling) desired = ceiling;
    if (!planner || !planner->bootstrap_expert_bytes) return desired;

    uint64_t previous = planner->selected_dense_bytes;
    uint64_t quantum = planner->bootstrap_expert_bytes;

    /* A changed mandatory envelope may put the previous budget outside the new
     * legal range. In that case move directly to the nearest safe boundary. */
    if (previous < floor) return floor;
    if (previous > ceiling) return ceiling;

    uint64_t low = previous > quantum ? previous - quantum : 0;
    uint64_t high = coli_resource_saturating_add(previous, quantum);
    if (low < floor) low = floor;
    if (high > ceiling) high = ceiling;
    if (desired < low) desired = low;
    if (desired > high) desired = high;
    return desired;
}

/* Called with the base expert-store mutex held. */
static inline int coli_v4_adaptive_resource_replan_locked(
    ColiV4AdaptiveResourcePlanner *planner, State *inner,
    const ColiExpertActivationTracker *tracker,
    uint64_t current_epoch,
    const ColiExpertResidencyPolicyConfig *policy) {
    if (!coli_v4_adaptive_resource_epoch_complete(
            tracker, inner, current_epoch) ||
        !coli_v4_adaptive_resource_should_replan(
            planner, tracker, current_epoch, policy))
        return 0;

    ColiV4DenseCacheStats dense = {0};
    coli_v4_dense_cache_stats(&dense);
    uint64_t transient_bytes = (uint64_t)inner->transient_slots *
                               inner->slot_bytes;
    if (transient_bytes > inner->offered_cache_bytes ||
        dense.resident_bytes > inner->offered_cache_bytes - transient_bytes ||
        dense.pinned_bytes > dense.resident_bytes)
        return -1;

    memset(planner->expert_selected, 0, planner->key_count);
    uint64_t forced_expert_bytes =
        coli_v4_adaptive_resource_forced_bytes_locked(planner, inner);
    if (forced_expert_bytes > inner->offered_cache_bytes - transient_bytes -
                              dense.pinned_bytes)
        return -1;

    ColiResourcePlan plan;
    coli_resource_plan_init(&plan);
    ColiResourceTier tier = coli_v4_adaptive_resource_tier();
    if (coli_resource_plan_set_budget(
            &plan, tier, inner->offered_cache_bytes) != 0 ||
        coli_resource_plan_reserve_mandatory(
            &plan, tier, transient_bytes) != 0 ||
        coli_resource_plan_reserve_mandatory(
            &plan, tier, dense.pinned_bytes) != 0 ||
        coli_resource_plan_reserve_mandatory(
            &plan, tier, forced_expert_bytes) != 0)
        return -1;

    size_t count = coli_v4_adaptive_resource_append_dense(
        planner, 0, plan.tier[tier].free_bytes);
    size_t dense_count = count;
    uint64_t exposed_ns_per_miss =
        coli_expert_store_stats_exposed_ns_per_miss(&inner->stats);

    /* V4 currently has comparable stored-byte measurements for dense and expert
     * misses, but no matching exposed-I/O timing for dense package tensors.
     * Select in bytes mode until that measurement exists; do not give dense a
     * fabricated latency. Experts at or below the dense ratio-1 baseline can be
     * omitted exactly because deterministic kind ordering would reject them. */
    for (size_t slot = 0; slot < tracker->capacity; slot++) {
        const ColiExpertActivationEntry *entry = &tracker->entries[slot];
        if (!entry->hash_tag) continue;
        size_t key_index = 0;
        if (!coli_v4_adaptive_resource_key_index(
                inner, entry->key, &key_index) ||
            planner->expert_selected[key_index])
            continue;

        ColiExpertResidencyCandidate expert =
            coli_expert_residency_policy_candidate(
                entry, current_epoch, inner->slot_bytes, 1,
                coli_v4_adaptive_resource_slot_resident(inner, entry->key),
                policy);
        ColiResourceBenefitEstimate estimate;
        if (coli_expert_residency_policy_resource_estimate(
                &expert, (uint32_t)key_index, inner->record_bytes,
                exposed_ns_per_miss, &estimate) != 0)
            continue;
        ColiResourceCandidate candidate;
        if (coli_resource_candidate_from_benefit(&estimate, &candidate) != 0)
            continue;
        if (coli_resource_ratio_compare(
                candidate.expected_bytes_avoided, candidate.resident_bytes,
                1, 1) <= 0)
            continue;
        if (count >= planner->candidate_capacity) return -1;
        planner->candidates[count++] = candidate;
    }

    ColiResourceSelection selection;
    if (coli_resource_plan_select_optional(
            &plan, tier, planner->candidates, count,
            COLI_RESOURCE_VALUE_BYTES,
            planner->candidate_selected, &selection) != 0)
        return -1;

    uint64_t selected_dense = 0;
    uint64_t selected_optional_experts = 0;
    for (size_t i = 0; i < count; i++) {
        if (!planner->candidate_selected[i]) continue;
        const ColiResourceCandidate *candidate = &planner->candidates[i];
        if (i < dense_count || candidate->kind == COLI_RESOURCE_DENSE_TENSOR) {
            selected_dense = coli_resource_saturating_add(
                selected_dense, candidate->resident_bytes);
        } else if (candidate->kind == COLI_RESOURCE_PERSISTENT_EXPERT &&
                   candidate->id < planner->key_count) {
            selected_optional_experts = coli_resource_saturating_add(
                selected_optional_experts, candidate->resident_bytes);
        }
    }

    /* Commit dense first. The unconstrained allocator supplies the desired class
     * split; rate-limit that split to one topology-derived transfer quantum per
     * complete token sweep. Live dense pins and forced expert bytes remain hard
     * bounds. Then reconcile expert selections against the actual dense budget. */
    uint64_t requested_dense = coli_resource_saturating_add(
        dense.pinned_bytes, selected_dense);
    uint64_t dense_ceiling = inner->offered_cache_bytes - transient_bytes -
                             forced_expert_bytes;
    requested_dense = coli_v4_adaptive_resource_limit_dense_transfer(
        planner, requested_dense, dense.pinned_bytes, dense_ceiling);
    uint64_t dense_budget = coli_v4_dense_cache_set_budget(requested_dense);
    if (dense_budget > inner->offered_cache_bytes - transient_bytes ||
        forced_expert_bytes > inner->offered_cache_bytes - transient_bytes -
                              dense_budget)
        return -1;
    uint64_t optional_expert_limit = inner->offered_cache_bytes -
        transient_bytes - dense_budget - forced_expert_bytes;
    while (selected_optional_experts > optional_expert_limit) {
        size_t weakest = coli_v4_adaptive_resource_weakest_selected_expert(
            planner, dense_count, count);
        if (weakest == count) return -1;
        const ColiResourceCandidate *candidate = &planner->candidates[weakest];
        planner->candidate_selected[weakest] = 0;
        if (selected_optional_experts >= candidate->resident_bytes)
            selected_optional_experts -= candidate->resident_bytes;
        else
            selected_optional_experts = 0;
    }
    while (selected_optional_experts < optional_expert_limit) {
        uint64_t available = optional_expert_limit - selected_optional_experts;
        size_t strongest = coli_v4_adaptive_resource_strongest_unselected_expert(
            planner, dense_count, count, available);
        if (strongest == count) break;
        const ColiResourceCandidate *candidate = &planner->candidates[strongest];
        planner->candidate_selected[strongest] = 1;
        selected_optional_experts = coli_resource_saturating_add(
            selected_optional_experts, candidate->resident_bytes);
    }

    for (size_t i = dense_count; i < count; i++) {
        if (!planner->candidate_selected[i]) continue;
        const ColiResourceCandidate *candidate = &planner->candidates[i];
        if (candidate->kind == COLI_RESOURCE_PERSISTENT_EXPERT &&
            candidate->id < planner->key_count)
            planner->expert_selected[candidate->id] = 1;
    }
    coli_v4_adaptive_resource_reclaim_locked(planner, inner);

    inner->dense_cache_budget_bytes = dense_budget;
    planner->selected_dense_bytes = dense_budget;
    planner->selected_expert_bytes = coli_resource_saturating_add(
        forced_expert_bytes, selected_optional_experts);
    inner->stats.capacity_bytes = coli_resource_saturating_add(
        transient_bytes, planner->selected_expert_bytes);
    planner->last_replan_epoch = current_epoch;
    planner->replan_count++;
    return 1;
}

#endif /* COLI_V4_ADAPTIVE_RESOURCE_PLANNER_H */
