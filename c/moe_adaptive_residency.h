#ifndef COLIBRI_MOE_ADAPTIVE_RESIDENCY_H
#define COLIBRI_MOE_ADAPTIVE_RESIDENCY_H

#include "expert_activation.h"
#include "expert_residency_policy.h"
#include "resource_planner.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Shared adaptive-residency controller for routed MoE experts.
 *
 * Engines provide only semantic observations (logical routes before physical
 * union/batching), physical record/resident byte costs, and an optional
 * resident-state callback. This controller owns the model-neutral activation
 * table, common policy, replan cadence and selected-expert map. Physical slots,
 * leases, I/O and backend residency remain below the engine/store adapter.
 *
 * Resource allocation and expert identity are intentionally separate:
 *   - common-horizon reuse determines HOW MANY expert bytes are worthwhile
 *     relative to other optional resource classes;
 *   - phase-weighted hotness + residency hysteresis determines WHICH experts
 *     occupy an expert byte envelope.
 *
 * The controller exposes both seams. V4 can append expert benefit candidates
 * beside dense/prefix candidates and then apply the resulting expert byte
 * envelope through the hot-expert selector. Qwen uses the expert-only
 * convenience selector. Neither engine owns the policy math.
 */
typedef int (*ColiMoeAdaptiveResidentFn)(void *context, ColiExpertKey key);

typedef struct {
    int layers;
    int experts;
    size_t key_count;
    ColiExpertActivationTracker tracker;
    ColiExpertActivationEntry *entries;
    unsigned char *selected;
    ColiResourceCandidate *scratch_candidates;
    ColiExpertResidencyCandidate *scratch_experts;
    unsigned char *scratch_selected;
    uint64_t current_epoch;
    uint64_t last_replan_epoch;
    uint64_t replan_count;
    ColiExpertResidencyPolicyConfig policy;
} ColiMoeAdaptiveResidency;

static inline size_t coli_moe_adaptive_tracker_capacity(size_t keys) {
    if (!keys || keys > SIZE_MAX / 2) return 0;
    size_t wanted = keys * 2;
    size_t capacity = 2;
    while (capacity < wanted) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity <<= 1;
    }
    return capacity;
}

static inline int coli_moe_adaptive_key_index(
    const ColiMoeAdaptiveResidency *state, ColiExpertKey key, size_t *index) {
    if (!state || !index || key.layer < 0 || key.layer >= state->layers ||
        key.expert < 0 || key.expert >= state->experts)
        return 0;
    *index = (size_t)key.layer * (size_t)state->experts +
             (size_t)key.expert;
    return 1;
}

static inline int coli_moe_adaptive_init(
    ColiMoeAdaptiveResidency *state, int layers, int experts) {
    if (!state) return -1;
    memset(state, 0, sizeof(*state));
    if (layers < 1 || experts < 1 ||
        (size_t)layers > SIZE_MAX / (size_t)experts)
        return -1;

    size_t key_count = (size_t)layers * (size_t)experts;
    size_t capacity = coli_moe_adaptive_tracker_capacity(key_count);
    if (!capacity || capacity > SIZE_MAX / sizeof(*state->entries) ||
        key_count > SIZE_MAX / sizeof(*state->scratch_candidates) ||
        key_count > SIZE_MAX / sizeof(*state->scratch_experts))
        return -1;

    state->entries = calloc(capacity, sizeof(*state->entries));
    state->selected = calloc(key_count, 1);
    state->scratch_candidates = calloc(
        key_count, sizeof(*state->scratch_candidates));
    state->scratch_experts = calloc(
        key_count, sizeof(*state->scratch_experts));
    state->scratch_selected = calloc(key_count, 1);
    if (!state->entries || !state->selected || !state->scratch_candidates ||
        !state->scratch_experts || !state->scratch_selected) {
        free(state->scratch_selected);
        free(state->scratch_experts);
        free(state->scratch_candidates);
        free(state->selected);
        free(state->entries);
        memset(state, 0, sizeof(*state));
        return -1;
    }
    if (coli_expert_activation_init(&state->tracker, state->entries, capacity) != 0) {
        free(state->scratch_selected);
        free(state->scratch_experts);
        free(state->scratch_candidates);
        free(state->selected);
        free(state->entries);
        memset(state, 0, sizeof(*state));
        return -1;
    }

    state->layers = layers;
    state->experts = experts;
    state->key_count = key_count;
    state->policy = coli_expert_residency_policy_default();
    return 0;
}

static inline void coli_moe_adaptive_destroy(ColiMoeAdaptiveResidency *state) {
    if (!state) return;
    free(state->scratch_selected);
    free(state->scratch_experts);
    free(state->scratch_candidates);
    free(state->selected);
    free(state->entries);
    memset(state, 0, sizeof(*state));
}

static inline int coli_moe_adaptive_observe_routes(
    ColiMoeAdaptiveResidency *state, int layer,
    const int *experts, size_t count, ColiExpertPhase phase,
    uint64_t epoch) {
    if (!state || !experts || layer < 0 || layer >= state->layers || !epoch)
        return -1;
    for (size_t i = 0; i < count; i++) {
        if (experts[i] < 0 || experts[i] >= state->experts) return -1;
        ColiExpertActivationSample sample = {
            .key = { layer, experts[i] },
            .phase = phase,
            .multiplicity = 1,
            .epoch = epoch,
        };
        if (coli_expert_activation_observe(&state->tracker, sample) < 0)
            return -1;
    }
    if (epoch > state->current_epoch) state->current_epoch = epoch;
    return 0;
}

static inline int coli_moe_adaptive_observe_samples(
    ColiMoeAdaptiveResidency *state,
    const ColiExpertActivationSample *samples, size_t count) {
    if (!state || (!samples && count)) return -1;
    for (size_t i = 0; i < count; i++) {
        size_t ignored = 0;
        if (!samples[i].epoch || !samples[i].multiplicity ||
            !coli_moe_adaptive_key_index(state, samples[i].key, &ignored))
            return -1;
        if (coli_expert_activation_observe(&state->tracker, samples[i]) < 0)
            return -1;
        if (samples[i].epoch > state->current_epoch)
            state->current_epoch = samples[i].epoch;
    }
    return 0;
}

/* Generic replan schedule. The first plan waits for confidence mass. Then one
 * plan per new logical-token epoch is allowed for one bounded confidence
 * window so short requests can reveal decode-hot routes; steady state returns
 * to the policy's decay quantum. Engine layer count never enters this logic. */
static inline int coli_moe_adaptive_should_replan(
    const ColiMoeAdaptiveResidency *state) {
    if (!state || !coli_expert_residency_policy_config_valid(&state->policy))
        return 0;
    if (!state->replan_count)
        return state->tracker.total_logical_activations >=
               state->policy.planner_confidence_mass;
    if (state->current_epoch <= state->last_replan_epoch) return 0;
    if (state->replan_count <= state->policy.planner_confidence_mass)
        return 1;
    return state->current_epoch - state->last_replan_epoch >=
           state->policy.recency_quantum_epochs;
}

static inline size_t coli_moe_adaptive_append_candidates(
    const ColiMoeAdaptiveResidency *state,
    ColiResourceCandidate *candidates, size_t count, size_t capacity,
    uint64_t resident_bytes, uint64_t bytes_per_miss,
    uint64_t exposed_ns_per_miss,
    ColiMoeAdaptiveResidentFn is_resident, void *resident_context) {
    if (!state || !candidates || !resident_bytes || !bytes_per_miss ||
        count > capacity)
        return count;

    uint64_t miss_cost_us = exposed_ns_per_miss / UINT64_C(1000);
    if (!miss_cost_us) miss_cost_us = 1;

    for (size_t slot = 0; slot < state->tracker.capacity; slot++) {
        const ColiExpertActivationEntry *entry = &state->tracker.entries[slot];
        if (!entry->hash_tag) continue;
        size_t key_index = 0;
        if (!coli_moe_adaptive_key_index(state, entry->key, &key_index))
            continue;
        int resident = is_resident
            ? is_resident(resident_context, entry->key) : 0;
        ColiExpertResidencyCandidate expert =
            coli_expert_residency_policy_candidate(
                entry, state->current_epoch, resident_bytes, miss_cost_us,
                resident, &state->policy);
        if (!expert.reuse_weight) continue;
        ColiResourceBenefitEstimate estimate;
        if (coli_expert_residency_policy_resource_estimate(
                &expert, (uint32_t)key_index, bytes_per_miss,
                exposed_ns_per_miss, &estimate) != 0)
            continue;
        if (count >= capacity) break;
        if (coli_resource_candidate_from_benefit(
                &estimate, &candidates[count]) == 0)
            count++;
    }
    return count;
}

static inline int coli_moe_adaptive_hot_qsort_compare(
    const void *left, const void *right) {
    const ColiExpertResidencyCandidate *a =
        (const ColiExpertResidencyCandidate *)left;
    const ColiExpertResidencyCandidate *b =
        (const ColiExpertResidencyCandidate *)right;
    int order = coli_expert_residency_policy_compare(a, b);
    return order > 0 ? -1 : order < 0 ? 1 : 0;
}

/* Apply an already-decided expert byte envelope to the phase-aware expert
 * policy. This is the generic HOW-MANY -> WHICH bridge for both expert-only and
 * mixed-resource plans. Only experts with positive common-horizon reuse are
 * eligible, so hotness cannot bypass cold-start confidence. */
static inline size_t coli_moe_adaptive_apply_expert_budget(
    ColiMoeAdaptiveResidency *state, uint64_t expert_budget_bytes,
    uint64_t resident_bytes, uint64_t exposed_ns_per_miss,
    ColiMoeAdaptiveResidentFn is_resident, void *resident_context) {
    if (!state || !resident_bytes) return 0;

    uint64_t miss_cost_us = exposed_ns_per_miss / UINT64_C(1000);
    if (!miss_cost_us) miss_cost_us = 1;
    size_t count = 0;
    for (size_t slot = 0; slot < state->tracker.capacity; slot++) {
        const ColiExpertActivationEntry *entry = &state->tracker.entries[slot];
        if (!entry->hash_tag) continue;
        size_t key_index = 0;
        if (!coli_moe_adaptive_key_index(state, entry->key, &key_index))
            continue;
        int resident = is_resident
            ? is_resident(resident_context, entry->key) : 0;
        ColiExpertResidencyCandidate candidate =
            coli_expert_residency_policy_candidate(
                entry, state->current_epoch, resident_bytes, miss_cost_us,
                resident, &state->policy);
        if (!candidate.reuse_weight || !candidate.score) continue;
        state->scratch_experts[count++] = candidate;
    }

    if (count > 1)
        qsort(state->scratch_experts, count, sizeof(*state->scratch_experts),
              coli_moe_adaptive_hot_qsort_compare);

    memset(state->selected, 0, state->key_count);
    size_t limit = (size_t)(expert_budget_bytes / resident_bytes);
    if (limit > count) limit = count;
    for (size_t i = 0; i < limit; i++) {
        size_t key_index = 0;
        if (coli_moe_adaptive_key_index(
                state, state->scratch_experts[i].key, &key_index))
            state->selected[key_index] = 1;
    }
    state->last_replan_epoch = state->current_epoch;
    state->replan_count++;
    return limit;
}

/* Apply exact expert identities from an external planner. Use this only when
 * that planner intentionally owns WHICH-expert ordering. Mixed resource plans
 * that use common-horizon reuse solely to determine an expert byte envelope
 * should instead call coli_moe_adaptive_apply_expert_budget(). */
static inline void coli_moe_adaptive_apply_selection(
    ColiMoeAdaptiveResidency *state,
    const ColiResourceCandidate *candidates,
    const unsigned char *candidate_selected, size_t count) {
    if (!state || (!candidates && count) || (!candidate_selected && count))
        return;
    memset(state->selected, 0, state->key_count);
    for (size_t i = 0; i < count; i++) {
        if (!candidate_selected[i]) continue;
        const ColiResourceCandidate *candidate = &candidates[i];
        if (candidate->kind == COLI_RESOURCE_PERSISTENT_EXPERT &&
            candidate->id < state->key_count)
            state->selected[candidate->id] = 1;
    }
    state->last_replan_epoch = state->current_epoch;
    state->replan_count++;
}

/* Convenience for engines whose non-expert memory is already mandatory.
 * Common-horizon reuse first decides how many expert bytes are justified; the
 * phase-aware policy then fills exactly that envelope with the hottest experts.
 * This prevents the global resource value scale from silently replacing the
 * expert-local decode weighting/hysteresis policy. */
static inline int coli_moe_adaptive_select_experts(
    ColiMoeAdaptiveResidency *state, uint64_t optional_budget_bytes,
    uint64_t resident_bytes, uint64_t bytes_per_miss,
    uint64_t exposed_ns_per_miss, ColiResourceValueMode mode,
    ColiMoeAdaptiveResidentFn is_resident, void *resident_context,
    ColiResourceSelection *selection) {
    if (!state || !selection || !coli_moe_adaptive_should_replan(state))
        return 0;

    size_t count = coli_moe_adaptive_append_candidates(
        state, state->scratch_candidates, 0, state->key_count,
        resident_bytes, bytes_per_miss, exposed_ns_per_miss,
        is_resident, resident_context);
    ColiResourceSelection allocation = {0};
    if (coli_resource_select(
            state->scratch_candidates, count, optional_budget_bytes, mode,
            state->scratch_selected, &allocation) != 0)
        return -1;

    size_t selected_count = coli_moe_adaptive_apply_expert_budget(
        state, allocation.selected_resident_bytes, resident_bytes,
        exposed_ns_per_miss, is_resident, resident_context);

    *selection = allocation;
    selection->selected_count = selected_count;
    selection->selected_resident_bytes = coli_resource_saturating_mul(
        (uint64_t)selected_count, resident_bytes);
    selection->expected_bytes_avoided = 0;
    selection->expected_exposed_ns_avoided = 0;
    for (size_t i = 0; i < selected_count; i++) {
        const ColiExpertResidencyCandidate *expert = &state->scratch_experts[i];
        selection->expected_bytes_avoided = coli_resource_saturating_add(
            selection->expected_bytes_avoided,
            coli_resource_saturating_mul(expert->reuse_weight, bytes_per_miss));
        selection->expected_exposed_ns_avoided = coli_resource_saturating_add(
            selection->expected_exposed_ns_avoided,
            coli_resource_saturating_mul(
                expert->reuse_weight, exposed_ns_per_miss));
    }
    return 1;
}

static inline int coli_moe_adaptive_selected(
    const ColiMoeAdaptiveResidency *state, ColiExpertKey key) {
    size_t index = 0;
    return state && coli_moe_adaptive_key_index(state, key, &index)
        ? state->selected[index] != 0 : 0;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_MOE_ADAPTIVE_RESIDENCY_H */
