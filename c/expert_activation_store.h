#ifndef COLIBRI_EXPERT_ACTIVATION_STORE_H
#define COLIBRI_EXPERT_ACTIVATION_STORE_H

#include "expert_activation.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Temporary/shared adapter for stores that have not yet moved their physical
 * residency into #95. It consumes the same logical activation observer contract
 * that the shared residency manager will own, without teaching the model engine
 * about tracker storage.
 *
 * Wrapping is best-effort: allocation failure leaves the original store usable
 * and must never fail model execution. The adapter owns `inner` only after a
 * successful wrap and destroys it through its original ops.
 */
typedef struct {
    ColiExpertStore *inner;
    ColiExpertActivationTracker tracker;
    ColiExpertActivationEntry *entries;
} ColiExpertActivationStoreState;

static inline int coli_expert_activation_store_lookup(
    ColiExpertStore *store, ColiExpertKey key, ColiExpertView *view) {
    ColiExpertActivationStoreState *state = store ? store->state : NULL;
    return state && state->inner
        ? coli_expert_lookup(state->inner, key, view) : -1;
}

static inline int coli_expert_activation_store_lookup_context(
    ColiExpertStore *store, ColiExpertKey key,
    const ColiExpertRequestContext *context, ColiExpertView *view) {
    ColiExpertActivationStoreState *state = store ? store->state : NULL;
    return state && state->inner
        ? coli_expert_lookup_context(state->inner, key, context, view) : -1;
}

static inline void coli_expert_activation_store_release(
    ColiExpertStore *store, ColiExpertView *view) {
    ColiExpertActivationStoreState *state = store ? store->state : NULL;
    if (state && state->inner) coli_expert_release(state->inner, view);
}

static inline int coli_expert_activation_store_prefetch(
    ColiExpertStore *store, const ColiExpertKey *keys, size_t count) {
    ColiExpertActivationStoreState *state = store ? store->state : NULL;
    if (!state || !state->inner || !state->inner->ops ||
        !state->inner->ops->prefetch)
        return 0;
    return state->inner->ops->prefetch(state->inner, keys, count);
}

static inline void coli_expert_activation_store_stats(
    const ColiExpertStore *store, ColiExpertStoreStats *stats) {
    ColiExpertActivationStoreState *state = store ? store->state : NULL;
    if (!state || !stats) return;
    if (state->inner && state->inner->ops && state->inner->ops->stats)
        state->inner->ops->stats(state->inner, stats);
    stats->logical_activations = state->tracker.total_logical_activations;
    stats->activation_observations = state->tracker.total_observations;
    stats->activation_keys = state->tracker.used;
    stats->activation_dropped_new_keys = state->tracker.dropped_new_keys;
}

static inline void coli_expert_activation_store_observe(
    ColiExpertStore *store,
    const struct ColiExpertActivationSample *samples,
    size_t count) {
    ColiExpertActivationStoreState *state = store ? store->state : NULL;
    if (!state || !samples || !count) return;
    coli_expert_activation_observe_many(
        &state->tracker, (const ColiExpertActivationSample *)samples, count);
}

static inline void coli_expert_activation_store_destroy(ColiExpertStore *store) {
    ColiExpertActivationStoreState *state = store ? store->state : NULL;
    if (state) {
        if (state->inner && state->inner->ops && state->inner->ops->destroy)
            state->inner->ops->destroy(state->inner);
        free(state->entries);
        free(state);
    }
    free(store);
}

static inline const ColiExpertStoreOps *coli_expert_activation_store_ops(void) {
    static const ColiExpertStoreOps ops = {
        .lookup = coli_expert_activation_store_lookup,
        .release = coli_expert_activation_store_release,
        .prefetch = coli_expert_activation_store_prefetch,
        .stats = coli_expert_activation_store_stats,
        .destroy = coli_expert_activation_store_destroy,
        .lookup_context = coli_expert_activation_store_lookup_context,
        .observe_activations = coli_expert_activation_store_observe,
    };
    return &ops;
}

static inline size_t coli_expert_activation_store_capacity(size_t key_hint) {
    if (!key_hint) return 0;
    if (key_hint > SIZE_MAX / 2) return 0;
    size_t wanted = key_hint * 2;
    size_t capacity = 2;
    while (capacity < wanted) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity <<= 1;
    }
    return capacity;
}

/* Returns 1 when wrapped, 0 when best-effort tracking could not be allocated,
 * and -1 for invalid arguments. On any non-1 result `inner` remains owned by
 * the caller and `*output` is unchanged. */
static inline int coli_expert_activation_store_wrap(
    ColiExpertStore *inner, size_t logical_key_hint, ColiExpertStore **output) {
    if (!inner || !output) return -1;
    size_t capacity = coli_expert_activation_store_capacity(logical_key_hint);
    if (!capacity || capacity > SIZE_MAX / sizeof(ColiExpertActivationEntry))
        return 0;

    ColiExpertStore *outer = calloc(1, sizeof(*outer));
    ColiExpertActivationStoreState *state = calloc(1, sizeof(*state));
    ColiExpertActivationEntry *entries = calloc(capacity, sizeof(*entries));
    if (!outer || !state || !entries) {
        free(entries);
        free(state);
        free(outer);
        return 0;
    }
    if (coli_expert_activation_init(&state->tracker, entries, capacity) != 0) {
        free(entries);
        free(state);
        free(outer);
        return 0;
    }
    state->inner = inner;
    state->entries = entries;
    outer->ops = coli_expert_activation_store_ops();
    outer->state = state;
    *output = outer;
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_EXPERT_ACTIVATION_STORE_H */
