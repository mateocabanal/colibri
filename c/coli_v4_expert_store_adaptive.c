/*
 * Adaptive-policy overlay for the current V4 COLI expert store.
 *
 * The physical store remains the owner of slots, loader concurrency, lease
 * generations, Metal registration and byte accounting. This translation unit
 * only adds the shared logical activation tracker/scorer at its policy seam.
 * The base store still uses its existing `usage[]` array for persistent
 * admission; before physical lookups begin for a routed layer we project the
 * shared deterministic policy rank into that array. This lets us change policy
 * without duplicating the store state machine while #95 becomes the common
 * physical residency owner.
 *
 * V4 package experts are uniform-size records with the same storage path, so
 * benefit/byte ordering is equivalent to policy hotness ordering. We choose an
 * artificial miss-cost equal to the record's 4 KiB allocation quanta; the
 * shared score therefore equals hotness exactly while still exercising the
 * common benefit/byte scorer. The projection reserves low bits so the base
 * store's legacy per-physical-lookup counter increment cannot perturb policy
 * ordering between routing observations.
 */
#include "expert_activation.h"
#include "expert_residency_policy.h"

#define coli_v4_coli_expert_store_open coli_v4_coli_expert_store_open_base
#include "coli_v4_expert_store.c"
#undef coli_v4_coli_expert_store_open

typedef struct {
    ColiExpertStore *inner;
    ColiExpertActivationTracker tracker;
    ColiExpertActivationEntry *entries;
    uint64_t *physical_usage;
    size_t key_count;
    uint64_t current_epoch;
    ColiExpertResidencyPolicyConfig policy;
} ColiV4AdaptiveExpertStoreState;

static size_t v4_adaptive_tracker_capacity(size_t keys) {
    if (!keys || keys > SIZE_MAX / 2) return 0;
    size_t wanted = keys * 2;
    size_t capacity = 2;
    while (capacity < wanted) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity <<= 1;
    }
    return capacity;
}

static int v4_adaptive_key_index(const State *inner, ColiExpertKey key,
                                 size_t *index) {
    if (!inner || !index || key.layer < 0 || key.layer >= inner->layers ||
        key.expert < 0 || key.expert >= inner->experts)
        return 0;
    *index = (size_t)key.layer * (size_t)inner->experts +
             (size_t)key.expert;
    return 1;
}

static int v4_adaptive_persistent_resident(State *inner, ColiExpertKey key) {
    Slot *slots = persistent_for(inner, key.layer);
    for (int i = 0; slots && i < inner->persistent_slots_per_layer; i++) {
        Slot *slot = &slots[i];
        if (slot->state == SLOT_RESIDENT && same_key(slot, key)) return 1;
    }
    return 0;
}

/* Encode the shared comparator's ordering in the base store's scalar policy
 * field. Layout, most significant first:
 *   score | resident | recency tie | lower expert id | 8-bit increment guard.
 * The score is capped only for projection; the tracker itself remains full
 * 64-bit/saturating. A policy score difference of one therefore dominates up
 * to 255 legacy physical-request increments before the next route observation.
 */
static uint64_t v4_adaptive_project_rank(
    const ColiExpertResidencyCandidate *candidate, uint64_t current_epoch) {
    if (!candidate || !candidate->score) return 0;
    const unsigned shift = 26;
    uint64_t score_cap = UINT64_MAX >> shift;
    uint64_t score = candidate->score > score_cap
        ? score_cap : candidate->score;
    uint64_t age = current_epoch > candidate->last_epoch
        ? current_epoch - candidate->last_epoch : 0;
    uint64_t recency = age >= 255 ? 0 : 255 - age;
    uint64_t expert = candidate->key.expert < 0 ? 511u
        : (uint64_t)candidate->key.expert;
    if (expert > 511u) expert = 511u;
    uint64_t expert_preference = 511u - expert;
    return (score << shift) |
           ((uint64_t)(candidate->currently_resident != 0) << 25) |
           (recency << 17) |
           (expert_preference << 8);
}

static void v4_adaptive_project_layer_locked(
    ColiV4AdaptiveExpertStoreState *state, State *inner, int layer) {
    if (!state || !inner || layer < 0 || layer >= inner->layers ||
        !inner->usage || !inner->slot_bytes)
        return;

    uint64_t quanta = inner->slot_bytes / UINT64_C(4096);
    if (inner->slot_bytes % UINT64_C(4096)) quanta++;
    if (!quanta) quanta = 1;

    for (int expert = 0; expert < inner->experts; expert++) {
        ColiExpertKey key = {layer, expert};
        const ColiExpertActivationEntry *entry =
            coli_expert_activation_find_const(&state->tracker, key);
        uint64_t rank = 0;
        if (entry) {
            ColiExpertResidencyCandidate candidate =
                coli_expert_residency_policy_candidate(
                    entry, state->current_epoch, inner->slot_bytes, quanta,
                    v4_adaptive_persistent_resident(inner, key),
                    &state->policy);
            rank = v4_adaptive_project_rank(&candidate, state->current_epoch);
        }
        inner->usage[(size_t)layer * (size_t)inner->experts +
                     (size_t)expert] = rank;
    }
}

static int v4_adaptive_lookup(ColiExpertStore *store, ColiExpertKey key,
                              ColiExpertView *view) {
    ColiV4AdaptiveExpertStoreState *state = store ? store->state : NULL;
    State *inner = state && state->inner ? state->inner->state : NULL;
    if (!state || !inner) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }

    size_t index = 0;
    if (v4_adaptive_key_index(inner, key, &index)) {
        pthread_mutex_lock(&inner->mutex);
        if (state->physical_usage[index] != UINT64_MAX)
            state->physical_usage[index]++;
        pthread_mutex_unlock(&inner->mutex);
    }
    return coli_expert_lookup(state->inner, key, view);
}

static int v4_adaptive_lookup_context(
    ColiExpertStore *store, ColiExpertKey key,
    const ColiExpertRequestContext *context, ColiExpertView *view) {
    (void)context;
    return v4_adaptive_lookup(store, key, view);
}

static void v4_adaptive_release(ColiExpertStore *store, ColiExpertView *view) {
    ColiV4AdaptiveExpertStoreState *state = store ? store->state : NULL;
    if (state && state->inner) coli_expert_release(state->inner, view);
}

static int v4_adaptive_prefetch(ColiExpertStore *store,
                                const ColiExpertKey *keys, size_t count) {
    ColiV4AdaptiveExpertStoreState *state = store ? store->state : NULL;
    if (!state || !state->inner || !state->inner->ops ||
        !state->inner->ops->prefetch)
        return 0;
    return state->inner->ops->prefetch(state->inner, keys, count);
}

static void v4_adaptive_stats(const ColiExpertStore *store,
                              ColiExpertStoreStats *stats) {
    ColiV4AdaptiveExpertStoreState *state = store ? store->state : NULL;
    State *inner = state && state->inner ? state->inner->state : NULL;
    if (!state || !inner || !stats) return;
    if (state->inner->ops && state->inner->ops->stats)
        state->inner->ops->stats(state->inner, stats);
    pthread_mutex_lock(&inner->mutex);
    stats->logical_activations = state->tracker.total_logical_activations;
    stats->activation_observations = state->tracker.total_observations;
    stats->activation_keys = state->tracker.used;
    stats->activation_dropped_new_keys = state->tracker.dropped_new_keys;
    pthread_mutex_unlock(&inner->mutex);
}

static void v4_adaptive_observe(
    ColiExpertStore *store,
    const struct ColiExpertActivationSample *raw_samples,
    size_t count) {
    ColiV4AdaptiveExpertStoreState *state = store ? store->state : NULL;
    State *inner = state && state->inner ? state->inner->state : NULL;
    const ColiExpertActivationSample *samples =
        (const ColiExpertActivationSample *)raw_samples;
    if (!state || !inner || !samples || !count) return;

    pthread_mutex_lock(&inner->mutex);
    (void)coli_expert_activation_observe_many(&state->tracker, samples, count);
    for (size_t i = 0; i < count; i++)
        if (samples[i].epoch > state->current_epoch)
            state->current_epoch = samples[i].epoch;

    /* One route batch can contain UNKNOWN/PREFILL/DECODE samples for the same
     * layer. Reproject each distinct layer once after the complete batch has
     * entered the tracker. Counts are small (<= routed experts * phases), so a
     * dependency-free O(n^2) distinct check is cheaper than extra hot-path
     * allocation and deterministic across platforms. */
    for (size_t i = 0; i < count; i++) {
        int layer = samples[i].key.layer;
        int seen = 0;
        for (size_t j = 0; j < i; j++)
            if (samples[j].key.layer == layer) {
                seen = 1;
                break;
            }
        if (!seen) v4_adaptive_project_layer_locked(state, inner, layer);
    }
    pthread_mutex_unlock(&inner->mutex);
}

static void v4_adaptive_destroy(ColiExpertStore *store) {
    ColiV4AdaptiveExpertStoreState *state = store ? store->state : NULL;
    if (state) {
        State *inner = state->inner ? state->inner->state : NULL;
        if (inner && inner->usage && state->physical_usage) {
            pthread_mutex_lock(&inner->mutex);
            memcpy(inner->usage, state->physical_usage,
                   state->key_count * sizeof(*state->physical_usage));
            pthread_mutex_unlock(&inner->mutex);
        }
        if (state->inner && state->inner->ops && state->inner->ops->destroy)
            state->inner->ops->destroy(state->inner);
        free(state->physical_usage);
        free(state->entries);
        free(state);
    }
    free(store);
}

static const ColiExpertStoreOps *v4_adaptive_ops(void) {
    static const ColiExpertStoreOps ops = {
        .lookup = v4_adaptive_lookup,
        .release = v4_adaptive_release,
        .prefetch = v4_adaptive_prefetch,
        .stats = v4_adaptive_stats,
        .destroy = v4_adaptive_destroy,
        .lookup_context = v4_adaptive_lookup_context,
        .observe_activations = v4_adaptive_observe,
    };
    return &ops;
}

int coli_v4_coli_expert_store_open(const ColiV4ColiExpertStoreOptions *options,
                                   ColiExpertStore **output,
                                   char *error, size_t error_size) {
    ColiExpertStore *inner = NULL;
    int result = coli_v4_coli_expert_store_open_base(
        options, &inner, error, error_size);
    if (result || !inner || !output) return result;

    State *base = inner->state;
    if (!base || base->layers < 1 || base->experts < 1 ||
        (size_t)base->layers > SIZE_MAX / (size_t)base->experts) {
        *output = inner;
        return 0;
    }
    size_t key_count = (size_t)base->layers * (size_t)base->experts;
    size_t capacity = v4_adaptive_tracker_capacity(key_count);
    if (!capacity || capacity > SIZE_MAX / sizeof(ColiExpertActivationEntry) ||
        key_count > SIZE_MAX / sizeof(uint64_t)) {
        *output = inner;
        return 0;
    }

    ColiExpertStore *outer = calloc(1, sizeof(*outer));
    ColiV4AdaptiveExpertStoreState *state = calloc(1, sizeof(*state));
    ColiExpertActivationEntry *entries = calloc(capacity, sizeof(*entries));
    uint64_t *physical_usage = calloc(key_count, sizeof(*physical_usage));
    if (!outer || !state || !entries || !physical_usage) {
        free(physical_usage);
        free(entries);
        free(state);
        free(outer);
        *output = inner;
        return 0;
    }
    if (coli_expert_activation_init(&state->tracker, entries, capacity) != 0) {
        free(physical_usage);
        free(entries);
        free(state);
        free(outer);
        *output = inner;
        return 0;
    }

    state->inner = inner;
    state->entries = entries;
    state->physical_usage = physical_usage;
    state->key_count = key_count;
    state->policy = coli_expert_residency_policy_default();

    /* The shared policy already gives currently-resident candidates percentage
     * hysteresis. Disable the base store's old absolute request-count margin so
     * it does not double-apply a V4-specific threshold. Byte geometry and the
     * explicit legacy layout remain unchanged. */
    pthread_mutex_lock(&base->mutex);
    if (!base->legacy_layout) base->hot_hysteresis = 0;
    pthread_mutex_unlock(&base->mutex);

    outer->ops = v4_adaptive_ops();
    outer->state = state;
    *output = outer;
    fprintf(stderr,
            "v4_residency adaptive_policy=frequency-decay "
            "prefill_weight=%u decode_weight=%u resident_hysteresis=%u%% "
            "recency_quantum=%llu persistent_slots_per_layer=%d\n",
            state->policy.prefill_weight, state->policy.decode_weight,
            state->policy.resident_hysteresis_percent,
            (unsigned long long)state->policy.recency_quantum_epochs,
            base->persistent_slots_per_layer);
    return 0;
}
