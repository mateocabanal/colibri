/*
 * Adaptive-policy overlay for the current V4 COLI expert store.
 *
 * The physical store remains the owner of slots, loader concurrency, lease
 * generations, Metal registration and byte accounting. This translation unit
 * adds the shared logical activation tracker/scorer at its policy seam and
 * lightweight miss-cost telemetry at the blocking operations already present in
 * the store. The base state machine itself remains unchanged while #95 becomes
 * the common physical residency owner.
 */
#include "expert_activation.h"
#include "expert_residency_policy.h"
#include "coli_executor.h"

#include <pthread.h>
#include <stdint.h>
#include <time.h>

typedef struct ColiV4AdaptiveExpertStoreState ColiV4AdaptiveExpertStoreState;

typedef struct {
    ColiV4AdaptiveExpertStoreState *state;
    uint64_t wait_started_ns;
    uint64_t physical_load_samples;
    uint64_t physical_load_ns;
} ColiV4AdaptiveLookupTiming;

static _Thread_local ColiV4AdaptiveLookupTiming g_v4_adaptive_lookup_timing;

/* C11 wall elapsed fallback for this migration overlay. Timing is advisory and
 * is discarded if the clock moves backwards. Once V4 uses #95 record-I/O, the
 * shared backend/request timestamps become authoritative without changing the
 * ExpertStore stats contract. Immediate resident hits never call this helper. */
static uint64_t v4_adaptive_now_ns(void) {
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) != TIME_UTC || ts.tv_sec < 0 ||
        (uint64_t)ts.tv_sec > UINT64_MAX / UINT64_C(1000000000))
        return 0;
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) +
           (uint64_t)ts.tv_nsec;
}

static uint64_t v4_adaptive_sat_add(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static void v4_adaptive_mark_exposed_wait(void) {
    ColiV4AdaptiveLookupTiming *timing = &g_v4_adaptive_lookup_timing;
    if (!timing->state || timing->wait_started_ns) return;
    timing->wait_started_ns = v4_adaptive_now_ns();
}

/* Only the two operations that can expose expert-miss latency are intercepted:
 * waiting for another generation/slot and owning the physical record load.
 * This avoids a timing syscall on immediate resident hits. */
static int v4_adaptive_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex) {
    v4_adaptive_mark_exposed_wait();
    return pthread_cond_wait(cond, mutex);
}

static int v4_adaptive_executor_load_expert(
    const ColiExecutor *executor, int32_t layer, int32_t expert,
    void *resident_slot, size_t resident_bytes,
    char *error, size_t error_size) {
    ColiV4AdaptiveLookupTiming *timing = &g_v4_adaptive_lookup_timing;
    v4_adaptive_mark_exposed_wait();
    uint64_t began = v4_adaptive_now_ns();
    int result = coli_executor_load_expert(
        executor, layer, expert, resident_slot, resident_bytes,
        error, error_size);
    uint64_t ended = v4_adaptive_now_ns();
    if (!result && timing->state && began && ended > began) {
        timing->physical_load_samples = v4_adaptive_sat_add(
            timing->physical_load_samples, UINT64_C(1));
        timing->physical_load_ns = v4_adaptive_sat_add(
            timing->physical_load_ns, ended - began);
    }
    return result;
}

#define pthread_cond_wait v4_adaptive_cond_wait
#define coli_executor_load_expert v4_adaptive_executor_load_expert
#define coli_v4_coli_expert_store_open coli_v4_coli_expert_store_open_base
#include "coli_v4_expert_store.c"
#undef coli_v4_coli_expert_store_open
#undef coli_executor_load_expert
#undef pthread_cond_wait

/* This adapter intentionally comes after the base implementation include: it
 * consumes the base State/Slot helpers without duplicating physical ownership.
 * The planner sees only currently borrowed dense bytes as mandatory; inactive
 * dense entries are reclaimable optional residency. Rename the base replan
 * helper so this adapter can add a short token-epoch warmup cadence without
 * changing the shared long-lived 64-epoch decay cadence. */
#define coli_v4_dense_cache_stats coli_v4_dense_cache_planner_stats
#define coli_v4_adaptive_resource_replan_locked \
    coli_v4_adaptive_resource_replan_locked_base
#include "coli_v4_adaptive_resource_planner.h"
#undef coli_v4_adaptive_resource_replan_locked
#undef coli_v4_dense_cache_stats

static int coli_v4_adaptive_resource_replan_locked(
    ColiV4AdaptiveResourcePlanner *planner, State *inner,
    const ColiExpertActivationTracker *tracker,
    uint64_t current_epoch,
    const ColiExpertResidencyPolicyConfig *policy) {
    if (planner && planner->enabled && tracker && policy &&
        planner->replan_count &&
        planner->replan_count <= policy->planner_confidence_mass &&
        current_epoch > planner->last_replan_epoch &&
        current_epoch - planner->last_replan_epoch <
            policy->recency_quantum_epochs) {
        /* The first global plan is intentionally allowed as soon as route mass
         * reaches confidence. After that, replan once per new logical token for
         * one small confidence window so decode-hot experts can emerge inside a
         * short request. The base helper still owns the actual allocation and
         * returns to its 64-epoch cadence after this bounded warmup. */
        uint64_t saved_count = planner->replan_count;
        planner->replan_count = 0;
        int result = coli_v4_adaptive_resource_replan_locked_base(
            planner, inner, tracker, current_epoch, policy);
        if (result > 0)
            planner->replan_count = saved_count + 1;
        else
            planner->replan_count = saved_count;
        return result;
    }
    return coli_v4_adaptive_resource_replan_locked_base(
        planner, inner, tracker, current_epoch, policy);
}

struct ColiV4AdaptiveExpertStoreState {
    ColiExpertStore *inner;
    ColiExpertActivationTracker tracker;
    ColiExpertActivationEntry *entries;
    uint64_t *physical_usage;
    size_t key_count;
    uint64_t current_epoch;
    ColiExpertResidencyPolicyConfig policy;
    ColiV4AdaptiveResourcePlanner resource_planner;
};

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
        size_t key_index = (size_t)layer * (size_t)inner->experts +
                           (size_t)expert;
        const ColiExpertActivationEntry *entry =
            coli_expert_activation_find_const(&state->tracker, key);
        uint64_t rank = 0;
        if (entry && coli_v4_adaptive_resource_selected(
                &state->resource_planner, key_index)) {
            ColiExpertResidencyCandidate candidate =
                coli_expert_residency_policy_candidate(
                    entry, state->current_epoch, inner->slot_bytes, quanta,
                    v4_adaptive_persistent_resident(inner, key),
                    &state->policy);
            rank = v4_adaptive_project_rank(&candidate, state->current_epoch);
        }
        inner->usage[key_index] = rank;
    }
}

static void v4_adaptive_commit_lookup_timing(
    State *inner, const ColiV4AdaptiveLookupTiming *timing,
    int lookup_succeeded) {
    if (!inner || !timing) return;

    uint64_t ended = 0;
    uint64_t exposed_ns = 0;
    if (lookup_succeeded && timing->wait_started_ns) {
        ended = v4_adaptive_now_ns();
        if (ended > timing->wait_started_ns)
            exposed_ns = ended - timing->wait_started_ns;
    }

    if (!timing->physical_load_samples && !exposed_ns) return;
    pthread_mutex_lock(&inner->mutex);
    inner->stats.physical_load_samples = v4_adaptive_sat_add(
        inner->stats.physical_load_samples, timing->physical_load_samples);
    inner->stats.physical_load_ns = v4_adaptive_sat_add(
        inner->stats.physical_load_ns, timing->physical_load_ns);
    if (exposed_ns) {
        inner->stats.exposed_wait_samples = v4_adaptive_sat_add(
            inner->stats.exposed_wait_samples, UINT64_C(1));
        inner->stats.exposed_wait_ns = v4_adaptive_sat_add(
            inner->stats.exposed_wait_ns, exposed_ns);
    }
    pthread_mutex_unlock(&inner->mutex);
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

    ColiV4AdaptiveLookupTiming previous = g_v4_adaptive_lookup_timing;
    g_v4_adaptive_lookup_timing = (ColiV4AdaptiveLookupTiming){
        .state = state,
    };
    int result = coli_expert_lookup(state->inner, key, view);
    ColiV4AdaptiveLookupTiming completed = g_v4_adaptive_lookup_timing;
    g_v4_adaptive_lookup_timing = previous;
    v4_adaptive_commit_lookup_timing(inner, &completed, result == 0);
    return result;
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
    coli_expert_activation_observe_many(&state->tracker, samples, count);
    for (size_t i = 0; i < count; i++)
        if (samples[i].epoch > state->current_epoch)
            state->current_epoch = samples[i].epoch;

    int replanned = coli_v4_adaptive_resource_replan_locked(
        &state->resource_planner, inner, &state->tracker,
        state->current_epoch, &state->policy);
    if (replanned > 0) {
        /* A global allocation can change any layer even when this route batch
         * touched only one of them, so refresh the complete local admission map. */
        for (int layer = 0; layer < inner->layers; layer++)
            v4_adaptive_project_layer_locked(state, inner, layer);
    } else {
        /* One route batch can contain UNKNOWN/PREFILL/DECODE samples for the same
         * layer. Reproject each distinct layer once after the complete batch has
         * entered the tracker. Counts are small, so a dependency-free O(n^2)
         * distinct check is cheaper than extra hot-path allocation. */
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
        if (state->resource_planner.enabled) {
            fprintf(stderr,
                    "v4_resource_planner status=done replans=%llu "
                    "last_epoch=%llu dense_budget=%.2fGiB "
                    "expert_budget=%.2fGiB\n",
                    (unsigned long long)state->resource_planner.replan_count,
                    (unsigned long long)state->resource_planner.last_replan_epoch,
                    state->resource_planner.selected_dense_bytes /
                        (1024.0 * 1024.0 * 1024.0),
                    state->resource_planner.selected_expert_bytes /
                        (1024.0 * 1024.0 * 1024.0));
        }
        coli_v4_adaptive_resource_destroy(&state->resource_planner);
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

    int planner_result = coli_v4_adaptive_resource_init(
        &state->resource_planner, base);
    if (planner_result < 0) {
        fprintf(stderr,
                "v4_resource_planner status=disabled reason=initialization "
                "fallback=base-residency\n");
    }

    /* Shared resident hysteresis already lives in the projected high bits. In
     * planner-managed mode reserve the low 8 bits as a hard admission guard:
     * the base store increments usage before choosing a slot, so rank-0
     * unselected experts must not beat a zero-value sentinel after one request. */
    pthread_mutex_lock(&base->mutex);
    if (!base->legacy_layout)
        base->hot_hysteresis = state->resource_planner.enabled
            ? UINT64_C(255) : 0;
    pthread_mutex_unlock(&base->mutex);

    outer->ops = v4_adaptive_ops();
    outer->state = state;
    *output = outer;
    fprintf(stderr,
            "v4_residency adaptive_policy=frequency-decay "
            "prefill_weight=%u decode_weight=%u resident_hysteresis=%u%% "
            "recency_quantum=%llu planner_horizon=%llu confidence_mass=%llu "
            "persistent_budget=%s dense_budget=%.2fGiB\n",
            state->policy.prefill_weight, state->policy.decode_weight,
            state->policy.resident_hysteresis_percent,
            (unsigned long long)state->policy.recency_quantum_epochs,
            (unsigned long long)state->policy.planning_horizon_epochs,
            (unsigned long long)state->policy.planner_confidence_mass,
            state->resource_planner.enabled ? "global-planner" : "base-layout",
            base->dense_cache_budget_bytes / (1024.0 * 1024.0 * 1024.0));
    return 0;
}
