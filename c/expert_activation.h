#ifndef COLIBRI_EXPERT_ACTIVATION_H
#define COLIBRI_EXPERT_ACTIVATION_H

#include "expert_store.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Shared logical activation signal for adaptive MoE residency.
 *
 * Engines report routing multiplicity here BEFORE expert union/batching can
 * collapse several logical selections into one physical store lookup. This is
 * intentionally independent from residency state, replacement policy and I/O.
 * A later policy layer can consume these counters without teaching the runtime
 * model names or routing implementations.
 *
 * The tracker is caller-allocated, bounded, allocation-free, and uses an
 * open-addressed power-of-two table. A full table drops only previously unseen
 * keys; observations for already-known keys continue to accumulate.
 */

typedef struct {
    ColiExpertKey key;
    ColiExpertPhase phase;
    /* Number of logical route selections represented by this observation.
     * For ordinary decode this is commonly 1. A batched/union prefill path may
     * report a larger value for one (layer, expert) before physical dedupe. */
    uint64_t multiplicity;
    /* Monotonic policy epoch chosen by the runtime/engine adapter. This may be
     * a token position, routing step, or request-global serial; generic policy
     * only relies on ordering, not model-specific meaning. */
    uint64_t epoch;
} ColiExpertActivationSample;

typedef struct {
    ColiExpertKey key;
    uint64_t logical_activations;
    uint64_t prefill_activations;
    uint64_t decode_activations;
    uint64_t observations;
    uint64_t last_epoch;
    uint64_t hash_tag; /* zero means empty */
} ColiExpertActivationEntry;

typedef struct {
    ColiExpertActivationEntry *entries;
    size_t capacity;
    size_t used;
    uint64_t total_logical_activations;
    uint64_t total_observations;
    uint64_t dropped_new_keys;
} ColiExpertActivationTracker;

static inline uint64_t coli_expert_activation_sat_add(uint64_t a, uint64_t b) {
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static inline int coli_expert_activation_capacity_valid(size_t capacity) {
    return capacity >= 2 && (capacity & (capacity - 1)) == 0;
}

static inline uint64_t coli_expert_activation_hash(ColiExpertKey key) {
    /* splitmix64 finalizer over the stable logical identity. */
    uint64_t x = ((uint64_t)(uint32_t)key.layer << 32) |
                 (uint64_t)(uint32_t)key.expert;
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    /* Reserve zero as the empty-table marker. */
    return x ? x : UINT64_C(1);
}

static inline int coli_expert_activation_key_equal(ColiExpertKey a,
                                                    ColiExpertKey b) {
    return a.layer == b.layer && a.expert == b.expert;
}

static inline int coli_expert_activation_init(
    ColiExpertActivationTracker *tracker,
    ColiExpertActivationEntry *storage,
    size_t capacity) {
    if (!tracker || !storage || !coli_expert_activation_capacity_valid(capacity))
        return -1;
    tracker->entries = storage;
    tracker->capacity = capacity;
    tracker->used = 0;
    tracker->total_logical_activations = 0;
    tracker->total_observations = 0;
    tracker->dropped_new_keys = 0;
    for (size_t i = 0; i < capacity; i++) {
        storage[i].key.layer = -1;
        storage[i].key.expert = -1;
        storage[i].logical_activations = 0;
        storage[i].prefill_activations = 0;
        storage[i].decode_activations = 0;
        storage[i].observations = 0;
        storage[i].last_epoch = 0;
        storage[i].hash_tag = 0;
    }
    return 0;
}

static inline ColiExpertActivationEntry *coli_expert_activation_find(
    ColiExpertActivationTracker *tracker, ColiExpertKey key) {
    if (!tracker || !tracker->entries || key.layer < 0 || key.expert < 0 ||
        !coli_expert_activation_capacity_valid(tracker->capacity))
        return NULL;
    uint64_t hash = coli_expert_activation_hash(key);
    size_t mask = tracker->capacity - 1;
    size_t slot = (size_t)hash & mask;
    for (size_t probe = 0; probe < tracker->capacity; probe++) {
        ColiExpertActivationEntry *entry = &tracker->entries[slot];
        if (!entry->hash_tag) return NULL;
        if (entry->hash_tag == hash &&
            coli_expert_activation_key_equal(entry->key, key))
            return entry;
        slot = (slot + 1) & mask;
    }
    return NULL;
}

static inline const ColiExpertActivationEntry *coli_expert_activation_find_const(
    const ColiExpertActivationTracker *tracker, ColiExpertKey key) {
    return coli_expert_activation_find((ColiExpertActivationTracker *)tracker, key);
}

/* Returns 1 when a new key is inserted, 0 when an existing key is updated,
 * -1 for invalid input, and -2 when the bounded table cannot admit a new key. */
static inline int coli_expert_activation_observe(
    ColiExpertActivationTracker *tracker,
    ColiExpertActivationSample sample) {
    if (!tracker || !tracker->entries || sample.key.layer < 0 ||
        sample.key.expert < 0 || !sample.multiplicity ||
        (sample.phase != COLI_EXPERT_PHASE_UNKNOWN &&
         sample.phase != COLI_EXPERT_PHASE_PREFILL &&
         sample.phase != COLI_EXPERT_PHASE_DECODE) ||
        !coli_expert_activation_capacity_valid(tracker->capacity))
        return -1;

    uint64_t hash = coli_expert_activation_hash(sample.key);
    size_t mask = tracker->capacity - 1;
    size_t slot = (size_t)hash & mask;
    ColiExpertActivationEntry *entry = NULL;
    int inserted = 0;

    for (size_t probe = 0; probe < tracker->capacity; probe++) {
        ColiExpertActivationEntry *candidate = &tracker->entries[slot];
        if (!candidate->hash_tag) {
            entry = candidate;
            inserted = 1;
            break;
        }
        if (candidate->hash_tag == hash &&
            coli_expert_activation_key_equal(candidate->key, sample.key)) {
            entry = candidate;
            break;
        }
        slot = (slot + 1) & mask;
    }

    if (!entry) {
        tracker->dropped_new_keys = coli_expert_activation_sat_add(
            tracker->dropped_new_keys, UINT64_C(1));
        return -2;
    }

    if (inserted) {
        entry->key = sample.key;
        entry->logical_activations = 0;
        entry->prefill_activations = 0;
        entry->decode_activations = 0;
        entry->observations = 0;
        entry->last_epoch = 0;
        /* Publish the occupied marker last for simple debugger/read-only
         * inspection. Mutation is intentionally owned by one policy thread;
         * concurrency belongs above this tiny accounting primitive. */
        entry->hash_tag = hash;
        tracker->used++;
    }

    entry->logical_activations = coli_expert_activation_sat_add(
        entry->logical_activations, sample.multiplicity);
    if (sample.phase == COLI_EXPERT_PHASE_PREFILL)
        entry->prefill_activations = coli_expert_activation_sat_add(
            entry->prefill_activations, sample.multiplicity);
    else if (sample.phase == COLI_EXPERT_PHASE_DECODE)
        entry->decode_activations = coli_expert_activation_sat_add(
            entry->decode_activations, sample.multiplicity);
    entry->observations = coli_expert_activation_sat_add(
        entry->observations, UINT64_C(1));
    if (sample.epoch > entry->last_epoch) entry->last_epoch = sample.epoch;

    tracker->total_logical_activations = coli_expert_activation_sat_add(
        tracker->total_logical_activations, sample.multiplicity);
    tracker->total_observations = coli_expert_activation_sat_add(
        tracker->total_observations, UINT64_C(1));
    return inserted;
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_EXPERT_ACTIVATION_H */
