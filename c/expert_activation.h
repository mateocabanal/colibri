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
 *
 * Exact lifetime counters remain available for telemetry. Policy decisions use
 * a separate lazy-decayed activity mass with one shared half-life quantum so a
 * long-lived server cannot make experts hot forever merely because they were
 * popular early in the process lifetime.
 */

#define COLI_EXPERT_ACTIVITY_DECAY_QUANTUM_EPOCHS UINT64_C(64)

typedef struct ColiExpertActivationSample {
    ColiExpertKey key;
    ColiExpertPhase phase;
    /* Number of logical route selections represented by this observation.
     * For ordinary decode this is commonly 1. A batched/union prefill path may
     * report a larger value for one (layer, expert) before physical dedupe. */
    uint64_t multiplicity;
    /* Monotonic logical-token epoch chosen by the runtime/engine adapter.
     * One epoch unit is one logical token step: every routed layer evaluating
     * the same token/span must use the same epoch, independent of model depth.
     * A batched observation may use the end-of-span epoch and advance policy
     * time by the number of logical tokens represented by that span. Do not use
     * a per-layer routing-call serial here: decay and planner horizons are
     * expressed in these epoch units and therefore require a stable timebase. */
    uint64_t epoch;
} ColiExpertActivationSample;

typedef struct {
    ColiExpertKey key;

    /* Lifetime telemetry. These counters saturate rather than wrap. */
    uint64_t logical_activations;
    uint64_t prefill_activations;
    uint64_t decode_activations;
    uint64_t observations;
    uint64_t last_epoch;

    /* Bounded policy signal. Every complete decay quantum halves these values.
     * Unknown is explicit here because recent logical total cannot be recovered
     * from lifetime counters after independent lazy decay. */
    uint64_t recent_unknown_activations;
    uint64_t recent_prefill_activations;
    uint64_t recent_decode_activations;
    uint64_t recent_decay_epoch;

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

static inline uint64_t coli_expert_activation_decay_value(uint64_t value,
                                                          uint64_t buckets) {
    if (buckets >= 64) return 0;
    return value >> (unsigned)buckets;
}

/* Mutating lazy decay used immediately before adding a newer observation. */
static inline void coli_expert_activation_decay_recent(
    ColiExpertActivationEntry *entry, uint64_t epoch) {
    if (!entry || epoch <= entry->recent_decay_epoch) return;
    uint64_t delta = epoch - entry->recent_decay_epoch;
    uint64_t buckets = delta / COLI_EXPERT_ACTIVITY_DECAY_QUANTUM_EPOCHS;
    if (!buckets) return;
    entry->recent_unknown_activations = coli_expert_activation_decay_value(
        entry->recent_unknown_activations, buckets);
    entry->recent_prefill_activations = coli_expert_activation_decay_value(
        entry->recent_prefill_activations, buckets);
    entry->recent_decode_activations = coli_expert_activation_decay_value(
        entry->recent_decode_activations, buckets);
    /* buckets * quantum <= delta, so this cannot advance past epoch. */
    entry->recent_decay_epoch +=
        buckets * COLI_EXPERT_ACTIVITY_DECAY_QUANTUM_EPOCHS;
}

/* Read the recent phase masses as of current_epoch without mutating the entry.
 * This lets ranking remain const/read-only while preserving the same decay
 * semantics used on observation. */
static inline void coli_expert_activation_recent_at(
    const ColiExpertActivationEntry *entry, uint64_t current_epoch,
    uint64_t *unknown, uint64_t *prefill, uint64_t *decode) {
    uint64_t u = entry ? entry->recent_unknown_activations : 0;
    uint64_t p = entry ? entry->recent_prefill_activations : 0;
    uint64_t d = entry ? entry->recent_decode_activations : 0;
    if (entry && current_epoch > entry->recent_decay_epoch) {
        uint64_t buckets = (current_epoch - entry->recent_decay_epoch) /
            COLI_EXPERT_ACTIVITY_DECAY_QUANTUM_EPOCHS;
        u = coli_expert_activation_decay_value(u, buckets);
        p = coli_expert_activation_decay_value(p, buckets);
        d = coli_expert_activation_decay_value(d, buckets);
    }
    if (unknown) *unknown = u;
    if (prefill) *prefill = p;
    if (decode) *decode = d;
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
        storage[i].recent_unknown_activations = 0;
        storage[i].recent_prefill_activations = 0;
        storage[i].recent_decode_activations = 0;
        storage[i].recent_decay_epoch = 0;
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
        entry->recent_unknown_activations = 0;
        entry->recent_prefill_activations = 0;
        entry->recent_decode_activations = 0;
        entry->recent_decay_epoch = sample.epoch;
        /* Publish the occupied marker last for simple debugger/read-only
         * inspection. Mutation is intentionally owned by one policy thread;
         * concurrency belongs above this tiny accounting primitive. */
        entry->hash_tag = hash;
        tracker->used++;
    } else {
        coli_expert_activation_decay_recent(entry, sample.epoch);
    }

    entry->logical_activations = coli_expert_activation_sat_add(
        entry->logical_activations, sample.multiplicity);
    if (sample.phase == COLI_EXPERT_PHASE_PREFILL) {
        entry->prefill_activations = coli_expert_activation_sat_add(
            entry->prefill_activations, sample.multiplicity);
        entry->recent_prefill_activations = coli_expert_activation_sat_add(
            entry->recent_prefill_activations, sample.multiplicity);
    } else if (sample.phase == COLI_EXPERT_PHASE_DECODE) {
        entry->decode_activations = coli_expert_activation_sat_add(
            entry->decode_activations, sample.multiplicity);
        entry->recent_decode_activations = coli_expert_activation_sat_add(
            entry->recent_decode_activations, sample.multiplicity);
    } else {
        entry->recent_unknown_activations = coli_expert_activation_sat_add(
            entry->recent_unknown_activations, sample.multiplicity);
    }
    entry->observations = coli_expert_activation_sat_add(
        entry->observations, UINT64_C(1));
    if (sample.epoch > entry->last_epoch) entry->last_epoch = sample.epoch;

    tracker->total_logical_activations = coli_expert_activation_sat_add(
        tracker->total_logical_activations, sample.multiplicity);
    tracker->total_observations = coli_expert_activation_sat_add(
        tracker->total_observations, UINT64_C(1));
    return inserted;
}

static inline void coli_expert_activation_observe_many(
    ColiExpertActivationTracker *tracker,
    const ColiExpertActivationSample *samples,
    size_t count) {
    if (!tracker || !samples) return;
    for (size_t i = 0; i < count; i++)
        (void)coli_expert_activation_observe(tracker, samples[i]);
}

#ifdef __cplusplus
}
#endif

#endif /* COLIBRI_EXPERT_ACTIVATION_H */
