#ifndef COLIBRI_EXPERT_RESIDENCY_H
#define COLIBRI_EXPERT_RESIDENCY_H

#include "expert_store.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef enum {
    COLI_EXPERT_RESIDENCY_COLD = 0,
    COLI_EXPERT_RESIDENCY_RESERVED = 1,
    COLI_EXPERT_RESIDENCY_LOADING = 2,
    COLI_EXPERT_RESIDENCY_PREPARING = 3,
    COLI_EXPERT_RESIDENCY_RESIDENT = 4,
    COLI_EXPERT_RESIDENCY_EVICTING = 5,
} ColiExpertResidencyState;

typedef enum {
    COLI_EXPERT_TIER_NONE = 0,
    COLI_EXPERT_TIER_HOST = 1u << 0,
    COLI_EXPERT_TIER_UMA = 1u << 1,
    COLI_EXPERT_TIER_PINNED_HOST = 1u << 2,
    COLI_EXPERT_TIER_DEVICE = 1u << 3,
} ColiExpertResidencyTier;

typedef enum {
    COLI_EXPERT_REQUEST_INVALID = -1,
    COLI_EXPERT_REQUEST_NO_BUDGET = 0,
    COLI_EXPERT_REQUEST_LOAD_OWNER = 1,
    COLI_EXPERT_REQUEST_JOIN_INFLIGHT = 2,
    COLI_EXPERT_REQUEST_HIT = 3,
} ColiExpertRequestResult;

typedef struct {
    uint64_t capacity_bytes;
    /* committed is the hard invariant. Reservation->resident publication only
     * changes classification; it never opens a transient hole another loader
     * can steal. */
    atomic_uint_fast64_t committed_bytes;
    atomic_uint_fast64_t reserved_bytes;
    atomic_uint_fast64_t resident_bytes;
    atomic_uint_fast64_t peak_committed_bytes;
} ColiExpertResidencyBudget;

typedef struct {
    ColiExpertKey key;
    atomic_int state;
    atomic_uint_fast64_t generation;
    atomic_uint refs;
    atomic_uint tier_mask;
    uint64_t allocation_bytes;
} ColiExpertResidencyEntry;

typedef struct {
    ColiExpertResidencyEntry *entry;
    ColiExpertKey key;
    uint64_t generation;
    unsigned tier_mask;
} ColiExpertResidencyLease;

static inline int coli_expert_key_equal(ColiExpertKey a, ColiExpertKey b) {
    return a.layer == b.layer && a.expert == b.expert;
}

static inline void coli_expert_residency_budget_init(
    ColiExpertResidencyBudget *budget, uint64_t capacity_bytes) {
    if (!budget) return;
    budget->capacity_bytes = capacity_bytes;
    atomic_init(&budget->committed_bytes, 0);
    atomic_init(&budget->reserved_bytes, 0);
    atomic_init(&budget->resident_bytes, 0);
    atomic_init(&budget->peak_committed_bytes, 0);
}

static inline uint64_t coli_expert_residency_budget_committed(
    const ColiExpertResidencyBudget *budget) {
    return budget ? atomic_load_explicit(
        &budget->committed_bytes, memory_order_acquire) : 0;
}

static inline void coli_expert_residency_update_peak(
    ColiExpertResidencyBudget *budget, uint64_t committed) {
    uint64_t peak = atomic_load_explicit(
        &budget->peak_committed_bytes, memory_order_acquire);
    while (committed > peak &&
           !atomic_compare_exchange_weak_explicit(
               &budget->peak_committed_bytes, &peak, committed,
               memory_order_acq_rel, memory_order_acquire)) {}
}

static inline int coli_expert_residency_budget_try_reserve(
    ColiExpertResidencyBudget *budget, uint64_t bytes) {
    if (!budget || !bytes || bytes > budget->capacity_bytes) return -1;
    uint64_t committed = atomic_load_explicit(
        &budget->committed_bytes, memory_order_acquire);
    for (;;) {
        if (committed > budget->capacity_bytes ||
            bytes > budget->capacity_bytes - committed)
            return 0;
        uint64_t desired = committed + bytes;
        if (atomic_compare_exchange_weak_explicit(
                &budget->committed_bytes, &committed, desired,
                memory_order_acq_rel, memory_order_acquire)) {
            atomic_fetch_add_explicit(&budget->reserved_bytes, bytes,
                                      memory_order_acq_rel);
            coli_expert_residency_update_peak(budget, desired);
            return 1;
        }
    }
}

static inline int coli_expert_residency_atomic_sub_checked(
    atomic_uint_fast64_t *value, uint64_t bytes) {
    uint64_t current = atomic_load_explicit(value, memory_order_acquire);
    for (;;) {
        if (current < bytes) return -1;
        if (atomic_compare_exchange_weak_explicit(
                value, &current, current - bytes,
                memory_order_acq_rel, memory_order_acquire))
            return 0;
    }
}

static inline int coli_expert_residency_budget_publish(
    ColiExpertResidencyBudget *budget, uint64_t bytes) {
    if (!budget || !bytes ||
        coli_expert_residency_atomic_sub_checked(
            &budget->reserved_bytes, bytes) != 0)
        return -1;
    atomic_fetch_add_explicit(&budget->resident_bytes, bytes,
                              memory_order_acq_rel);
    return 0;
}

static inline int coli_expert_residency_budget_cancel(
    ColiExpertResidencyBudget *budget, uint64_t bytes) {
    if (!budget || !bytes ||
        coli_expert_residency_atomic_sub_checked(
            &budget->reserved_bytes, bytes) != 0)
        return -1;
    return coli_expert_residency_atomic_sub_checked(
        &budget->committed_bytes, bytes);
}

static inline int coli_expert_residency_budget_evict(
    ColiExpertResidencyBudget *budget, uint64_t bytes) {
    if (!budget || !bytes ||
        coli_expert_residency_atomic_sub_checked(
            &budget->resident_bytes, bytes) != 0)
        return -1;
    return coli_expert_residency_atomic_sub_checked(
        &budget->committed_bytes, bytes);
}

static inline int coli_expert_residency_entry_init(
    ColiExpertResidencyEntry *entry, ColiExpertKey key) {
    if (!entry || key.layer < 0 || key.expert < 0) return -1;
    memset(entry, 0, sizeof(*entry));
    entry->key = key;
    atomic_init(&entry->state, COLI_EXPERT_RESIDENCY_COLD);
    atomic_init(&entry->generation, 0);
    atomic_init(&entry->refs, 0);
    atomic_init(&entry->tier_mask, COLI_EXPERT_TIER_NONE);
    return 0;
}

static inline ColiExpertResidencyState coli_expert_residency_state(
    const ColiExpertResidencyEntry *entry) {
    return entry ? (ColiExpertResidencyState)atomic_load_explicit(
        &entry->state, memory_order_acquire) : COLI_EXPERT_RESIDENCY_COLD;
}

static inline int coli_expert_residency_acquire(
    ColiExpertResidencyEntry *entry, ColiExpertResidencyLease *lease) {
    if (!entry || !lease) return -1;
    memset(lease, 0, sizeof(*lease));
    if (atomic_load_explicit(&entry->state, memory_order_acquire) !=
        COLI_EXPERT_RESIDENCY_RESIDENT)
        return 0;
    uint64_t generation = atomic_load_explicit(
        &entry->generation, memory_order_acquire);
    unsigned tier = atomic_load_explicit(&entry->tier_mask, memory_order_acquire);
    if (!generation || !tier) return -1;

    atomic_fetch_add_explicit(&entry->refs, 1, memory_order_acq_rel);
    if (atomic_load_explicit(&entry->state, memory_order_acquire) !=
            COLI_EXPERT_RESIDENCY_RESIDENT ||
        atomic_load_explicit(&entry->generation, memory_order_acquire) != generation) {
        atomic_fetch_sub_explicit(&entry->refs, 1, memory_order_acq_rel);
        return 0;
    }

    lease->entry = entry;
    lease->key = entry->key;
    lease->generation = generation;
    lease->tier_mask = tier;
    return 1;
}

/* Exactly one contender can claim COLD as LOAD_OWNER. Everyone observing the
 * intermediate states joins the same physical load/prepare operation. */
static inline ColiExpertRequestResult coli_expert_residency_request(
    ColiExpertResidencyEntry *entry,
    ColiExpertResidencyBudget *budget,
    uint64_t allocation_bytes,
    ColiExpertResidencyLease *lease) {
    if (!entry || !budget || !allocation_bytes || !lease)
        return COLI_EXPERT_REQUEST_INVALID;

    for (;;) {
        int state = atomic_load_explicit(&entry->state, memory_order_acquire);
        if (state == COLI_EXPERT_RESIDENCY_RESIDENT) {
            int hit = coli_expert_residency_acquire(entry, lease);
            if (hit > 0) return COLI_EXPERT_REQUEST_HIT;
            if (hit < 0) return COLI_EXPERT_REQUEST_INVALID;
            continue;
        }
        if (state == COLI_EXPERT_RESIDENCY_RESERVED ||
            state == COLI_EXPERT_RESIDENCY_LOADING ||
            state == COLI_EXPERT_RESIDENCY_PREPARING ||
            state == COLI_EXPERT_RESIDENCY_EVICTING)
            return COLI_EXPERT_REQUEST_JOIN_INFLIGHT;
        if (state != COLI_EXPERT_RESIDENCY_COLD)
            return COLI_EXPERT_REQUEST_INVALID;

        int expected = COLI_EXPERT_RESIDENCY_COLD;
        if (!atomic_compare_exchange_weak_explicit(
                &entry->state, &expected, COLI_EXPERT_RESIDENCY_RESERVED,
                memory_order_acq_rel, memory_order_acquire))
            continue;

        int reserved = coli_expert_residency_budget_try_reserve(
            budget, allocation_bytes);
        if (reserved <= 0) {
            atomic_store_explicit(&entry->state, COLI_EXPERT_RESIDENCY_COLD,
                                  memory_order_release);
            return reserved < 0 ? COLI_EXPERT_REQUEST_INVALID
                                : COLI_EXPERT_REQUEST_NO_BUDGET;
        }
        entry->allocation_bytes = allocation_bytes;
        memset(lease, 0, sizeof(*lease));
        return COLI_EXPERT_REQUEST_LOAD_OWNER;
    }
}

static inline int coli_expert_residency_mark_loading(
    ColiExpertResidencyEntry *entry) {
    if (!entry) return -1;
    int expected = COLI_EXPERT_RESIDENCY_RESERVED;
    return atomic_compare_exchange_strong_explicit(
        &entry->state, &expected, COLI_EXPERT_RESIDENCY_LOADING,
        memory_order_acq_rel, memory_order_acquire) ? 0 : -1;
}

static inline int coli_expert_residency_mark_preparing(
    ColiExpertResidencyEntry *entry) {
    if (!entry) return -1;
    int expected = COLI_EXPERT_RESIDENCY_LOADING;
    if (atomic_compare_exchange_strong_explicit(
            &entry->state, &expected, COLI_EXPERT_RESIDENCY_PREPARING,
            memory_order_acq_rel, memory_order_acquire))
        return 0;
    expected = COLI_EXPERT_RESIDENCY_RESERVED;
    return atomic_compare_exchange_strong_explicit(
        &entry->state, &expected, COLI_EXPERT_RESIDENCY_PREPARING,
        memory_order_acq_rel, memory_order_acquire) ? 0 : -1;
}

static inline int coli_expert_residency_publish(
    ColiExpertResidencyEntry *entry,
    ColiExpertResidencyBudget *budget,
    uint64_t generation,
    unsigned tier_mask) {
    if (!entry || !budget || !generation || !tier_mask || !entry->allocation_bytes)
        return -1;
    int state = atomic_load_explicit(&entry->state, memory_order_acquire);
    if (state != COLI_EXPERT_RESIDENCY_RESERVED &&
        state != COLI_EXPERT_RESIDENCY_LOADING &&
        state != COLI_EXPERT_RESIDENCY_PREPARING)
        return -1;
    if (coli_expert_residency_budget_publish(
            budget, entry->allocation_bytes) != 0)
        return -1;
    atomic_store_explicit(&entry->generation, generation, memory_order_release);
    atomic_store_explicit(&entry->tier_mask, tier_mask, memory_order_release);
    atomic_store_explicit(&entry->state, COLI_EXPERT_RESIDENCY_RESIDENT,
                          memory_order_release);
    return 0;
}

static inline int coli_expert_residency_fail_load(
    ColiExpertResidencyEntry *entry, ColiExpertResidencyBudget *budget) {
    if (!entry || !budget || !entry->allocation_bytes) return -1;
    int state = atomic_load_explicit(&entry->state, memory_order_acquire);
    if (state != COLI_EXPERT_RESIDENCY_RESERVED &&
        state != COLI_EXPERT_RESIDENCY_LOADING &&
        state != COLI_EXPERT_RESIDENCY_PREPARING)
        return -1;
    if (coli_expert_residency_budget_cancel(
            budget, entry->allocation_bytes) != 0)
        return -1;
    entry->allocation_bytes = 0;
    atomic_store_explicit(&entry->tier_mask, COLI_EXPERT_TIER_NONE,
                          memory_order_release);
    atomic_store_explicit(&entry->state, COLI_EXPERT_RESIDENCY_COLD,
                          memory_order_release);
    return 0;
}

static inline int coli_expert_residency_release(
    ColiExpertResidencyLease *lease) {
    if (!lease || !lease->entry || !lease->generation) return -1;
    ColiExpertResidencyEntry *entry = lease->entry;
    if (!coli_expert_key_equal(entry->key, lease->key) ||
        atomic_load_explicit(&entry->generation, memory_order_acquire) !=
            lease->generation)
        return -1;
    unsigned refs = atomic_load_explicit(&entry->refs, memory_order_acquire);
    for (;;) {
        if (!refs) return -1;
        if (atomic_compare_exchange_weak_explicit(
                &entry->refs, &refs, refs - 1,
                memory_order_acq_rel, memory_order_acquire)) {
            memset(lease, 0, sizeof(*lease));
            return 0;
        }
    }
}

static inline int coli_expert_residency_begin_evict(
    ColiExpertResidencyEntry *entry) {
    if (!entry || atomic_load_explicit(&entry->refs, memory_order_acquire) != 0)
        return 0;
    int expected = COLI_EXPERT_RESIDENCY_RESIDENT;
    if (!atomic_compare_exchange_strong_explicit(
            &entry->state, &expected, COLI_EXPERT_RESIDENCY_EVICTING,
            memory_order_acq_rel, memory_order_acquire))
        return 0;
    if (atomic_load_explicit(&entry->refs, memory_order_acquire) != 0) {
        atomic_store_explicit(&entry->state, COLI_EXPERT_RESIDENCY_RESIDENT,
                              memory_order_release);
        return 0;
    }
    return 1;
}

static inline int coli_expert_residency_finish_evict(
    ColiExpertResidencyEntry *entry, ColiExpertResidencyBudget *budget) {
    if (!entry || !budget ||
        atomic_load_explicit(&entry->state, memory_order_acquire) !=
            COLI_EXPERT_RESIDENCY_EVICTING ||
        atomic_load_explicit(&entry->refs, memory_order_acquire) != 0 ||
        !entry->allocation_bytes)
        return -1;
    if (coli_expert_residency_budget_evict(
            budget, entry->allocation_bytes) != 0)
        return -1;
    entry->allocation_bytes = 0;
    atomic_store_explicit(&entry->tier_mask, COLI_EXPERT_TIER_NONE,
                          memory_order_release);
    atomic_store_explicit(&entry->state, COLI_EXPERT_RESIDENCY_COLD,
                          memory_order_release);
    return 0;
}

#endif
