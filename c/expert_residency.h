#ifndef COLIBRI_EXPERT_RESIDENCY_H
#define COLIBRI_EXPERT_RESIDENCY_H

#include "expert_store.h"
#include "expert_representation.h"

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
    COLI_EXPERT_REQUEST_NO_SLOT = 4,
} ColiExpertRequestResult;

enum {
    COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY = 4,
};
#define COLI_EXPERT_VARIANT_NONE UINT32_MAX

typedef struct {
    uint64_t capacity_bytes;
    atomic_uint_fast64_t committed_bytes;
    atomic_uint_fast64_t reserved_bytes;
    atomic_uint_fast64_t resident_bytes;
    atomic_uint_fast64_t peak_committed_bytes;
} ColiExpertResidencyBudget;

typedef struct ColiExpertResidentVariant {
    atomic_int state;
    atomic_uint_fast64_t generation;
    atomic_uint_fast64_t pending_generation;
    atomic_uint refs;
    atomic_uint tier_mask;
    uint64_t allocation_bytes;
    ColiRepresentationId representation;
    uint64_t resident_bytes;
    void *physical;
    ColiExpertResidencyBudget *budget;
} ColiExpertResidentVariant;

typedef struct ColiExpertResidencyEntry {
    ColiExpertKey key;
    atomic_uint_fast64_t generation_allocator;
    atomic_int preferred_variant;
    atomic_int variant_lock;
    ColiExpertResidentVariant variants[COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY];
} ColiExpertResidencyEntry;

typedef struct {
    ColiExpertResidencyEntry *entry;
    ColiExpertResidentVariant *variant;
    uint32_t variant_id;
    ColiExpertKey key;
    uint64_t generation;
    unsigned tier_mask;
    ColiRepresentationId representation;
    uint64_t resident_bytes;
    uint64_t allocation_bytes;
    void *physical;
} ColiExpertResidencyLease;

typedef struct {
    ColiExpertKey key;
    ColiRepresentationId representation;
    uint64_t generation;
    unsigned tier_mask;
    uint64_t resident_bytes;
    uint64_t allocation_bytes;
    void *physical;
} ColiExpertResidentView;

typedef struct {
    uint32_t variant_id;
    ColiRepresentationId representation;
    ColiExpertResidencyState state;
    uint64_t generation;
    uint64_t pending_generation;
    unsigned refs;
    unsigned tier_mask;
    uint64_t resident_bytes;
    uint64_t allocation_bytes;
    int preferred;
} ColiExpertResidentVariantInfo;

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

static inline ColiExpertResidentVariant *coli_expert_residency_variant(
        ColiExpertResidencyEntry *entry, uint32_t variant_id) {
    if (!entry || variant_id >= COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY)
        return NULL;
    return &entry->variants[variant_id];
}

static inline const ColiExpertResidentVariant *coli_expert_residency_variant_const(
        const ColiExpertResidencyEntry *entry, uint32_t variant_id) {
    if (!entry || variant_id >= COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY)
        return NULL;
    return &entry->variants[variant_id];
}

static inline void coli_expert_residency_family_lock(
        ColiExpertResidencyEntry *entry) {
    int expected = 0;
    while (!atomic_compare_exchange_weak_explicit(
            &entry->variant_lock, &expected, 1,
            memory_order_acq_rel, memory_order_acquire))
        expected = 0;
}

static inline void coli_expert_residency_family_unlock(
        ColiExpertResidencyEntry *entry) {
    atomic_store_explicit(&entry->variant_lock, 0, memory_order_release);
}

static inline uint64_t coli_expert_residency_allocate_generation(
        ColiExpertResidencyEntry *entry) {
    if (!entry) return 0;
    uint64_t current = atomic_load_explicit(
        &entry->generation_allocator, memory_order_acquire);
    for (;;) {
        if (current == UINT64_MAX) return 0;
        uint64_t next = current + 1;
        if (atomic_compare_exchange_weak_explicit(
                &entry->generation_allocator, &current, next,
                memory_order_acq_rel, memory_order_acquire))
            return next;
    }
}

static inline int coli_expert_residency_claim_generation(
        ColiExpertResidencyEntry *entry, uint64_t generation) {
    if (!entry || !generation) return -1;
    uint64_t current = atomic_load_explicit(
        &entry->generation_allocator, memory_order_acquire);
    for (;;) {
        if (generation <= current) return -1;
        if (atomic_compare_exchange_weak_explicit(
                &entry->generation_allocator, &current, generation,
                memory_order_acq_rel, memory_order_acquire))
            return 0;
    }
}

static inline int coli_expert_residency_entry_init(
    ColiExpertResidencyEntry *entry, ColiExpertKey key) {
    if (!entry || key.layer < 0 || key.expert < 0) return -1;
    memset(entry, 0, sizeof(*entry));
    entry->key = key;
    atomic_init(&entry->generation_allocator, 0);
    atomic_init(&entry->preferred_variant, -1);
    atomic_init(&entry->variant_lock, 0);
    for (uint32_t i = 0; i < COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY; ++i) {
        ColiExpertResidentVariant *variant = &entry->variants[i];
        atomic_init(&variant->state, COLI_EXPERT_RESIDENCY_COLD);
        atomic_init(&variant->generation, 0);
        atomic_init(&variant->pending_generation, 0);
        atomic_init(&variant->refs, 0);
        atomic_init(&variant->tier_mask, COLI_EXPERT_TIER_NONE);
    }
    return 0;
}

static inline ColiExpertResidencyState coli_expert_residency_variant_state(
        const ColiExpertResidencyEntry *entry, uint32_t variant_id) {
    const ColiExpertResidentVariant *variant =
        coli_expert_residency_variant_const(entry, variant_id);
    return variant ? (ColiExpertResidencyState)atomic_load_explicit(
        &variant->state, memory_order_acquire) : COLI_EXPERT_RESIDENCY_COLD;
}

static inline ColiExpertResidencyState coli_expert_residency_state(
        const ColiExpertResidencyEntry *entry) {
    return coli_expert_residency_variant_state(entry, 0);
}

static inline int coli_expert_residency_preferred_variant(
        const ColiExpertResidencyEntry *entry) {
    return entry ? atomic_load_explicit(
        &entry->preferred_variant, memory_order_acquire) : -1;
}

static inline int coli_expert_residency_release(
        ColiExpertResidencyLease *lease);

static inline int coli_expert_residency_acquire_variant(
        ColiExpertResidencyEntry *entry,
        uint32_t variant_id,
        ColiExpertResidencyLease *lease) {
    if (!entry || !lease) return -1;
    ColiExpertResidentVariant *variant =
        coli_expert_residency_variant(entry, variant_id);
    if (!variant) return -1;
    memset(lease, 0, sizeof(*lease));
    if (atomic_load_explicit(&variant->state, memory_order_acquire) !=
        COLI_EXPERT_RESIDENCY_RESIDENT)
        return 0;
    uint64_t generation = atomic_load_explicit(
        &variant->generation, memory_order_acquire);
    unsigned tier = atomic_load_explicit(
        &variant->tier_mask, memory_order_acquire);
    if (!generation || !tier) return -1;
    atomic_fetch_add_explicit(&variant->refs, 1, memory_order_acq_rel);
    if (atomic_load_explicit(&variant->state, memory_order_acquire) !=
            COLI_EXPERT_RESIDENCY_RESIDENT ||
        atomic_load_explicit(&variant->generation, memory_order_acquire) !=
            generation) {
        atomic_fetch_sub_explicit(&variant->refs, 1, memory_order_acq_rel);
        return 0;
    }
    lease->entry = entry;
    lease->variant = variant;
    lease->variant_id = variant_id;
    lease->key = entry->key;
    lease->generation = generation;
    lease->tier_mask = tier;
    lease->representation = variant->representation;
    lease->resident_bytes = variant->resident_bytes;
    lease->allocation_bytes = variant->allocation_bytes;
    lease->physical = variant->physical;
    return 1;
}

static inline int coli_expert_residency_acquire(
        ColiExpertResidencyEntry *entry, ColiExpertResidencyLease *lease) {
    return coli_expert_residency_acquire_variant(entry, 0, lease);
}

static inline int coli_expert_residency_acquire_preferred(
        ColiExpertResidencyEntry *entry, ColiExpertResidencyLease *lease) {
    if (!entry || !lease) return -1;
    int preferred = atomic_load_explicit(
        &entry->preferred_variant, memory_order_acquire);
    if (preferred >= 0 &&
        preferred < (int)COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY) {
        int hit = coli_expert_residency_acquire_variant(
            entry, (uint32_t)preferred, lease);
        if (hit != 0) return hit;
    }
    for (uint32_t i = 0; i < COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY; ++i) {
        if ((int)i == preferred) continue;
        int hit = coli_expert_residency_acquire_variant(entry, i, lease);
        if (hit > 0) return hit;
        if (hit < 0) return hit;
    }
    memset(lease, 0, sizeof(*lease));
    return 0;
}

static inline int coli_expert_residency_acquire_compatible(
        ColiExpertResidencyEntry *entry,
        const ColiRepresentationId *representation,
        ColiExpertResidencyLease *lease) {
    if (!entry || !representation || !lease ||
        !coli_representation_known(representation))
        return -1;
    int preferred = atomic_load_explicit(
        &entry->preferred_variant, memory_order_acquire);
    if (preferred >= 0 &&
        preferred < (int)COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY) {
        int hit = coli_expert_residency_acquire_variant(
            entry, (uint32_t)preferred, lease);
        if (hit < 0) return hit;
        if (hit > 0) {
            if (coli_representation_equal(
                    &lease->representation, representation))
                return 1;
            if (coli_expert_residency_release(lease) != 0) return -1;
        }
    }
    for (uint32_t i = 0; i < COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY; ++i) {
        if ((int)i == preferred) continue;
        int hit = coli_expert_residency_acquire_variant(entry, i, lease);
        if (hit < 0) return hit;
        if (!hit) continue;
        if (coli_representation_equal(
                &lease->representation, representation))
            return 1;
        if (coli_expert_residency_release(lease) != 0) return -1;
    }
    memset(lease, 0, sizeof(*lease));
    return 0;
}

static inline int coli_expert_residency_lease_view(
        const ColiExpertResidencyLease *lease, ColiExpertResidentView *view) {
    if (!lease || !view || !lease->entry || !lease->variant ||
        !lease->generation)
        return -1;
    view->key = lease->key;
    view->representation = lease->representation;
    view->generation = lease->generation;
    view->tier_mask = lease->tier_mask;
    view->resident_bytes = lease->resident_bytes;
    view->allocation_bytes = lease->allocation_bytes;
    view->physical = lease->physical;
    return 0;
}

static inline int coli_expert_residency_lease_valid(
        const ColiExpertResidencyLease *lease) {
    if (!lease || !lease->entry || !lease->variant ||
        !lease->generation ||
        lease->variant_id >= COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY ||
        lease->variant != &lease->entry->variants[lease->variant_id] ||
        !coli_expert_key_equal(lease->entry->key, lease->key))
        return 0;
    return atomic_load_explicit(
               &lease->variant->state, memory_order_acquire) ==
               COLI_EXPERT_RESIDENCY_RESIDENT &&
        atomic_load_explicit(
               &lease->variant->generation, memory_order_acquire) ==
               lease->generation &&
        atomic_load_explicit(
               &lease->variant->refs, memory_order_acquire) != 0;
}

static inline int coli_expert_residency_query_variant(
        ColiExpertResidencyEntry *entry,
        uint32_t variant_id,
        ColiExpertResidentVariantInfo *info) {
    if (!entry || !info ||
        variant_id >= COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY)
        return -1;
    coli_expert_residency_family_lock(entry);
    ColiExpertResidentVariant *variant = &entry->variants[variant_id];
    info->variant_id = variant_id;
    info->representation = variant->representation;
    info->state = (ColiExpertResidencyState)atomic_load_explicit(
        &variant->state, memory_order_acquire);
    info->generation = atomic_load_explicit(
        &variant->generation, memory_order_acquire);
    info->pending_generation = atomic_load_explicit(
        &variant->pending_generation, memory_order_acquire);
    info->refs = atomic_load_explicit(&variant->refs, memory_order_acquire);
    info->tier_mask = atomic_load_explicit(
        &variant->tier_mask, memory_order_acquire);
    info->resident_bytes = variant->resident_bytes;
    info->allocation_bytes = variant->allocation_bytes;
    info->preferred = atomic_load_explicit(
        &entry->preferred_variant, memory_order_acquire) == (int)variant_id;
    coli_expert_residency_family_unlock(entry);
    return 0;
}

static inline unsigned coli_expert_residency_resident_variant_count(
        const ColiExpertResidencyEntry *entry) {
    if (!entry) return 0;
    unsigned count = 0;
    for (uint32_t i = 0; i < COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY; ++i) {
        if (atomic_load_explicit(
                &entry->variants[i].state, memory_order_acquire) ==
                COLI_EXPERT_RESIDENCY_RESIDENT)
            ++count;
    }
    return count;
}

static inline int coli_expert_residency_set_preferred(
        ColiExpertResidencyEntry *entry, uint32_t variant_id) {
    if (!entry || variant_id >= COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY)
        return -1;
    ColiExpertResidencyLease pin;
    int acquired = coli_expert_residency_acquire_variant(
        entry, variant_id, &pin);
    if (acquired <= 0) return acquired;
    if (!coli_representation_known(&pin.representation)) {
        (void)coli_expert_residency_release(&pin);
        return -1;
    }
    atomic_store_explicit(
        &entry->preferred_variant, (int)variant_id, memory_order_release);
    return coli_expert_residency_release(&pin);
}

static inline ColiExpertRequestResult coli_expert_residency_reserve_variant(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        const ColiRepresentationId *representation,
        unsigned tier_mask,
        uint64_t allocation_bytes,
        uint32_t *variant_id_out,
        uint64_t *generation_out) {
    if (!entry || !budget || !representation ||
        !coli_representation_known(representation) ||
        !tier_mask || !allocation_bytes ||
        !variant_id_out || !generation_out)
        return COLI_EXPERT_REQUEST_INVALID;
    coli_expert_residency_family_lock(entry);
    for (uint32_t i = 0; i < COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY; ++i) {
        ColiExpertResidentVariant *variant = &entry->variants[i];
        int state = atomic_load_explicit(&variant->state, memory_order_acquire);
        if (state == COLI_EXPERT_RESIDENCY_COLD ||
            !coli_representation_equal(
                &variant->representation, representation) ||
            atomic_load_explicit(
                &variant->tier_mask, memory_order_acquire) != tier_mask)
            continue;
        *variant_id_out = i;
        *generation_out = (state == COLI_EXPERT_RESIDENCY_RESIDENT ||
                           state == COLI_EXPERT_RESIDENCY_EVICTING)
            ? atomic_load_explicit(&variant->generation, memory_order_acquire)
            : atomic_load_explicit(
                &variant->pending_generation, memory_order_acquire);
        coli_expert_residency_family_unlock(entry);
        return state == COLI_EXPERT_RESIDENCY_RESIDENT
            ? COLI_EXPERT_REQUEST_HIT
            : COLI_EXPERT_REQUEST_JOIN_INFLIGHT;
    }
    ColiExpertResidentVariant *target = NULL;
    uint32_t target_id = COLI_EXPERT_VARIANT_NONE;
    for (uint32_t i = 0; i < COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY; ++i) {
        ColiExpertResidentVariant *variant = &entry->variants[i];
        int expected = COLI_EXPERT_RESIDENCY_COLD;
        if (atomic_compare_exchange_strong_explicit(
                &variant->state, &expected, COLI_EXPERT_RESIDENCY_RESERVED,
                memory_order_acq_rel, memory_order_acquire)) {
            target = variant;
            target_id = i;
            break;
        }
    }
    if (!target) {
        coli_expert_residency_family_unlock(entry);
        return COLI_EXPERT_REQUEST_NO_SLOT;
    }
    int reserved = coli_expert_residency_budget_try_reserve(
        budget, allocation_bytes);
    if (reserved <= 0) {
        atomic_store_explicit(
            &target->state, COLI_EXPERT_RESIDENCY_COLD, memory_order_release);
        coli_expert_residency_family_unlock(entry);
        return reserved < 0 ? COLI_EXPERT_REQUEST_INVALID
                            : COLI_EXPERT_REQUEST_NO_BUDGET;
    }
    uint64_t generation = coli_expert_residency_allocate_generation(entry);
    if (!generation) {
        (void)coli_expert_residency_budget_cancel(budget, allocation_bytes);
        atomic_store_explicit(
            &target->state, COLI_EXPERT_RESIDENCY_COLD, memory_order_release);
        coli_expert_residency_family_unlock(entry);
        return COLI_EXPERT_REQUEST_INVALID;
    }
    target->representation = *representation;
    target->allocation_bytes = allocation_bytes;
    target->resident_bytes = 0;
    target->physical = NULL;
    target->budget = budget;
    atomic_store_explicit(
        &target->tier_mask, tier_mask, memory_order_release);
    atomic_store_explicit(
        &target->pending_generation, generation, memory_order_release);
    *variant_id_out = target_id;
    *generation_out = generation;
    coli_expert_residency_family_unlock(entry);
    return COLI_EXPERT_REQUEST_LOAD_OWNER;
}

static inline int coli_expert_residency_mark_variant_loading(
        ColiExpertResidencyEntry *entry,
        uint32_t variant_id,
        uint64_t generation) {
    ColiExpertResidentVariant *variant =
        coli_expert_residency_variant(entry, variant_id);
    if (!variant || !generation ||
        atomic_load_explicit(
            &variant->pending_generation, memory_order_acquire) != generation)
        return -1;
    int expected = COLI_EXPERT_RESIDENCY_RESERVED;
    return atomic_compare_exchange_strong_explicit(
        &variant->state, &expected, COLI_EXPERT_RESIDENCY_LOADING,
        memory_order_acq_rel, memory_order_acquire) ? 0 : -1;
}

static inline int coli_expert_residency_mark_variant_preparing(
        ColiExpertResidencyEntry *entry,
        uint32_t variant_id,
        uint64_t generation) {
    ColiExpertResidentVariant *variant =
        coli_expert_residency_variant(entry, variant_id);
    if (!variant || !generation ||
        atomic_load_explicit(
            &variant->pending_generation, memory_order_acquire) != generation)
        return -1;
    int expected = COLI_EXPERT_RESIDENCY_LOADING;
    if (atomic_compare_exchange_strong_explicit(
            &variant->state, &expected, COLI_EXPERT_RESIDENCY_PREPARING,
            memory_order_acq_rel, memory_order_acquire))
        return 0;
    expected = COLI_EXPERT_RESIDENCY_RESERVED;
    return atomic_compare_exchange_strong_explicit(
        &variant->state, &expected, COLI_EXPERT_RESIDENCY_PREPARING,
        memory_order_acq_rel, memory_order_acquire) ? 0 : -1;
}

static inline int coli_expert_residency_publish_variant(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        uint32_t variant_id,
        uint64_t generation,
        uint64_t resident_bytes,
        void *physical) {
    ColiExpertResidentVariant *variant =
        coli_expert_residency_variant(entry, variant_id);
    if (!variant || !budget || !generation ||
        variant->budget != budget ||
        !variant->allocation_bytes || !resident_bytes ||
        resident_bytes > variant->allocation_bytes ||
        !coli_representation_known(&variant->representation) ||
        !atomic_load_explicit(&variant->tier_mask, memory_order_acquire) ||
        atomic_load_explicit(
            &variant->pending_generation, memory_order_acquire) != generation)
        return -1;
    int state = atomic_load_explicit(&variant->state, memory_order_acquire);
    if (state != COLI_EXPERT_RESIDENCY_RESERVED &&
        state != COLI_EXPERT_RESIDENCY_LOADING &&
        state != COLI_EXPERT_RESIDENCY_PREPARING)
        return -1;
    if (coli_expert_residency_budget_publish(
            budget, variant->allocation_bytes) != 0)
        return -1;
    coli_expert_residency_family_lock(entry);
    variant->resident_bytes = resident_bytes;
    variant->physical = physical;
    atomic_store_explicit(
        &variant->generation, generation, memory_order_release);
    atomic_store_explicit(
        &variant->pending_generation, 0, memory_order_release);
    atomic_store_explicit(
        &variant->state, COLI_EXPERT_RESIDENCY_RESIDENT, memory_order_release);
    int expected = -1;
    (void)atomic_compare_exchange_strong_explicit(
        &entry->preferred_variant, &expected, (int)variant_id,
        memory_order_acq_rel, memory_order_acquire);
    coli_expert_residency_family_unlock(entry);
    return 0;
}

static inline int coli_expert_residency_publish_variant_from_source(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        uint32_t variant_id,
        uint64_t generation,
        uint64_t resident_bytes,
        void *physical,
        const ColiExpertResidencyLease *source_lease) {
    if (!entry || !source_lease || source_lease->entry != entry ||
        !coli_expert_residency_lease_valid(source_lease))
        return -1;
    return coli_expert_residency_publish_variant(
        entry, budget, variant_id, generation, resident_bytes, physical);
}

static inline int coli_expert_residency_fail_variant(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        uint32_t variant_id,
        uint64_t generation) {
    ColiExpertResidentVariant *variant =
        coli_expert_residency_variant(entry, variant_id);
    if (!variant || !budget || !generation || variant->budget != budget ||
        atomic_load_explicit(
            &variant->pending_generation, memory_order_acquire) != generation ||
        !variant->allocation_bytes)
        return -1;
    int state = atomic_load_explicit(&variant->state, memory_order_acquire);
    if (state != COLI_EXPERT_RESIDENCY_RESERVED &&
        state != COLI_EXPERT_RESIDENCY_LOADING &&
        state != COLI_EXPERT_RESIDENCY_PREPARING)
        return -1;
    if (coli_expert_residency_budget_cancel(
            budget, variant->allocation_bytes) != 0)
        return -1;
    coli_expert_residency_family_lock(entry);
    variant->allocation_bytes = 0;
    coli_representation_clear(&variant->representation);
    variant->resident_bytes = 0;
    variant->physical = NULL;
    variant->budget = NULL;
    atomic_store_explicit(
        &variant->pending_generation, 0, memory_order_release);
    atomic_store_explicit(
        &variant->tier_mask, COLI_EXPERT_TIER_NONE, memory_order_release);
    atomic_store_explicit(
        &variant->state, COLI_EXPERT_RESIDENCY_COLD, memory_order_release);
    coli_expert_residency_family_unlock(entry);
    return 0;
}

static inline int coli_expert_residency_release(
        ColiExpertResidencyLease *lease) {
    if (!lease || !lease->entry || !lease->variant ||
        !lease->generation ||
        lease->variant_id >= COLI_EXPERT_RESIDENCY_VARIANT_CAPACITY ||
        lease->variant != &lease->entry->variants[lease->variant_id] ||
        !coli_expert_key_equal(lease->entry->key, lease->key) ||
        atomic_load_explicit(
            &lease->variant->generation, memory_order_acquire) !=
            lease->generation)
        return -1;
    unsigned refs = atomic_load_explicit(
        &lease->variant->refs, memory_order_acquire);
    for (;;) {
        if (!refs) return -1;
        if (atomic_compare_exchange_weak_explicit(
                &lease->variant->refs, &refs, refs - 1,
                memory_order_acq_rel, memory_order_acquire)) {
            memset(lease, 0, sizeof(*lease));
            return 0;
        }
    }
}

static inline int coli_expert_residency_begin_variant_evict(
        ColiExpertResidencyEntry *entry, uint32_t variant_id) {
    ColiExpertResidentVariant *variant =
        coli_expert_residency_variant(entry, variant_id);
    if (!variant ||
        atomic_load_explicit(&variant->refs, memory_order_acquire) != 0)
        return 0;
    int preferred = (int)variant_id;
    (void)atomic_compare_exchange_strong_explicit(
        &entry->preferred_variant, &preferred, -1,
        memory_order_acq_rel, memory_order_acquire);
    int expected = COLI_EXPERT_RESIDENCY_RESIDENT;
    if (!atomic_compare_exchange_strong_explicit(
            &variant->state, &expected, COLI_EXPERT_RESIDENCY_EVICTING,
            memory_order_acq_rel, memory_order_acquire))
        return 0;
    if (atomic_load_explicit(&variant->refs, memory_order_acquire) != 0) {
        atomic_store_explicit(
            &variant->state, COLI_EXPERT_RESIDENCY_RESIDENT,
            memory_order_release);
        int no_preferred = -1;
        (void)atomic_compare_exchange_strong_explicit(
            &entry->preferred_variant, &no_preferred, (int)variant_id,
            memory_order_acq_rel, memory_order_acquire);
        return 0;
    }
    return 1;
}

static inline int coli_expert_residency_finish_variant_evict(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        uint32_t variant_id) {
    ColiExpertResidentVariant *variant =
        coli_expert_residency_variant(entry, variant_id);
    if (!variant || !budget || variant->budget != budget ||
        atomic_load_explicit(&variant->state, memory_order_acquire) !=
            COLI_EXPERT_RESIDENCY_EVICTING ||
        atomic_load_explicit(&variant->refs, memory_order_acquire) != 0 ||
        !variant->allocation_bytes)
        return -1;
    if (coli_expert_residency_budget_evict(
            budget, variant->allocation_bytes) != 0)
        return -1;
    coli_expert_residency_family_lock(entry);
    variant->allocation_bytes = 0;
    coli_representation_clear(&variant->representation);
    variant->resident_bytes = 0;
    variant->physical = NULL;
    variant->budget = NULL;
    atomic_store_explicit(
        &variant->pending_generation, 0, memory_order_release);
    atomic_store_explicit(
        &variant->tier_mask, COLI_EXPERT_TIER_NONE, memory_order_release);
    atomic_store_explicit(
        &variant->state, COLI_EXPERT_RESIDENCY_COLD, memory_order_release);
    coli_expert_residency_family_unlock(entry);
    return 0;
}

static inline ColiExpertRequestResult coli_expert_residency_request(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        uint64_t allocation_bytes,
        ColiExpertResidencyLease *lease) {
    if (!entry || !budget || !allocation_bytes || !lease)
        return COLI_EXPERT_REQUEST_INVALID;
    ColiExpertResidentVariant *variant = &entry->variants[0];
    for (;;) {
        int state = atomic_load_explicit(&variant->state, memory_order_acquire);
        if (state == COLI_EXPERT_RESIDENCY_RESIDENT) {
            int hit = coli_expert_residency_acquire_variant(entry, 0, lease);
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
        coli_expert_residency_family_lock(entry);
        int expected = COLI_EXPERT_RESIDENCY_COLD;
        if (!atomic_compare_exchange_strong_explicit(
                &variant->state, &expected, COLI_EXPERT_RESIDENCY_RESERVED,
                memory_order_acq_rel, memory_order_acquire)) {
            coli_expert_residency_family_unlock(entry);
            continue;
        }
        int reserved = coli_expert_residency_budget_try_reserve(
            budget, allocation_bytes);
        if (reserved <= 0) {
            atomic_store_explicit(
                &variant->state, COLI_EXPERT_RESIDENCY_COLD,
                memory_order_release);
            coli_expert_residency_family_unlock(entry);
            return reserved < 0 ? COLI_EXPERT_REQUEST_INVALID
                                : COLI_EXPERT_REQUEST_NO_BUDGET;
        }
        variant->allocation_bytes = allocation_bytes;
        variant->resident_bytes = 0;
        variant->physical = NULL;
        variant->budget = budget;
        coli_representation_clear(&variant->representation);
        atomic_store_explicit(
            &variant->pending_generation, 0, memory_order_release);
        atomic_store_explicit(
            &variant->tier_mask, COLI_EXPERT_TIER_NONE, memory_order_release);
        memset(lease, 0, sizeof(*lease));
        coli_expert_residency_family_unlock(entry);
        return COLI_EXPERT_REQUEST_LOAD_OWNER;
    }
}

static inline int coli_expert_residency_mark_loading(
        ColiExpertResidencyEntry *entry) {
    if (!entry) return -1;
    ColiExpertResidentVariant *variant = &entry->variants[0];
    int expected = COLI_EXPERT_RESIDENCY_RESERVED;
    return atomic_compare_exchange_strong_explicit(
        &variant->state, &expected, COLI_EXPERT_RESIDENCY_LOADING,
        memory_order_acq_rel, memory_order_acquire) ? 0 : -1;
}

static inline int coli_expert_residency_mark_preparing(
        ColiExpertResidencyEntry *entry) {
    if (!entry) return -1;
    ColiExpertResidentVariant *variant = &entry->variants[0];
    int expected = COLI_EXPERT_RESIDENCY_LOADING;
    if (atomic_compare_exchange_strong_explicit(
            &variant->state, &expected, COLI_EXPERT_RESIDENCY_PREPARING,
            memory_order_acq_rel, memory_order_acquire))
        return 0;
    expected = COLI_EXPERT_RESIDENCY_RESERVED;
    return atomic_compare_exchange_strong_explicit(
        &variant->state, &expected, COLI_EXPERT_RESIDENCY_PREPARING,
        memory_order_acq_rel, memory_order_acquire) ? 0 : -1;
}

static inline int coli_expert_residency_publish_legacy_impl(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        uint64_t generation,
        unsigned tier_mask,
        const ColiRepresentationId *representation,
        uint64_t resident_bytes,
        void *physical,
        int require_representation) {
    if (!entry || !budget || !generation || !tier_mask)
        return -1;
    ColiExpertResidentVariant *variant = &entry->variants[0];
    if (variant->budget != budget || !variant->allocation_bytes ||
        !resident_bytes || resident_bytes > variant->allocation_bytes)
        return -1;
    if (require_representation &&
        !coli_representation_known(representation))
        return -1;
    int state = atomic_load_explicit(&variant->state, memory_order_acquire);
    if (state != COLI_EXPERT_RESIDENCY_RESERVED &&
        state != COLI_EXPERT_RESIDENCY_LOADING &&
        state != COLI_EXPERT_RESIDENCY_PREPARING)
        return -1;
    if (coli_expert_residency_claim_generation(entry, generation) != 0)
        return -1;
    if (coli_expert_residency_budget_publish(
            budget, variant->allocation_bytes) != 0)
        return -1;
    coli_expert_residency_family_lock(entry);
    if (representation) variant->representation = *representation;
    else coli_representation_clear(&variant->representation);
    variant->resident_bytes = resident_bytes;
    variant->physical = physical;
    atomic_store_explicit(
        &variant->tier_mask, tier_mask, memory_order_release);
    atomic_store_explicit(
        &variant->generation, generation, memory_order_release);
    atomic_store_explicit(
        &variant->pending_generation, 0, memory_order_release);
    atomic_store_explicit(
        &variant->state, COLI_EXPERT_RESIDENCY_RESIDENT, memory_order_release);
    int expected = -1;
    (void)atomic_compare_exchange_strong_explicit(
        &entry->preferred_variant, &expected, 0,
        memory_order_acq_rel, memory_order_acquire);
    coli_expert_residency_family_unlock(entry);
    return 0;
}

static inline int coli_expert_residency_publish(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        uint64_t generation,
        unsigned tier_mask) {
    return coli_expert_residency_publish_legacy_impl(
        entry, budget, generation, tier_mask, NULL,
        entry ? entry->variants[0].allocation_bytes : 0, NULL, 0);
}

static inline int coli_expert_residency_publish_representation(
        ColiExpertResidencyEntry *entry,
        ColiExpertResidencyBudget *budget,
        uint64_t generation,
        unsigned tier_mask,
        const ColiRepresentationId *representation,
        uint64_t resident_bytes,
        void *physical) {
    return coli_expert_residency_publish_legacy_impl(
        entry, budget, generation, tier_mask, representation,
        resident_bytes, physical, 1);
}

static inline int coli_expert_residency_fail_load(
        ColiExpertResidencyEntry *entry, ColiExpertResidencyBudget *budget) {
    if (!entry || !budget) return -1;
    ColiExpertResidentVariant *variant = &entry->variants[0];
    if (variant->budget != budget || !variant->allocation_bytes)
        return -1;
    int state = atomic_load_explicit(&variant->state, memory_order_acquire);
    if (state != COLI_EXPERT_RESIDENCY_RESERVED &&
        state != COLI_EXPERT_RESIDENCY_LOADING &&
        state != COLI_EXPERT_RESIDENCY_PREPARING)
        return -1;
    if (coli_expert_residency_budget_cancel(
            budget, variant->allocation_bytes) != 0)
        return -1;
    coli_expert_residency_family_lock(entry);
    variant->allocation_bytes = 0;
    coli_representation_clear(&variant->representation);
    variant->resident_bytes = 0;
    variant->physical = NULL;
    variant->budget = NULL;
    atomic_store_explicit(
        &variant->pending_generation, 0, memory_order_release);
    atomic_store_explicit(
        &variant->tier_mask, COLI_EXPERT_TIER_NONE, memory_order_release);
    atomic_store_explicit(
        &variant->state, COLI_EXPERT_RESIDENCY_COLD, memory_order_release);
    coli_expert_residency_family_unlock(entry);
    return 0;
}

static inline int coli_expert_residency_begin_evict(
        ColiExpertResidencyEntry *entry) {
    return coli_expert_residency_begin_variant_evict(entry, 0);
}

static inline int coli_expert_residency_finish_evict(
        ColiExpertResidencyEntry *entry, ColiExpertResidencyBudget *budget) {
    return coli_expert_residency_finish_variant_evict(entry, budget, 0);
}

#endif
