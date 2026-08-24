#ifndef COLIBRI_EXPERT_BACKING_POOL_H
#define COLIBRI_EXPERT_BACKING_POOL_H

/*
 * Backend-neutral physical backing pool for transient MoE execution slots.
 *
 * Logical expert residency and physical allocation lifetime are deliberately
 * separate. A logical (layer, expert) generation may come and go while an
 * expensive aligned/pinned/device-visible allocation remains available for a
 * later generation. The runtime owns only slot state, generations, byte/layout
 * compatibility and telemetry; adapters own the opaque payload and decide how
 * to load/execute/reset it.
 *
 * The pool is globally bounded. `slot_count` therefore describes actual
 * transient concurrency, not `layers * slot_count`. Persistent expert backing
 * is budgeted separately by the residency/resource planner.
 */

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    COLI_EXPERT_BACKING_AVAILABLE = 0,
    COLI_EXPERT_BACKING_LEASED = 1,
} ColiExpertBackingState;

typedef struct {
    atomic_int state;
    atomic_uint_fast64_t generation;
    void *payload;
    uint64_t allocation_bytes;
    uint64_t layout_tag;
    uint32_t tier_mask;
} ColiExpertBackingSlot;

typedef struct {
    ColiExpertBackingSlot *slots;
    size_t slot_count;
    atomic_uint_fast64_t next_generation;
    atomic_uint_fast64_t acquires;
    atomic_uint_fast64_t reuses;
    atomic_uint_fast64_t repurposes;
    atomic_uint_fast64_t releases;
    atomic_uint_fast64_t waits;
} ColiExpertBackingPool;

typedef struct {
    ColiExpertBackingPool *pool;
    size_t index;
    uint64_t generation;
    int reused;
} ColiExpertBackingLease;

typedef struct {
    uint64_t acquires;
    uint64_t reuses;
    uint64_t repurposes;
    uint64_t releases;
    uint64_t waits;
    uint64_t allocated_bytes;
    size_t slots_with_payload;
    size_t leased_slots;
} ColiExpertBackingPoolStats;

static inline int coli_expert_backing_pool_init(
    ColiExpertBackingPool *pool, size_t slot_count) {
    if (!pool || !slot_count ||
        slot_count > SIZE_MAX / sizeof(ColiExpertBackingSlot))
        return -1;
    memset(pool, 0, sizeof(*pool));
    pool->slots = (ColiExpertBackingSlot *)calloc(
        slot_count, sizeof(ColiExpertBackingSlot));
    if (!pool->slots) return -1;
    pool->slot_count = slot_count;
    atomic_init(&pool->next_generation, 1);
    atomic_init(&pool->acquires, 0);
    atomic_init(&pool->reuses, 0);
    atomic_init(&pool->repurposes, 0);
    atomic_init(&pool->releases, 0);
    atomic_init(&pool->waits, 0);
    for (size_t i = 0; i < slot_count; i++) {
        atomic_init(&pool->slots[i].state, COLI_EXPERT_BACKING_AVAILABLE);
        atomic_init(&pool->slots[i].generation, 0);
    }
    return 0;
}

/* The adapter owns payload destruction. Destroy succeeds only when no physical
 * slot is leased; callers may then walk payloads, free/unregister them, and
 * finally release the pool metadata. */
static inline int coli_expert_backing_pool_can_destroy(
    const ColiExpertBackingPool *pool) {
    if (!pool || !pool->slots) return 1;
    for (size_t i = 0; i < pool->slot_count; i++)
        if (atomic_load_explicit(&pool->slots[i].state,
                                 memory_order_acquire) ==
            COLI_EXPERT_BACKING_LEASED)
            return 0;
    return 1;
}

static inline void coli_expert_backing_pool_destroy(
    ColiExpertBackingPool *pool) {
    if (!pool) return;
    free(pool->slots);
    memset(pool, 0, sizeof(*pool));
}

static inline int coli_expert_backing_compatible(
    const ColiExpertBackingSlot *slot, uint64_t layout_tag,
    uint32_t tier_mask, uint64_t min_bytes) {
    if (!slot || !slot->payload) return 0;
    if (layout_tag && slot->layout_tag != layout_tag) return 0;
    if (tier_mask && (slot->tier_mask & tier_mask) != tier_mask) return 0;
    return slot->allocation_bytes >= min_bytes;
}

static inline int coli_expert_backing_try_index(
    ColiExpertBackingPool *pool, size_t index,
    uint64_t layout_tag, uint32_t tier_mask, uint64_t min_bytes,
    int require_compatible, int require_empty,
    ColiExpertBackingLease *lease) {
    ColiExpertBackingSlot *slot = &pool->slots[index];
    if (require_compatible && !coli_expert_backing_compatible(
            slot, layout_tag, tier_mask, min_bytes))
        return 0;
    if (require_empty && slot->payload) return 0;

    int expected = COLI_EXPERT_BACKING_AVAILABLE;
    if (!atomic_compare_exchange_strong_explicit(
            &slot->state, &expected, COLI_EXPERT_BACKING_LEASED,
            memory_order_acq_rel, memory_order_acquire))
        return 0;

    uint64_t generation = atomic_fetch_add_explicit(
        &pool->next_generation, 1, memory_order_relaxed);
    if (!generation)
        generation = atomic_fetch_add_explicit(
            &pool->next_generation, 1, memory_order_relaxed);
    atomic_store_explicit(&slot->generation, generation, memory_order_release);
    lease->pool = pool;
    lease->index = index;
    lease->generation = generation;
    lease->reused = coli_expert_backing_compatible(
        slot, layout_tag, tier_mask, min_bytes);
    atomic_fetch_add_explicit(&pool->acquires, 1, memory_order_relaxed);
    if (lease->reused)
        atomic_fetch_add_explicit(&pool->reuses, 1, memory_order_relaxed);
    else if (slot->payload)
        atomic_fetch_add_explicit(&pool->repurposes, 1, memory_order_relaxed);
    return 1;
}

/* Prefer an already-compatible physical allocation, then an empty metadata
 * slot, then an incompatible available payload that the adapter may repurpose.
 * Returning 0 means all globally bounded transient slots are currently leased. */
static inline int coli_expert_backing_pool_acquire(
    ColiExpertBackingPool *pool, uint64_t layout_tag,
    uint32_t tier_mask, uint64_t min_bytes,
    ColiExpertBackingLease *lease) {
    if (!pool || !pool->slots || !pool->slot_count || !lease) return -1;
    memset(lease, 0, sizeof(*lease));
    for (size_t i = 0; i < pool->slot_count; i++)
        if (coli_expert_backing_try_index(
                pool, i, layout_tag, tier_mask, min_bytes, 1, 0, lease))
            return 1;
    for (size_t i = 0; i < pool->slot_count; i++)
        if (coli_expert_backing_try_index(
                pool, i, layout_tag, tier_mask, min_bytes, 0, 1, lease))
            return 1;
    for (size_t i = 0; i < pool->slot_count; i++)
        if (coli_expert_backing_try_index(
                pool, i, layout_tag, tier_mask, min_bytes, 0, 0, lease))
            return 1;
    atomic_fetch_add_explicit(&pool->waits, 1, memory_order_relaxed);
    return 0;
}

static inline ColiExpertBackingSlot *coli_expert_backing_lease_slot(
    const ColiExpertBackingLease *lease) {
    if (!lease || !lease->pool || !lease->pool->slots ||
        lease->index >= lease->pool->slot_count || !lease->generation)
        return NULL;
    ColiExpertBackingSlot *slot = &lease->pool->slots[lease->index];
    if (atomic_load_explicit(&slot->state, memory_order_acquire) !=
            COLI_EXPERT_BACKING_LEASED ||
        atomic_load_explicit(&slot->generation, memory_order_acquire) !=
            lease->generation)
        return NULL;
    return slot;
}

/* Replace/attach adapter-owned payload while the slot is exclusively leased.
 * The old payload is returned to the caller; the generic runtime never frees
 * backend allocations or assumes their concrete type. */
static inline int coli_expert_backing_lease_set_payload(
    ColiExpertBackingLease *lease, void *payload,
    uint64_t allocation_bytes, uint64_t layout_tag, uint32_t tier_mask,
    void **old_payload) {
    ColiExpertBackingSlot *slot = coli_expert_backing_lease_slot(lease);
    if (!slot) return -1;
    if (old_payload) *old_payload = slot->payload;
    slot->payload = payload;
    slot->allocation_bytes = payload ? allocation_bytes : 0;
    slot->layout_tag = payload ? layout_tag : 0;
    slot->tier_mask = payload ? tier_mask : 0;
    return 0;
}

static inline int coli_expert_backing_pool_release(
    ColiExpertBackingLease *lease) {
    ColiExpertBackingSlot *slot = coli_expert_backing_lease_slot(lease);
    if (!slot) return -1;
    atomic_store_explicit(&slot->state, COLI_EXPERT_BACKING_AVAILABLE,
                          memory_order_release);
    atomic_fetch_add_explicit(&lease->pool->releases, 1, memory_order_relaxed);
    memset(lease, 0, sizeof(*lease));
    return 0;
}

/* Adapter convenience for returning a lease recovered from backend-specific
 * ownership metadata. The generation check prevents stale owners from making a
 * newer physical generation available. */
static inline int coli_expert_backing_pool_release_generation(
    ColiExpertBackingPool *pool, size_t index, uint64_t generation) {
    if (!pool || !pool->slots || index >= pool->slot_count || !generation)
        return -1;
    ColiExpertBackingLease lease = {
        .pool = pool, .index = index, .generation = generation,
    };
    return coli_expert_backing_pool_release(&lease);
}

static inline ColiExpertBackingPoolStats coli_expert_backing_pool_stats(
    const ColiExpertBackingPool *pool) {
    ColiExpertBackingPoolStats stats = {0};
    if (!pool) return stats;
    stats.acquires = atomic_load_explicit(&pool->acquires, memory_order_relaxed);
    stats.reuses = atomic_load_explicit(&pool->reuses, memory_order_relaxed);
    stats.repurposes = atomic_load_explicit(&pool->repurposes, memory_order_relaxed);
    stats.releases = atomic_load_explicit(&pool->releases, memory_order_relaxed);
    stats.waits = atomic_load_explicit(&pool->waits, memory_order_relaxed);
    if (!pool->slots) return stats;
    for (size_t i = 0; i < pool->slot_count; i++) {
        const ColiExpertBackingSlot *slot = &pool->slots[i];
        if (slot->payload) {
            stats.slots_with_payload++;
            if (UINT64_MAX - stats.allocated_bytes < slot->allocation_bytes)
                stats.allocated_bytes = UINT64_MAX;
            else
                stats.allocated_bytes += slot->allocation_bytes;
        }
        if (atomic_load_explicit(&slot->state, memory_order_relaxed) ==
            COLI_EXPERT_BACKING_LEASED)
            stats.leased_slots++;
    }
    return stats;
}

#endif /* COLIBRI_EXPERT_BACKING_POOL_H */
