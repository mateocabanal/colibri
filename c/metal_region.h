#ifndef COLIBRI_METAL_REGION_H
#define COLIBRI_METAL_REGION_H

#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

typedef struct {
    uint64_t region_id;
    uint64_t generation;
    size_t offset;
    size_t bytes;
} ColiMetalRegionRef;

typedef struct {
    void *base;
    size_t bytes;
    uint64_t region_id;
    atomic_uint_fast64_t generation;
    atomic_uint inflight;
    atomic_int published;
} ColiMetalRegion;

static inline int coli_metal_region_init(ColiMetalRegion *region,
                                         void *base,
                                         size_t bytes,
                                         uint64_t region_id) {
    if (!region || !base || !bytes || !region_id) return -1;
    region->base = base;
    region->bytes = bytes;
    region->region_id = region_id;
    atomic_init(&region->generation, 0);
    atomic_init(&region->inflight, 0);
    atomic_init(&region->published, 0);
    return 0;
}

/* Mark the current generation unavailable to new dispatch before an expert
 * slot is overwritten. Existing users remain counted in `inflight`; callers
 * may reuse the physical bytes only after coli_metal_region_can_overwrite(). */
static inline void coli_metal_region_begin_overwrite(ColiMetalRegion *region) {
    if (!region) return;
    atomic_store_explicit(&region->published, 0, memory_order_release);
}

static inline int coli_metal_region_can_overwrite(const ColiMetalRegion *region) {
    return region &&
        atomic_load_explicit(&region->inflight, memory_order_acquire) == 0;
}

/* Publish freshly prepared bytes under a new generation. Generation zero is
 * reserved for the unpublished/initial state. */
static inline int coli_metal_region_publish(ColiMetalRegion *region,
                                            uint64_t generation) {
    if (!region || !generation ||
        atomic_load_explicit(&region->published, memory_order_acquire) ||
        atomic_load_explicit(&region->inflight, memory_order_acquire) != 0)
        return -1;
    atomic_store_explicit(&region->generation, generation, memory_order_release);
    atomic_store_explicit(&region->published, 1, memory_order_release);
    return 0;
}

static inline uint64_t coli_metal_region_generation(const ColiMetalRegion *region) {
    return region
        ? atomic_load_explicit(&region->generation, memory_order_acquire) : 0;
}

static inline unsigned coli_metal_region_inflight(const ColiMetalRegion *region) {
    return region
        ? atomic_load_explicit(&region->inflight, memory_order_acquire) : 0;
}

static inline int coli_metal_region_ref(const ColiMetalRegion *region,
                                        uint64_t generation,
                                        size_t offset,
                                        size_t bytes,
                                        ColiMetalRegionRef *out) {
    if (!region || !out || !generation || offset > region->bytes ||
        bytes > region->bytes - offset ||
        !atomic_load_explicit(&region->published, memory_order_acquire) ||
        atomic_load_explicit(&region->generation, memory_order_acquire) != generation)
        return -1;
    out->region_id = region->region_id;
    out->generation = generation;
    out->offset = offset;
    out->bytes = bytes;
    return 0;
}

/* Retain is deliberately generation-aware. The post-increment recheck closes
 * the race where overwrite begins between the optimistic validation and the
 * inflight increment: a stale retain backs itself out and fails. */
static inline int coli_metal_region_retain(ColiMetalRegion *region,
                                           uint64_t expected_generation) {
    if (!region || !expected_generation) return -1;
    if (!atomic_load_explicit(&region->published, memory_order_acquire) ||
        atomic_load_explicit(&region->generation, memory_order_acquire) !=
            expected_generation)
        return -1;

    atomic_fetch_add_explicit(&region->inflight, 1, memory_order_acq_rel);
    if (!atomic_load_explicit(&region->published, memory_order_acquire) ||
        atomic_load_explicit(&region->generation, memory_order_acquire) !=
            expected_generation) {
        atomic_fetch_sub_explicit(&region->inflight, 1, memory_order_acq_rel);
        return -1;
    }
    return 0;
}

static inline int coli_metal_region_release(ColiMetalRegion *region,
                                            uint64_t expected_generation) {
    if (!region || !expected_generation ||
        atomic_load_explicit(&region->generation, memory_order_acquire) !=
            expected_generation)
        return -1;

    /* A plain load followed by fetch_sub can underflow if two erroneous release
     * calls race while the count is one. CAS makes the zero check and decrement
     * one atomic transition, so misuse fails closed instead of wrapping inflight. */
    unsigned before = atomic_load_explicit(&region->inflight, memory_order_acquire);
    while (before) {
        if (atomic_compare_exchange_weak_explicit(
                &region->inflight, &before, before - 1,
                memory_order_acq_rel, memory_order_acquire))
            return 0;
    }
    return -1;
}

static inline int coli_metal_region_ref_matches(const ColiMetalRegion *region,
                                                const ColiMetalRegionRef *ref) {
    if (!region || !ref || ref->region_id != region->region_id ||
        !ref->generation || ref->offset > region->bytes ||
        ref->bytes > region->bytes - ref->offset)
        return 0;
    return atomic_load_explicit(&region->published, memory_order_acquire) &&
        atomic_load_explicit(&region->generation, memory_order_acquire) ==
            ref->generation;
}

#endif
