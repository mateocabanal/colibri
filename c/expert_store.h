#ifndef COLIBRI_EXPERT_STORE_H
#define COLIBRI_EXPERT_STORE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ColiExpertStore ColiExpertStore;

typedef struct {
    int layer;
    int expert;
} ColiExpertKey;

typedef struct {
    ColiExpertKey key;
    ColiTensorView gate;
    ColiTensorView down;
    ColiTensorView up;
    void *lease;
    /* Exact physical-slot generation retained by this lease. Implementations
     * that do not reuse physical slots may leave this zero. Streamed stores
     * should validate it on release so an old view can never alias a newer
     * expert after slot reuse. */
    uint64_t lease_generation;
} ColiExpertView;

typedef struct {
    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t prefetched;
    uint64_t prefetch_hits;
    uint64_t bytes_read;
    uint64_t resident_bytes;
    uint64_t capacity_bytes;

    /* Backend-neutral residency diagnostics. Appended to preserve the existing
     * leading layout used by older callers. */
    uint64_t persistent_hits;
    uint64_t transient_hits;
    uint64_t inflight_joins;
    uint64_t loads_started;
    uint64_t loads_failed;
    uint64_t evictions;
    uint64_t slot_waits;
    uint64_t peak_inflight;
} ColiExpertStoreStats;

/*
 * ExpertStore lease contract:
 *
 * - After a successful lookup(), the caller must call release() exactly once
 *   on the same view (same store). Do not copy ColiExpertView; the lease is
 *   not shareable. Do not pass a view that already holds an active lease to
 *   lookup().
 * - On lookup failure the view is cleared; the caller must not use it and
 *   must not call release().
 * - release() clears the entire view. release() on an already-cleared or
 *   zero-initialized view is a no-op.
 * - A streamed/reusable store should publish a monotonically changing
 *   lease_generation when a physical slot is repurposed. The lease pins that
 *   exact generation until release/backend completion.
 * - destroy() requires zero active leases (debug builds assert).
 * - Thread-safety is implementation-specific. Callers must not assume that
 *   lookup/release/prefetch/stats/destroy are safe to call concurrently on
 *   the same store unless the concrete store documents that guarantee.
 *   Views must not be used concurrently from multiple threads.
 * - prefetch() is advisory, holds no lease, and must not evict a slot that
 *   still has an active lease.
 */
typedef struct {
    /* Returns zero on success. The view remains valid until release(). */
    int (*lookup)(ColiExpertStore *store, ColiExpertKey key,
                  ColiExpertView *view);
    void (*release)(ColiExpertStore *store, ColiExpertView *view);
    /* Prefetch is advisory. Unsupported or rejected requests return zero. */
    int (*prefetch)(ColiExpertStore *store, const ColiExpertKey *keys,
                    size_t count);
    void (*stats)(const ColiExpertStore *store, ColiExpertStoreStats *stats);
    void (*destroy)(ColiExpertStore *store);
} ColiExpertStoreOps;

struct ColiExpertStore {
    const ColiExpertStoreOps *ops;
    void *state;
};

static inline int coli_expert_lookup(ColiExpertStore *store,
                                     ColiExpertKey key,
                                     ColiExpertView *view) {
    if (!store || !store->ops || !store->ops->lookup) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    int result = store->ops->lookup(store, key, view);
    if (result != 0 && view) memset(view, 0, sizeof(*view));
    return result;
}

static inline void coli_expert_release(ColiExpertStore *store,
                                       ColiExpertView *view) {
    if (store && store->ops && store->ops->release)
        store->ops->release(store, view);
    if (view) memset(view, 0, sizeof(*view));
}

#ifdef __cplusplus
}
#endif

#endif
