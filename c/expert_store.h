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
struct ColiExpertActivationSample;

typedef struct {
    int layer;
    int expert;
} ColiExpertKey;

typedef enum {
    COLI_EXPERT_PHASE_UNKNOWN = 0,
    COLI_EXPERT_PHASE_PREFILL = 1,
    COLI_EXPERT_PHASE_DECODE = 2,
} ColiExpertPhase;

/* Optional logical request identity for routed-expert telemetry. This is kept
 * separate from ColiExpertKey: `(layer, expert)` is stable storage identity,
 * while request/token/rank/weight describe one use of that identity. Stores
 * that do not consume tracing context may ignore this structure entirely. */
typedef struct {
    uint64_t request_id;
    int64_t token_position;
    int route_rank;
    float route_weight;
    ColiExpertPhase phase;
} ColiExpertRequestContext;

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
 * - lookup_context(), when implemented, has exactly the same lease semantics;
 *   its context is observational metadata and must never change model output.
 * - Logical activation observation is also non-failing/observational. Engines
 *   may report routing multiplicity before batching/union; a store/runtime that
 *   does not consume adaptive residency signals simply ignores it. Policy
 *   accounting must never become a model-correctness dependency.
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
    /* Optional context-aware lookup. Appended so existing positional ops
     * initializers remain source-compatible and default this field to NULL. */
    int (*lookup_context)(ColiExpertStore *store, ColiExpertKey key,
                          const ColiExpertRequestContext *context,
                          ColiExpertView *view);
    /* Optional shared adaptive-residency signal. The sample type is forward
     * declared to keep expert_store.h independent from the tracker/policy
     * implementation and avoid a header cycle. */
    void (*observe_activations)(
        ColiExpertStore *store,
        const struct ColiExpertActivationSample *samples,
        size_t count);
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

static inline int coli_expert_lookup_context(
        ColiExpertStore *store, ColiExpertKey key,
        const ColiExpertRequestContext *context, ColiExpertView *view) {
    if (!store || !store->ops) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    int result;
    if (store->ops->lookup_context)
        result = store->ops->lookup_context(store, key, context, view);
    else if (store->ops->lookup)
        result = store->ops->lookup(store, key, view);
    else {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    if (result != 0 && view) memset(view, 0, sizeof(*view));
    return result;
}

static inline void coli_expert_observe_activations(
        ColiExpertStore *store,
        const struct ColiExpertActivationSample *samples,
        size_t count) {
    if (store && store->ops && store->ops->observe_activations &&
        samples && count)
        store->ops->observe_activations(store, samples, count);
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
