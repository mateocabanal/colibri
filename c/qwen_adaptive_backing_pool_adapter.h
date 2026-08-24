#ifndef COLIBRI_QWEN_ADAPTIVE_BACKING_POOL_ADAPTER_H
#define COLIBRI_QWEN_ADAPTIVE_BACKING_POOL_ADAPTER_H

#include "expert_backing_pool.h"

/*
 * Qwen physical adapter for the generic transient backing pool.
 *
 * This header is included before Qwen's Slot/Model declarations on purpose.
 * Everything that touches those types is a macro and therefore expands only at
 * the existing route seam, after the concrete engine types are visible.
 *
 * The key invariant is that the shared planner's `transient_slots` is GLOBAL.
 * We therefore circulate at most that many physical expert allocations between
 * layers. Persistent selected experts remain in their layer caches. A transient
 * backing keeps its aligned/pinned/Metal-visible allocation while its logical
 * expert identity changes, matching the old Qwen LRU's cheap overwrite behavior
 * without turning the transient allowance into `layers * topk` residency.
 */

typedef struct {
    ColiExpertBackingPool pool;
    void *model;
    size_t slot_count;
    int initialized;
    int last_layer;
    uint64_t deposited;
    uint64_t returned;
    uint64_t attached;
    uint64_t promoted;
    uint64_t hard_reclaims;
    uint64_t resets;
} ColiQwenAdaptiveBackingState;

static ColiQwenAdaptiveBackingState g_qwen_adaptive_backing = {
    .last_layer = -1,
};

/* One stable owner pointer is enough to associate a Qwen Slot with the generic
 * pool lease. All experts in one model have one format/layout; the remaining
 * pointers are views into the same allocation or format-specific companions. */
#define QWEN_ADAPTIVE_BACKING_ID(_s) \
    ((void *)((_s)->mxbase ? (void *)(_s)->mxbase : \
      (_s)->mxg ? (void *)(_s)->mxg : \
      (_s)->g4 ? (void *)(_s)->g4 : \
      (_s)->g ? (void *)(_s)->g : \
      (_s)->bgu ? (void *)(_s)->bgu : \
      (_s)->gu ? (void *)(_s)->gu : NULL))

#define QWEN_ADAPTIVE_ZERO_SLOT(_s) do { \
    memset((_s), 0, sizeof(*(_s))); \
    (_s)->eid = -2; \
    (_s)->loading_eid = -1; \
    (_s)->mio_slot = -1; \
    QWEN_ADAPTIVE_METALIO_REINIT((_s)); \
} while (0)

/* Reset pool metadata when model identity or transient concurrency changes.
 * AVAILABLE payloads are detached from every layer and therefore owned here;
 * free them fully. LEASED payloads are only shadow metadata: the live backing
 * is still owned by a layer Slot and must not be freed twice. */
#define QWEN_ADAPTIVE_BACKING_ENSURE(_m, _slots) do { \
    size_t _qab_need = (size_t)(_slots); \
    if (_qab_need < 1) _qab_need = 1; \
    if (g_qwen_adaptive_backing.initialized && \
        (g_qwen_adaptive_backing.model != (void *)(_m) || \
         g_qwen_adaptive_backing.slot_count != _qab_need)) { \
        for (size_t _qab_i = 0; _qab_i < g_qwen_adaptive_backing.pool.slot_count; _qab_i++) { \
            ColiExpertBackingSlot *_qab_ps = &g_qwen_adaptive_backing.pool.slots[_qab_i]; \
            Slot *_qab_shadow = (Slot *)_qab_ps->payload; \
            if (!_qab_shadow) continue; \
            if (atomic_load_explicit(&_qab_ps->state, memory_order_acquire) == \
                    COLI_EXPERT_BACKING_AVAILABLE) \
                QWEN_ADAPTIVE_RELEASE_SLOT(_qab_shadow); \
            free(_qab_shadow); \
            _qab_ps->payload = NULL; \
        } \
        coli_expert_backing_pool_destroy(&g_qwen_adaptive_backing.pool); \
        memset(&g_qwen_adaptive_backing, 0, sizeof(g_qwen_adaptive_backing)); \
        g_qwen_adaptive_backing.last_layer = -1; \
        g_qwen_adaptive_backing.resets++; \
    } \
    if (!g_qwen_adaptive_backing.initialized && \
        coli_expert_backing_pool_init(&g_qwen_adaptive_backing.pool, _qab_need) == 0) { \
        g_qwen_adaptive_backing.initialized = 1; \
        g_qwen_adaptive_backing.model = (void *)(_m); \
        g_qwen_adaptive_backing.slot_count = _qab_need; \
        g_qwen_adaptive_backing.last_layer = -1; \
    } \
} while (0)

/* Locate the live generic lease corresponding to a Qwen Slot by its stable
 * physical owner pointer. The pool is tiny (top-k / measured concurrency), so
 * a bounded linear scan is cheaper and safer than adding model-specific lease
 * fields to the engine's Slot ABI. */
#define QWEN_ADAPTIVE_FIND_LEASED(_slot, _out_index) do { \
    (_out_index) = SIZE_MAX; \
    void *_qab_owner = QWEN_ADAPTIVE_BACKING_ID((_slot)); \
    if (_qab_owner && g_qwen_adaptive_backing.initialized) { \
        for (size_t _qab_i = 0; _qab_i < g_qwen_adaptive_backing.pool.slot_count; _qab_i++) { \
            ColiExpertBackingSlot *_qab_ps = &g_qwen_adaptive_backing.pool.slots[_qab_i]; \
            if (atomic_load_explicit(&_qab_ps->state, memory_order_acquire) != \
                    COLI_EXPERT_BACKING_LEASED || !_qab_ps->payload) \
                continue; \
            Slot *_qab_shadow = (Slot *)_qab_ps->payload; \
            if (QWEN_ADAPTIVE_BACKING_ID(_qab_shadow) == _qab_owner) { \
                (_out_index) = _qab_i; \
                break; \
            } \
        } \
    } \
} while (0)

/* Return a borrowed transient allocation to the global pool without touching
 * the backing allocation or Metal registration. */
#define QWEN_ADAPTIVE_RETURN_BORROWED(_slot, _returned) do { \
    (_returned) = 0; \
    size_t _qab_idx; \
    QWEN_ADAPTIVE_FIND_LEASED((_slot), _qab_idx); \
    if (_qab_idx != SIZE_MAX) { \
        ColiExpertBackingSlot *_qab_ps = &g_qwen_adaptive_backing.pool.slots[_qab_idx]; \
        Slot *_qab_shadow = (Slot *)_qab_ps->payload; \
        uint64_t _qab_gen = atomic_load_explicit(&_qab_ps->generation, memory_order_acquire); \
        *_qab_shadow = *(_slot); \
        QWEN_ADAPTIVE_ZERO_SLOT((_slot)); \
        if (coli_expert_backing_pool_release_generation( \
                &g_qwen_adaptive_backing.pool, _qab_idx, _qab_gen) == 0) { \
            g_qwen_adaptive_backing.returned++; \
            (_returned) = 1; \
        } \
    } \
} while (0)

/* A transient backing that becomes selected/persistent leaves the transient
 * pool without being freed. This is a physical promotion, not a copy. */
#define QWEN_ADAPTIVE_PROMOTE_BORROWED(_slot, _promoted) do { \
    (_promoted) = 0; \
    size_t _qab_idx; \
    QWEN_ADAPTIVE_FIND_LEASED((_slot), _qab_idx); \
    if (_qab_idx != SIZE_MAX) { \
        ColiExpertBackingSlot *_qab_ps = &g_qwen_adaptive_backing.pool.slots[_qab_idx]; \
        uint64_t _qab_gen = atomic_load_explicit(&_qab_ps->generation, memory_order_acquire); \
        ColiExpertBackingLease _qab_lease = { \
            .pool = &g_qwen_adaptive_backing.pool, \
            .index = _qab_idx, \
            .generation = _qab_gen, \
        }; \
        void *_qab_old = NULL; \
        if (coli_expert_backing_lease_set_payload( \
                &_qab_lease, NULL, 0, 0, 0, &_qab_old) == 0 && \
            coli_expert_backing_pool_release(&_qab_lease) == 0) { \
            free(_qab_old); /* shadow only; live backing remains in _slot */ \
            g_qwen_adaptive_backing.promoted++; \
            (_promoted) = 1; \
        } \
    } \
} while (0)

/* Move a legacy/non-pool transient Slot into an EMPTY generic pool slot. Never
 * replace an existing reusable payload just to save another allocation: if the
 * global pool is already physically populated, this extra backing is outside
 * the transient budget and should be reclaimed. */
#define QWEN_ADAPTIVE_DEPOSIT_TRANSIENT(_slot, _bytes, _deposited) do { \
    (_deposited) = 0; \
    if (QWEN_ADAPTIVE_BACKING_ID((_slot)) && g_qwen_adaptive_backing.initialized) { \
        ColiExpertBackingLease _qab_lease = {0}; \
        for (size_t _qab_i = 0; _qab_i < g_qwen_adaptive_backing.pool.slot_count; _qab_i++) { \
            if (coli_expert_backing_try_index( \
                    &g_qwen_adaptive_backing.pool, _qab_i, \
                    (uint64_t)(_bytes), 1u, (uint64_t)(_bytes), \
                    0, 1, &_qab_lease)) \
                break; \
        } \
        if (_qab_lease.pool) { \
            Slot *_qab_shadow = (Slot *)malloc(sizeof(Slot)); \
            if (_qab_shadow) { \
                *_qab_shadow = *(_slot); \
                if (coli_expert_backing_lease_set_payload( \
                        &_qab_lease, _qab_shadow, (uint64_t)(_bytes), \
                        (uint64_t)(_bytes), 1u, NULL) == 0 && \
                    coli_expert_backing_pool_release(&_qab_lease) == 0) { \
                    QWEN_ADAPTIVE_ZERO_SLOT((_slot)); \
                    g_qwen_adaptive_backing.deposited++; \
                    (_deposited) = 1; \
                } else { \
                    free(_qab_shadow); \
                    if (_qab_lease.pool) \
                        (void)coli_expert_backing_pool_release(&_qab_lease); \
                } \
            } else { \
                (void)coli_expert_backing_pool_release(&_qab_lease); \
            } \
        } \
    } \
} while (0)

/* Attach compatible AVAILABLE pool payloads to inactive capacity immediately
 * after lc->n. `expert_get()` already uses exactly that next Slot on a miss, so
 * its existing idempotent format allocators overwrite the borrowed backing
 * instead of allocating/registering another slab. Do not increment lc->n: the
 * slot becomes logically active only when expert_get assigns an expert. */
#define QWEN_ADAPTIVE_ATTACH_TRANSIENT(_lc, _bytes) do { \
    if (g_qwen_adaptive_backing.initialized) { \
        for (int _qab_dst_i = (_lc)->n; _qab_dst_i < (_lc)->cap; _qab_dst_i++) { \
            Slot *_qab_dst = &(_lc)->slots[_qab_dst_i]; \
            if (QWEN_ADAPTIVE_BACKING_ID(_qab_dst)) continue; \
            ColiExpertBackingLease _qab_lease = {0}; \
            for (size_t _qab_i = 0; _qab_i < g_qwen_adaptive_backing.pool.slot_count; _qab_i++) { \
                if (coli_expert_backing_try_index( \
                        &g_qwen_adaptive_backing.pool, _qab_i, \
                        (uint64_t)(_bytes), 1u, (uint64_t)(_bytes), \
                        1, 0, &_qab_lease)) \
                    break; \
            } \
            if (!_qab_lease.pool) break; \
            ColiExpertBackingSlot *_qab_ps = \
                coli_expert_backing_lease_slot(&_qab_lease); \
            Slot *_qab_shadow = _qab_ps ? (Slot *)_qab_ps->payload : NULL; \
            if (!_qab_shadow) { \
                (void)coli_expert_backing_pool_release(&_qab_lease); \
                break; \
            } \
            *_qab_dst = *_qab_shadow; \
            _qab_dst->eid = -2; \
            _qab_dst->loading_eid = -1; \
            _qab_dst->pinned = 0; \
            _qab_dst->used = 0; \
            g_qwen_adaptive_backing.attached++; \
        } \
    } \
} while (0)

/* Replace the first-generation physical reconciler. Policy and byte-envelope
 * selection stay in the generic adaptive controller/resource planner; this
 * layer only realizes persistent pins plus a GLOBAL reusable transient pool. */
#undef QWEN_ADAPTIVE_RECONCILE
#define QWEN_ADAPTIVE_RECONCILE(_m, _layer) do { \
    Model *_qa_m = (_m); \
    int _qa_l = (_layer); \
    if (_qa_m && _qa_m->is_pinned && _qa_l >= 0 && _qa_l < _qa_m->c.n_layers) { \
        uint64_t _qa_rb = g_qwen_adaptive.resident_bytes_per_expert; \
        pthread_mutex_lock(&g_mx); \
        LCache *_qa_probe = &_qa_m->cache[_qa_l]; \
        for (int _qa_i = 0; _qa_i < _qa_probe->n && !_qa_rb; _qa_i++) \
            if (_qa_probe->slots[_qa_i].eid >= 0) \
                _qa_rb = qwen_adaptive_resident_bytes_for_format( \
                    _qa_probe->slots[_qa_i].fmt, _qa_m->c.hidden, _qa_m->c.moe_inter); \
        pthread_mutex_unlock(&g_mx); \
        if (_qa_rb) { \
            uint64_t _qa_transient = (uint64_t)_qa_m->c.topk; \
            if (g_prefetch) _qa_transient *= 2; \
            int _qa_plan_changed = qwen_adaptive_adapter_maybe_plan( \
                (void *)_qa_m, _qa_rb, _qa_transient); \
            int _qa_prev = g_qwen_adaptive_backing.last_layer; \
            if (_qa_prev >= 0 && _qa_prev != _qa_l && \
                _qa_prev < _qa_m->c.n_layers) { \
                unsigned char *_qa_prev_pin = _qa_m->is_pinned + \
                    (size_t)_qa_prev * (size_t)_qa_m->c.n_experts; \
                (void)qwen_adaptive_adapter_copy_layer( \
                    (void *)_qa_m, _qa_prev, _qa_prev_pin, _qa_m->c.n_experts); \
            } \
            unsigned char *_qa_pin = _qa_m->is_pinned + \
                (size_t)_qa_l * (size_t)_qa_m->c.n_experts; \
            int _qa_selected = qwen_adaptive_adapter_copy_layer( \
                (void *)_qa_m, _qa_l, _qa_pin, _qa_m->c.n_experts); \
            if (_qa_selected >= 0) { \
                pthread_mutex_lock(&g_mx); \
                QWEN_ADAPTIVE_BACKING_ENSURE(_qa_m, _qa_transient); \
                if (g_qwen_adaptive_backing.initialized && \
                    _qa_prev >= 0 && _qa_prev != _qa_l && \
                    _qa_prev < _qa_m->c.n_layers) { \
                    LCache *_qab_prev_lc = &_qa_m->cache[_qa_prev]; \
                    unsigned char *_qab_prev_pin = _qa_m->is_pinned + \
                        (size_t)_qa_prev * (size_t)_qa_m->c.n_experts; \
                    /* Return unused borrowed backing parked beyond logical n. */ \
                    for (int _qab_i = _qab_prev_lc->n; _qab_i < _qab_prev_lc->cap; _qab_i++) { \
                        Slot *_qab_s = &_qab_prev_lc->slots[_qab_i]; \
                        if (!QWEN_ADAPTIVE_BACKING_ID(_qab_s)) continue; \
                        int _qab_returned; \
                        QWEN_ADAPTIVE_RETURN_BORROWED(_qab_s, _qab_returned); \
                    } \
                    /* Retire non-persistent logical identities, but keep up to
                     * the globally bounded backing count physically reusable. */ \
                    for (int _qab_i = _qab_prev_lc->n - 1; _qab_i >= 0; _qab_i--) { \
                        Slot *_qab_s = &_qab_prev_lc->slots[_qab_i]; \
                        int _qab_e = _qab_s->eid >= 0 ? _qab_s->eid : _qab_s->loading_eid; \
                        _qab_s->pinned = _qab_e >= 0 && _qab_e < _qa_m->c.n_experts \
                            ? _qab_prev_pin[_qab_e] != 0 : 0; \
                        if (_qab_s->eid < 0) continue; \
                        if (_qab_s->pinned) { \
                            int _qab_promoted; \
                            QWEN_ADAPTIVE_PROMOTE_BORROWED(_qab_s, _qab_promoted); \
                            continue; \
                        } \
                        int _qab_done; \
                        QWEN_ADAPTIVE_RETURN_BORROWED(_qab_s, _qab_done); \
                        if (!_qab_done) \
                            QWEN_ADAPTIVE_DEPOSIT_TRANSIENT(_qab_s, _qa_rb, _qab_done); \
                        if (!_qab_done) { \
                            QWEN_ADAPTIVE_RELEASE_SLOT(_qab_s); \
                            g_qwen_adaptive_backing.hard_reclaims++; \
                        } \
                        int _qab_last = _qab_prev_lc->n - 1; \
                        if (_qab_i != _qab_last) \
                            _qab_prev_lc->slots[_qab_i] = _qab_prev_lc->slots[_qab_last]; \
                        QWEN_ADAPTIVE_ZERO_SLOT(&_qab_prev_lc->slots[_qab_last]); \
                        _qab_prev_lc->n--; \
                    } \
                } \
                LCache *_qa_lc = &_qa_m->cache[_qa_l]; \
                for (int _qa_i = 0; _qa_i < _qa_lc->n; _qa_i++) { \
                    Slot *_qa_s = &_qa_lc->slots[_qa_i]; \
                    int _qa_e = _qa_s->eid >= 0 ? _qa_s->eid : _qa_s->loading_eid; \
                    _qa_s->pinned = _qa_e >= 0 && _qa_e < _qa_m->c.n_experts \
                        ? _qa_pin[_qa_e] != 0 : 0; \
                    if (_qa_s->pinned) { \
                        int _qa_promoted; \
                        QWEN_ADAPTIVE_PROMOTE_BORROWED(_qa_s, _qa_promoted); \
                    } \
                } \
                int _qa_desired = _qa_selected + (int)_qa_transient; \
                if (_qa_desired < (int)_qa_transient) _qa_desired = (int)_qa_transient; \
                if (_qa_desired > _qa_m->c.n_experts) _qa_desired = _qa_m->c.n_experts; \
                if (_qa_desired < _qa_lc->n) _qa_desired = _qa_lc->n; \
                if (_qa_desired != _qa_lc->cap) { \
                    /* Any inactive borrowed backing must return before realloc
                     * can move/drop the descriptor array. */ \
                    for (int _qa_i = _qa_lc->n; _qa_i < _qa_lc->cap; _qa_i++) { \
                        Slot *_qa_s = &_qa_lc->slots[_qa_i]; \
                        if (!QWEN_ADAPTIVE_BACKING_ID(_qa_s)) continue; \
                        int _qa_returned; \
                        QWEN_ADAPTIVE_RETURN_BORROWED(_qa_s, _qa_returned); \
                    } \
                    int _qa_oldcap = _qa_lc->cap; \
                    Slot *_qa_new = realloc(_qa_lc->slots, \
                        (size_t)_qa_desired * sizeof(Slot)); \
                    if (_qa_new) { \
                        _qa_lc->slots = _qa_new; \
                        if (_qa_desired > _qa_oldcap) { \
                            memset(_qa_lc->slots + _qa_oldcap, 0, \
                                   (size_t)(_qa_desired - _qa_oldcap) * sizeof(Slot)); \
                            for (int _qa_i = _qa_oldcap; _qa_i < _qa_desired; _qa_i++) { \
                                _qa_lc->slots[_qa_i].eid = -2; \
                                _qa_lc->slots[_qa_i].loading_eid = -1; \
                                _qa_lc->slots[_qa_i].mio_slot = -1; \
                                QWEN_ADAPTIVE_METALIO_REINIT(&_qa_lc->slots[_qa_i]); \
                            } \
                        } \
                        _qa_lc->cap = _qa_desired; \
                    } \
                } \
                QWEN_ADAPTIVE_ATTACH_TRANSIENT(_qa_lc, _qa_rb); \
                g_qwen_adaptive_backing.last_layer = _qa_l; \
                if (_qa_plan_changed) { \
                    ColiExpertBackingPoolStats _qa_bs = \
                        coli_expert_backing_pool_stats(&g_qwen_adaptive_backing.pool); \
                    fprintf(stderr, \
                        "qwen_backing_pool slots=%zu payloads=%zu leased=%zu " \
                        "bytes=%.2fMiB attaches=%llu returns=%llu promotions=%llu " \
                        "hard_reclaims=%llu reuses=%llu waits=%llu\n", \
                        g_qwen_adaptive_backing.slot_count, _qa_bs.slots_with_payload, \
                        _qa_bs.leased_slots, _qa_bs.allocated_bytes / (1024.0 * 1024.0), \
                        (unsigned long long)g_qwen_adaptive_backing.attached, \
                        (unsigned long long)g_qwen_adaptive_backing.returned, \
                        (unsigned long long)g_qwen_adaptive_backing.promoted, \
                        (unsigned long long)g_qwen_adaptive_backing.hard_reclaims, \
                        (unsigned long long)_qa_bs.reuses, \
                        (unsigned long long)_qa_bs.waits); \
                } \
                pthread_mutex_unlock(&g_mx); \
            } \
        } \
    } \
} while (0)

#endif /* COLIBRI_QWEN_ADAPTIVE_BACKING_POOL_ADAPTER_H */
