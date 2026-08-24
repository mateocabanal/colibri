#ifndef COLIBRI_QWEN_ADAPTIVE_RESIDENCY_ADAPTER_H
#define COLIBRI_QWEN_ADAPTIVE_RESIDENCY_ADAPTER_H

#include "moe_adaptive_residency.h"
#include "reuse_projection.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Thin Qwen adapter for the shared MoE adaptive-residency controller.
 *
 * This header is intentionally included after route_trace.h but before Qwen's
 * Model/Slot declarations. The only pieces that know Qwen's structs are macros
 * expanded later at the existing rt_init()/rt_route() call sites. Policy,
 * activation accounting, common-horizon projection and global expert ranking
 * remain in shared runtime headers.
 *
 * Why wrap route_trace instead of adding a second router callback in qwen_moe.c:
 * rt_route already receives the logical top-k rows before that information is
 * forgotten. `(rt_route)(...)` calls the original static function even after
 * the function-like macro below is defined, so ROUTE_TRACE/COLI_USAGE remain
 * byte-for-byte on their existing path.
 */

typedef struct {
    pthread_mutex_t mutex;
    int initialized;
    int enabled;
    void *model;
    int layers;
    int experts;
    int topk;
    int legacy_cap;

    /* Qwen's layer-major prefill emits rt_route(layer,row,...) one row at a
     * time. The first routed layer is buffered until the next layer tells us
     * the span width; subsequent layers can then publish exact token epochs.
     * Decode is the same shape with width=1. */
    int first_layer;
    int last_layer;
    int span_width;
    int first_k;
    int first_rows;
    int first_capacity_rows;
    int *first_ids;
    uint64_t span_base_epoch;
    uint64_t next_span_epoch;

    ColiMoeAdaptiveResidency adaptive;
    uint64_t selected_count;
    uint64_t selected_bytes;
    uint64_t resident_bytes_per_expert;
} ColiQwenAdaptiveAdapter;

static ColiQwenAdaptiveAdapter g_qwen_adaptive = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .first_layer = -1,
    .last_layer = -1,
};

static inline int qwen_adaptive_env_disabled(void) {
    const char *value = getenv("COLI_ADAPTIVE_RESIDENCY");
    if (value && value[0] && strcmp(value, "0") == 0) return 1;

    /* Explicit legacy placement controls are benchmarking/override contracts.
     * PIN=auto is only a historical seed and does not disable adaptation. */
    value = getenv("HOT");
    if (value && value[0] && atoi(value) > 0) return 1;
    value = getenv("PIN");
    if (value && value[0] && strcmp(value, "auto") != 0) return 1;
    return 0;
}

static void qwen_adaptive_adapter_destroy(void) {
    pthread_mutex_lock(&g_qwen_adaptive.mutex);
    if (g_qwen_adaptive.initialized)
        coli_moe_adaptive_destroy(&g_qwen_adaptive.adaptive);
    free(g_qwen_adaptive.first_ids);
    g_qwen_adaptive.first_ids = NULL;
    g_qwen_adaptive.first_capacity_rows = 0;
    g_qwen_adaptive.first_rows = 0;
    g_qwen_adaptive.initialized = 0;
    g_qwen_adaptive.enabled = 0;
    pthread_mutex_unlock(&g_qwen_adaptive.mutex);
}

static void qwen_adaptive_adapter_init(
    void *model, int layers, int experts, int topk, int legacy_cap) {
    pthread_mutex_lock(&g_qwen_adaptive.mutex);
    if (g_qwen_adaptive.initialized) {
        coli_moe_adaptive_destroy(&g_qwen_adaptive.adaptive);
        free(g_qwen_adaptive.first_ids);
        g_qwen_adaptive.first_ids = NULL;
    }
    memset((unsigned char *)&g_qwen_adaptive + sizeof(g_qwen_adaptive.mutex), 0,
           sizeof(g_qwen_adaptive) - sizeof(g_qwen_adaptive.mutex));
    g_qwen_adaptive.first_layer = -1;
    g_qwen_adaptive.last_layer = -1;
    g_qwen_adaptive.model = model;
    g_qwen_adaptive.layers = layers;
    g_qwen_adaptive.experts = experts;
    g_qwen_adaptive.topk = topk;
    g_qwen_adaptive.legacy_cap = legacy_cap;
    g_qwen_adaptive.span_base_epoch = 1;
    g_qwen_adaptive.next_span_epoch = 1;
    g_qwen_adaptive.initialized = 1;
    g_qwen_adaptive.enabled = !qwen_adaptive_env_disabled() &&
        layers > 0 && experts > 0 && topk > 0 && legacy_cap > 0 &&
        coli_moe_adaptive_init(&g_qwen_adaptive.adaptive, layers, experts) == 0;
    if (!g_qwen_adaptive.enabled) {
        fprintf(stderr,
                "qwen_residency adaptive=off policy=legacy-lru-hot "
                "hint=COLI_ADAPTIVE_RESIDENCY=1-and-no-explicit-HOT/PIN\n");
    } else {
        fprintf(stderr,
                "qwen_residency adaptive=shared policy=frequency-decay-global "
                "legacy_cap=%d topk=%d planner_horizon=%llu confidence_mass=%llu\n",
                legacy_cap, topk,
                (unsigned long long)g_qwen_adaptive.adaptive.policy.planning_horizon_epochs,
                (unsigned long long)g_qwen_adaptive.adaptive.policy.planner_confidence_mass);
    }
    pthread_mutex_unlock(&g_qwen_adaptive.mutex);
}

static inline int qwen_adaptive_first_reserve_rows_locked(int rows, int k) {
    if (rows <= g_qwen_adaptive.first_capacity_rows &&
        k == g_qwen_adaptive.first_k)
        return 0;
    if (rows < 1 || k < 1 ||
        (size_t)rows > SIZE_MAX / (size_t)k ||
        (size_t)rows * (size_t)k > SIZE_MAX / sizeof(int))
        return -1;
    int *next = realloc(g_qwen_adaptive.first_ids,
                        (size_t)rows * (size_t)k * sizeof(int));
    if (!next) return -1;
    g_qwen_adaptive.first_ids = next;
    g_qwen_adaptive.first_capacity_rows = rows;
    g_qwen_adaptive.first_k = k;
    return 0;
}

static inline void qwen_adaptive_flush_first_locked(void) {
    if (!g_qwen_adaptive.first_rows || g_qwen_adaptive.first_layer < 0 ||
        g_qwen_adaptive.first_k < 1)
        return;
    int width = g_qwen_adaptive.first_rows;
    ColiExpertPhase phase = width > 1
        ? COLI_EXPERT_PHASE_PREFILL : COLI_EXPERT_PHASE_DECODE;
    for (int row = 0; row < width; row++)
        (void)coli_moe_adaptive_observe_routes(
            &g_qwen_adaptive.adaptive, g_qwen_adaptive.first_layer,
            g_qwen_adaptive.first_ids + (size_t)row * g_qwen_adaptive.first_k,
            (size_t)g_qwen_adaptive.first_k, phase,
            g_qwen_adaptive.span_base_epoch + (uint64_t)row);
    g_qwen_adaptive.span_width = width;
    g_qwen_adaptive.next_span_epoch =
        g_qwen_adaptive.span_base_epoch + (uint64_t)width;
    g_qwen_adaptive.first_rows = 0;
}

/* Observe one Qwen route row. Qwen's engine is layer-major: all rows for layer
 * L arrive before layer L+1, and a new token/chunk restarts at the first routed
 * layer. This adapter turns that engine-specific call order into the shared
 * logical-token epoch contract. */
static void qwen_adaptive_adapter_observe(
    void *model, int layer, int row, const int *ids, int k) {
    pthread_mutex_lock(&g_qwen_adaptive.mutex);
    if (!g_qwen_adaptive.enabled || model != g_qwen_adaptive.model ||
        !ids || k < 1 || row < 0 || layer < 0 ||
        layer >= g_qwen_adaptive.layers) {
        pthread_mutex_unlock(&g_qwen_adaptive.mutex);
        return;
    }

    int new_span = 0;
    if (g_qwen_adaptive.first_layer < 0) {
        g_qwen_adaptive.first_layer = layer;
    } else if (row == 0 &&
               (layer < g_qwen_adaptive.last_layer ||
                (layer == g_qwen_adaptive.first_layer &&
                 g_qwen_adaptive.last_layer == g_qwen_adaptive.first_layer &&
                 g_qwen_adaptive.first_rows > 0))) {
        new_span = 1;
    }

    if (new_span) {
        qwen_adaptive_flush_first_locked();
        g_qwen_adaptive.span_base_epoch = g_qwen_adaptive.next_span_epoch;
        g_qwen_adaptive.span_width = 0;
        g_qwen_adaptive.first_layer = layer;
        g_qwen_adaptive.first_rows = 0;
        g_qwen_adaptive.first_k = 0;
    }

    if (layer == g_qwen_adaptive.first_layer &&
        (g_qwen_adaptive.span_width == 0 ||
         g_qwen_adaptive.last_layer == g_qwen_adaptive.first_layer)) {
        if (g_qwen_adaptive.first_k && g_qwen_adaptive.first_k != k) {
            pthread_mutex_unlock(&g_qwen_adaptive.mutex);
            return;
        }
        if (qwen_adaptive_first_reserve_rows_locked(row + 1, k) == 0) {
            memcpy(g_qwen_adaptive.first_ids + (size_t)row * k,
                   ids, (size_t)k * sizeof(int));
            if (row + 1 > g_qwen_adaptive.first_rows)
                g_qwen_adaptive.first_rows = row + 1;
        }
        g_qwen_adaptive.last_layer = layer;
        pthread_mutex_unlock(&g_qwen_adaptive.mutex);
        return;
    }

    if (g_qwen_adaptive.last_layer == g_qwen_adaptive.first_layer &&
        g_qwen_adaptive.first_rows > 0)
        qwen_adaptive_flush_first_locked();

    int width = g_qwen_adaptive.span_width > 0
        ? g_qwen_adaptive.span_width : 1;
    if (row < width) {
        ColiExpertPhase phase = width > 1
            ? COLI_EXPERT_PHASE_PREFILL : COLI_EXPERT_PHASE_DECODE;
        (void)coli_moe_adaptive_observe_routes(
            &g_qwen_adaptive.adaptive, layer, ids, (size_t)k, phase,
            g_qwen_adaptive.span_base_epoch + (uint64_t)row);
    }
    g_qwen_adaptive.last_layer = layer;
    pthread_mutex_unlock(&g_qwen_adaptive.mutex);
}

static int qwen_adaptive_selected_before(void *context, ColiExpertKey key) {
    ColiQwenAdaptiveAdapter *state = (ColiQwenAdaptiveAdapter *)context;
    return state ? coli_moe_adaptive_selected(&state->adaptive, key) : 0;
}

static int qwen_adaptive_adapter_maybe_plan(
    void *model, uint64_t resident_bytes_per_expert,
    uint64_t transient_slots) {
    int changed = 0;
    pthread_mutex_lock(&g_qwen_adaptive.mutex);
    if (!g_qwen_adaptive.enabled || model != g_qwen_adaptive.model ||
        !resident_bytes_per_expert || !g_qwen_adaptive.layers ||
        !g_qwen_adaptive.legacy_cap) {
        pthread_mutex_unlock(&g_qwen_adaptive.mutex);
        return 0;
    }

    uint64_t total_slots = (uint64_t)g_qwen_adaptive.legacy_cap *
                           (uint64_t)g_qwen_adaptive.layers;
    uint64_t persistent_slots = total_slots > transient_slots
        ? total_slots - transient_slots : 0;
    uint64_t budget = coli_reuse_sat_mul(
        persistent_slots, resident_bytes_per_expert);
    ColiResourceSelection selection = {0};
    int rc = coli_moe_adaptive_select_experts(
        &g_qwen_adaptive.adaptive, budget,
        resident_bytes_per_expert, resident_bytes_per_expert, 0,
        COLI_RESOURCE_VALUE_BYTES,
        qwen_adaptive_selected_before, &g_qwen_adaptive, &selection);
    if (rc > 0) {
        g_qwen_adaptive.selected_count = selection.selected_count;
        g_qwen_adaptive.selected_bytes = selection.selected_resident_bytes;
        g_qwen_adaptive.resident_bytes_per_expert = resident_bytes_per_expert;
        fprintf(stderr,
                "qwen_resource_planner status=replan count=%llu epoch=%llu "
                "selected=%llu expert_budget=%.2fGiB transient_slots=%llu\n",
                (unsigned long long)g_qwen_adaptive.adaptive.replan_count,
                (unsigned long long)g_qwen_adaptive.adaptive.current_epoch,
                (unsigned long long)selection.selected_count,
                budget / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)transient_slots);
        changed = 1;
    }
    pthread_mutex_unlock(&g_qwen_adaptive.mutex);
    return changed;
}

/* Copy one layer of the selected map into Qwen's existing pin map. Return -1
 * only until the first planner decision. A later valid zero-expert plan must
 * clear stale pins rather than being confused with cold start. */
static int qwen_adaptive_adapter_copy_layer(
    void *model, int layer, unsigned char *out, int experts) {
    int count = -1;
    pthread_mutex_lock(&g_qwen_adaptive.mutex);
    if (g_qwen_adaptive.enabled && model == g_qwen_adaptive.model &&
        g_qwen_adaptive.adaptive.replan_count > 0 && layer >= 0 &&
        layer < g_qwen_adaptive.layers && experts == g_qwen_adaptive.experts &&
        out) {
        size_t base = (size_t)layer * (size_t)experts;
        memcpy(out, g_qwen_adaptive.adaptive.selected + base, (size_t)experts);
        count = 0;
        for (int e = 0; e < experts; e++) count += out[e] != 0;
    }
    pthread_mutex_unlock(&g_qwen_adaptive.mutex);
    return count;
}

static uint64_t qwen_adaptive_resident_bytes_for_format(
    int fmt, int hidden, int intermediate) {
    if (hidden < 1 || intermediate < 1) return 0;
    uint64_t H = (uint64_t)hidden, I = (uint64_t)intermediate;
    if (H > UINT64_MAX / I) return 0;
    uint64_t hi = H * I;
    if (hi > UINT64_MAX / 3) return 0;
    if (fmt == 0) return coli_reuse_sat_mul(hi * 3, 4);
    if (fmt == 16) return coli_reuse_sat_mul(hi * 3, 2);
    if (fmt == 8) {
        uint64_t scales = coli_reuse_sat_mul(I * 2 + H, 4);
        return coli_reuse_sat_add(hi * 3, scales);
    }
    if (fmt == 4 || fmt == 5) {
        uint64_t rbH = fmt == 5 ? ((H + 63) / 64) * 24 : (H + 1) / 2;
        uint64_t rbI = fmt == 5 ? ((I + 63) / 64) * 24 : (I + 1) / 2;
        uint64_t weights = coli_reuse_sat_add(
            coli_reuse_sat_mul(coli_reuse_sat_mul(rbH, I), 2),
            coli_reuse_sat_mul(rbI, H));
        uint64_t ngH = (H + 63) / 64, ngI = (I + 63) / 64;
        uint64_t scales = coli_reuse_sat_add(
            coli_reuse_sat_mul(coli_reuse_sat_mul(I, ngH), 2),
            coli_reuse_sat_mul(H, ngI));
        scales = coli_reuse_sat_mul(scales, 4);
        return coli_reuse_sat_add(weights, scales);
    }
    if (fmt == 7) {
        uint64_t weights = (hi * 3 + 1) / 2;
        uint64_t scales = coli_reuse_sat_add(
            coli_reuse_sat_mul(I, (H + 31) / 32) * 2,
            coli_reuse_sat_mul(H, (I + 31) / 32));
        return coli_reuse_sat_add(weights, scales);
    }
    return 0;
}

#ifdef COLI_METAL
#define QWEN_ADAPTIVE_METAL_UNREGISTER(_s) do { \
    if (g_metal_compute) { \
        coli_metal_unregister((_s)->g); \
        coli_metal_unregister((_s)->g4); \
        coli_metal_unregister((_s)->gs); \
        coli_metal_unregister((_s)->g4s); \
        if ((_s)->mxbase) coli_metal_unregister((_s)->mxbase); \
        else { coli_metal_unregister((_s)->mxg); coli_metal_unregister((_s)->mxgs); } \
    } \
} while (0)
#else
#define QWEN_ADAPTIVE_METAL_UNREGISTER(_s) ((void)0)
#endif

#ifdef COLI_METALIO
#define QWEN_ADAPTIVE_METALIO_RELEASE(_s) do { \
    if ((_s)->mio_slot >= 0) metalio_slot_free((_s)->mio_slot); \
} while (0)
#define QWEN_ADAPTIVE_METALIO_REINIT(_s) do { \
    if (g_metal_io && metalio_active()) { (_s)->mio = 1; (_s)->mio_slot = -1; } \
} while (0)
#else
#define QWEN_ADAPTIVE_METALIO_RELEASE(_s) ((void)0)
#define QWEN_ADAPTIVE_METALIO_REINIT(_s) ((void)0)
#endif

/* Keep the macro-local name distinct from the caller expression. In C the
 * scope of a declarator begins before its initializer, so `Slot *_qa_s =
 * (_qa_s)` reads the new uninitialized variable when the caller also names its
 * pointer `_qa_s` (the sanitizer-caught first-reclaim crash). */
#define QWEN_ADAPTIVE_RELEASE_SLOT(_slot_expr) do { \
    Slot *_qa_release = (_slot_expr); \
    int _qa_mio_resident = _qa_release->mio_resident; \
    QWEN_ADAPTIVE_METAL_UNREGISTER(_qa_release); \
    QWEN_ADAPTIVE_METALIO_RELEASE(_qa_release); \
    if (!_qa_mio_resident) { \
        free(_qa_release->gu); \
        free(_qa_release->bgu); \
        free(_qa_release->g); \
        free(_qa_release->gs); \
        free(_qa_release->g4); \
        free(_qa_release->g4s); \
        if (_qa_release->mxbase) free(_qa_release->mxbase); \
        else { free(_qa_release->mxg); free(_qa_release->mxgs); } \
    } \
    memset(_qa_release, 0, sizeof(*_qa_release)); \
    _qa_release->eid = -2; \
    _qa_release->loading_eid = -1; \
    _qa_release->mio_slot = -1; \
    QWEN_ADAPTIVE_METALIO_REINIT(_qa_release); \
} while (0)

#define QWEN_ADAPTIVE_RECONCILE(_m, _layer) do { \
    Model *_qa_m = (_m); \
    int _qa_l = (_layer); \
    if (_qa_m && _qa_m->is_pinned && _qa_l >= 0 && _qa_l < _qa_m->c.n_layers) { \
        uint64_t _qa_rb = 0; \
        pthread_mutex_lock(&g_mx); \
        LCache *_qa_lc = &_qa_m->cache[_qa_l]; \
        for (int _qa_i = 0; _qa_i < _qa_lc->n && !_qa_rb; _qa_i++) \
            if (_qa_lc->slots[_qa_i].eid >= 0) \
                _qa_rb = qwen_adaptive_resident_bytes_for_format( \
                    _qa_lc->slots[_qa_i].fmt, _qa_m->c.hidden, _qa_m->c.moe_inter); \
        pthread_mutex_unlock(&g_mx); \
        if (_qa_rb) { \
            uint64_t _qa_transient = (uint64_t)_qa_m->c.topk; \
            if (g_prefetch) _qa_transient *= 2; \
            (void)qwen_adaptive_adapter_maybe_plan((void *)_qa_m, _qa_rb, _qa_transient); \
            unsigned char *_qa_pin = _qa_m->is_pinned + \
                (size_t)_qa_l * (size_t)_qa_m->c.n_experts; \
            int _qa_selected = qwen_adaptive_adapter_copy_layer( \
                (void *)_qa_m, _qa_l, _qa_pin, _qa_m->c.n_experts); \
            if (_qa_selected >= 0) { \
                pthread_mutex_lock(&g_mx); \
                _qa_lc = &_qa_m->cache[_qa_l]; \
                for (int _qa_i = 0; _qa_i < _qa_lc->n; _qa_i++) { \
                    Slot *_qa_s = &_qa_lc->slots[_qa_i]; \
                    int _qa_e = _qa_s->eid >= 0 ? _qa_s->eid : _qa_s->loading_eid; \
                    _qa_s->pinned = _qa_e >= 0 && _qa_e < _qa_m->c.n_experts \
                        ? _qa_pin[_qa_e] != 0 : 0; \
                } \
                for (int _qa_i = _qa_lc->n - 1; _qa_i >= 0; _qa_i--) { \
                    Slot *_qa_s = &_qa_lc->slots[_qa_i]; \
                    if (_qa_s->eid < 0 || _qa_s->pinned) continue; \
                    QWEN_ADAPTIVE_RELEASE_SLOT(_qa_s); \
                    int _qa_last = _qa_lc->n - 1; \
                    if (_qa_i != _qa_last) _qa_lc->slots[_qa_i] = _qa_lc->slots[_qa_last]; \
                    memset(&_qa_lc->slots[_qa_last], 0, sizeof(Slot)); \
                    _qa_lc->slots[_qa_last].eid = -2; \
                    _qa_lc->slots[_qa_last].loading_eid = -1; \
                    _qa_lc->slots[_qa_last].mio_slot = -1; \
                    _qa_lc->n--; \
                } \
                int _qa_desired = _qa_selected + _qa_m->c.topk; \
                if (_qa_desired < _qa_m->c.topk) _qa_desired = _qa_m->c.topk; \
                if (_qa_desired > _qa_m->c.n_experts) _qa_desired = _qa_m->c.n_experts; \
                if (_qa_desired < _qa_lc->n) _qa_desired = _qa_lc->n; \
                if (_qa_desired != _qa_lc->cap) { \
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
                pthread_mutex_unlock(&g_mx); \
            } \
        } \
    } \
} while (0)

#define rt_init(_engine, _layers, _experts) do { \
    (rt_init)((_engine), (_layers), (_experts)); \
    qwen_adaptive_adapter_init((void *)(m), (_layers), (_experts), \
                               (m)->c.topk, \
                               ((_layers) > 0 && (m)->cache) ? (m)->cache[0].cap : 0); \
} while (0)

#define rt_route(_layer, _row, _ids, _gates, _k) do { \
    (rt_route)((_layer), (_row), (_ids), (_gates), (_k)); \
    qwen_adaptive_adapter_observe((void *)(m), (_layer), (_row), (_ids), (_k)); \
    QWEN_ADAPTIVE_RECONCILE((m), (_layer)); \
} while (0)

#endif /* COLIBRI_QWEN_ADAPTIVE_RESIDENCY_ADAPTER_H */
