#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * #56 route-context overlay for COLI_V4_UNIT_BLOCK_HYBRID.
 *
 * Keep the large hybrid block implementation in deepseek_v4.c, but intercept
 * the attention boundaries that carry authoritative layer/token positions,
 * router calls that carry authoritative top-k rank/weight, and expert lookups
 * that materialize the physical lease. This preserves logical route identity
 * without changing execution order or the expert-store contract.
 *
 * Batched prefill is the important case: routing happens for every item before
 * the union of selected experts is loaded. One physical lookup can therefore
 * serve many logical `(token, rank)` selections. Route events are captured at
 * the router boundary, then all selections served by one lookup receive the
 * same lookup_id, lookup duration, and returned lease generation. The latter
 * is a stable join key to the existing physical v1 load/hit/evict lifecycle.
 */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef COLI_V4_GIT_SHA
#define COLI_V4_GIT_SHA "unknown"
#endif

#ifndef COLI_V4_DISABLE_BF16_ROUTE
/* The BF16 route entry point is intentionally block-local in the amalgamation
 * and is not declared by deepseek_v4_internal.h. Declare the original symbol
 * before the overlay macro rewrites the block unit's local prototype/calls. */
int coli_v4_route_bf16(float *weights, int *indices, const float *hidden,
                       const uint16_t *gate, const float *bias,
                       const int *forced_indices, int experts, int dimension,
                       int topk, float route_scale);
#endif

typedef struct {
    uint64_t seq;
    uint64_t request_id;
    int64_t token_position;
    int layer;
    int expert;
    int route_rank;
    float route_weight;
    ColiExpertPhase phase;
    uint64_t lookup_id;
    uint64_t lookup_ns;
    uint64_t lease_generation;
    uint32_t lookup_routes;
    int lookup_result;
} ColiV4RouteTraceEvent;

typedef struct {
    pthread_mutex_t mutex;
    ColiV4RouteTraceEvent *events;
    size_t count;
    size_t capacity;
    uint64_t dropped;
    uint64_t seq;
    uint64_t request_id;
    uint64_t lookup_seq;
    uint64_t correlation_misses;
    size_t pending_start;
    size_t pending_end;
    int pending_layer;
    int pending_active;
    char *path;
    int enabled;
} ColiV4RouteTraceState;

typedef struct {
    int valid;
    int layer;
    int64_t start_position;
    int next_item;
    int batch;
    ColiExpertPhase phase;
    size_t event_start;
} ColiV4RouteCallContext;

typedef struct {
    uint64_t id;
    size_t start;
    size_t end;
    uint32_t routes;
} ColiV4LookupTicket;

static ColiV4RouteTraceState g_v4_route_trace = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};
static pthread_once_t g_v4_route_trace_once = PTHREAD_ONCE_INIT;
static _Thread_local ColiV4RouteCallContext g_v4_route_call;

static uint64_t v4_route_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static char *v4_route_trace_copy(const char *text) {
    if (!text) return NULL;
    size_t n = strlen(text) + 1;
    char *copy = malloc(n);
    if (copy) memcpy(copy, text, n);
    return copy;
}

static char *v4_route_trace_sidecar(const char *expert_path) {
    static const char suffix[] = ".routes.jsonl";
    if (!expert_path) return NULL;
    size_t n = strlen(expert_path);
    if (n > SIZE_MAX - sizeof(suffix)) return NULL;
    char *path = malloc(n + sizeof(suffix));
    if (!path) return NULL;
    memcpy(path, expert_path, n);
    memcpy(path + n, suffix, sizeof(suffix));
    return path;
}

static const char *v4_route_phase_name(ColiExpertPhase phase) {
    switch (phase) {
    case COLI_EXPERT_PHASE_PREFILL: return "prefill";
    case COLI_EXPERT_PHASE_DECODE: return "decode";
    default: return "unknown";
    }
}

static void v4_route_trace_flush(void) {
    ColiV4RouteTraceState *state = &g_v4_route_trace;
    if (!state->enabled || !state->path) return;

    FILE *file = fopen(state->path, "w");
    if (!file) {
        fprintf(stderr,
                "v4_route_trace status=error path=%s reason=open\n",
                state->path);
        return;
    }

    pthread_mutex_lock(&state->mutex);
    uint64_t uncorrelated = 0;
    for (size_t i = 0; i < state->count; i++)
        if (!state->events[i].lookup_id) uncorrelated++;
    fprintf(file,
            "{\"schema\":\"colibri.v4.expert_trace.v2\","
            "\"build\":\"%s\",\"record_bytes\":0,"
            "\"events\":%llu,\"dropped\":%llu,"
            "\"source\":\"route_selected\","
            "\"physical_lookups\":%llu,"
            "\"correlation_misses\":%llu,"
            "\"uncorrelated_routes\":%llu}\n",
            COLI_V4_GIT_SHA,
            (unsigned long long)state->count,
            (unsigned long long)state->dropped,
            (unsigned long long)state->lookup_seq,
            (unsigned long long)state->correlation_misses,
            (unsigned long long)uncorrelated);
    for (size_t i = 0; i < state->count; i++) {
        const ColiV4RouteTraceEvent *event = &state->events[i];
        fprintf(file,
                "{\"seq\":%llu,\"event\":\"request\","
                "\"layer\":%d,\"expert\":%d,"
                "\"request_id\":%llu,\"token_position\":%lld,"
                "\"phase\":\"%s\",\"route_rank\":%d,"
                "\"route_weight\":",
                (unsigned long long)event->seq,
                event->layer, event->expert,
                (unsigned long long)event->request_id,
                (long long)event->token_position,
                v4_route_phase_name(event->phase),
                event->route_rank);
        if (isfinite(event->route_weight))
            fprintf(file, "%.9g", (double)event->route_weight);
        else
            fputs("null", file);
        fprintf(file,
                ",\"lookup_id\":%llu,\"lookup_ns\":%llu,"
                "\"lookup_routes\":%u,\"lookup_result\":%d,"
                "\"lease_generation\":%llu}\n",
                (unsigned long long)event->lookup_id,
                (unsigned long long)event->lookup_ns,
                event->lookup_routes,
                event->lookup_result,
                (unsigned long long)event->lease_generation);
    }
    size_t count = state->count;
    uint64_t dropped = state->dropped;
    uint64_t lookups = state->lookup_seq;
    uint64_t misses = state->correlation_misses;
    pthread_mutex_unlock(&state->mutex);
    fclose(file);

    fprintf(stderr,
            "v4_route_trace status=written path=%s events=%llu dropped=%llu "
            "lookups=%llu correlation_misses=%llu\n",
            state->path,
            (unsigned long long)count,
            (unsigned long long)dropped,
            (unsigned long long)lookups,
            (unsigned long long)misses);
}

static void v4_route_trace_init(void) {
    ColiV4RouteTraceState *state = &g_v4_route_trace;
    const char *explicit_path = getenv("V4_ROUTE_TRACE");
    const char *expert_path = getenv("V4_EXPERT_TRACE");
    if ((!explicit_path || !*explicit_path) &&
        (!expert_path || !*expert_path))
        return;

    if (explicit_path && *explicit_path)
        state->path = v4_route_trace_copy(explicit_path);
    else
        state->path = v4_route_trace_sidecar(expert_path);
    if (!state->path) return;

    if (expert_path && *expert_path && !strcmp(state->path, expert_path)) {
        fprintf(stderr,
                "v4_route_trace status=error path=%s reason=collides-with-expert-trace\n",
                state->path);
        free(state->path);
        state->path = NULL;
        return;
    }

    size_t capacity = 65536;
    const char *cap = getenv("V4_ROUTE_TRACE_CAP");
    if (!cap || !*cap) cap = getenv("V4_EXPERT_TRACE_CAP");
    if (cap && *cap) {
        unsigned long long parsed = strtoull(cap, NULL, 10);
        if (parsed >= 1024 && parsed <= 10000000)
            capacity = (size_t)parsed;
    }

    state->request_id = 1;
    const char *request_id = getenv("V4_ROUTE_REQUEST_ID");
    if (request_id && *request_id) {
        unsigned long long parsed = strtoull(request_id, NULL, 10);
        if (parsed) state->request_id = (uint64_t)parsed;
    }

    state->events = calloc(capacity, sizeof(*state->events));
    if (!state->events) {
        free(state->path);
        state->path = NULL;
        return;
    }
    state->capacity = capacity;
    state->enabled = 1;
    atexit(v4_route_trace_flush);
    fprintf(stderr,
            "v4_route_trace status=buffering path=%s capacity=%llu request_id=%llu\n",
            state->path,
            (unsigned long long)capacity,
            (unsigned long long)state->request_id);
}

static int v4_route_trace_enabled(void) {
    pthread_once(&g_v4_route_trace_once, v4_route_trace_init);
    return g_v4_route_trace.enabled;
}

static void v4_route_trace_context(
    const ColiDeepSeekV4LayerWeights *weights,
    int start_position, int batch, ColiExpertPhase phase) {
    if (!v4_route_trace_enabled()) return;
    ColiV4RouteTraceState *state = &g_v4_route_trace;
    pthread_mutex_lock(&state->mutex);
    if (state->pending_active) {
        /* A normal block consumes every lookup before the next layer routes.
         * Keep tracing if an error path violates that ordering, but surface the
         * lost correlation instead of silently attaching the next block. */
        state->correlation_misses++;
        state->pending_active = 0;
    }
    g_v4_route_call.event_start = state->count;
    pthread_mutex_unlock(&state->mutex);

    g_v4_route_call.valid = weights && batch > 0 && start_position >= 0;
    g_v4_route_call.layer = weights ? weights->plan.layer : -1;
    g_v4_route_call.start_position = start_position;
    g_v4_route_call.next_item = 0;
    g_v4_route_call.batch = batch;
    g_v4_route_call.phase = phase;
}

static void v4_route_trace_selected(const int *indices, const float *weights,
                                    int topk) {
    if (!v4_route_trace_enabled() || !g_v4_route_call.valid ||
        !indices || !weights || topk <= 0)
        return;

    int item = g_v4_route_call.next_item++;
    int complete = g_v4_route_call.next_item >= g_v4_route_call.batch;
    int64_t position = g_v4_route_call.start_position + item;

    ColiV4RouteTraceState *state = &g_v4_route_trace;
    pthread_mutex_lock(&state->mutex);
    for (int rank = 0; rank < topk; rank++) {
        if (state->count >= state->capacity) {
            state->dropped += (uint64_t)(topk - rank);
            break;
        }
        ColiV4RouteTraceEvent *event = &state->events[state->count++];
        event->seq = ++state->seq;
        event->request_id = state->request_id;
        event->token_position = position;
        event->layer = g_v4_route_call.layer;
        event->expert = indices[rank];
        event->route_rank = rank;
        event->route_weight = weights[rank];
        event->phase = g_v4_route_call.phase;
        event->lookup_result = -1;
    }
    if (complete) {
        state->pending_start = g_v4_route_call.event_start;
        state->pending_end = state->count;
        state->pending_layer = g_v4_route_call.layer;
        state->pending_active = state->pending_end > state->pending_start;
        g_v4_route_call.valid = 0;
    }
    pthread_mutex_unlock(&state->mutex);
}

static ColiV4LookupTicket v4_route_lookup_begin(ColiExpertKey key) {
    ColiV4LookupTicket ticket = {0};
    if (!v4_route_trace_enabled()) return ticket;
    ColiV4RouteTraceState *state = &g_v4_route_trace;
    pthread_mutex_lock(&state->mutex);
    ticket.id = ++state->lookup_seq;
    if (state->pending_active && state->pending_layer == key.layer) {
        ticket.start = state->pending_start;
        ticket.end = state->pending_end;
        for (size_t i = ticket.start; i < ticket.end; i++) {
            ColiV4RouteTraceEvent *event = &state->events[i];
            if (!event->lookup_id && event->layer == key.layer &&
                event->expert == key.expert) {
                event->lookup_id = ticket.id;
                ticket.routes++;
            }
        }
        if (!ticket.routes) state->correlation_misses++;

        int unassigned = 0;
        for (size_t i = state->pending_start; i < state->pending_end; i++)
            if (!state->events[i].lookup_id) {
                unassigned = 1;
                break;
            }
        if (!unassigned) state->pending_active = 0;
    } else {
        state->correlation_misses++;
    }
    pthread_mutex_unlock(&state->mutex);
    return ticket;
}

static void v4_route_lookup_end(const ColiV4LookupTicket *ticket,
                                int result, uint64_t elapsed_ns,
                                const ColiExpertView *view) {
    if (!ticket || !ticket->id || !ticket->routes) return;
    ColiV4RouteTraceState *state = &g_v4_route_trace;
    uint64_t generation = result == 0 && view ? view->lease_generation : 0;
    pthread_mutex_lock(&state->mutex);
    size_t end = ticket->end < state->count ? ticket->end : state->count;
    for (size_t i = ticket->start; i < end; i++) {
        ColiV4RouteTraceEvent *event = &state->events[i];
        if (event->lookup_id != ticket->id) continue;
        event->lookup_ns = elapsed_ns;
        event->lease_generation = generation;
        event->lookup_routes = ticket->routes;
        event->lookup_result = result;
    }
    pthread_mutex_unlock(&state->mutex);
}

/* These wrappers preserve exact runtime behavior. Side effects exist only when
 * detailed tracing is enabled; the disabled path adds one cached pthread_once
 * check at the wrapper boundaries and performs no clock calls or allocations. */
int coli_v4_trace_attention_token_ref(
    float *output, const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size) {
    v4_route_trace_context(weights, position, 1, COLI_EXPERT_PHASE_UNKNOWN);
    return coli_v4_attention_token_ref(
        output, weights, config, input, position, error, error_size);
}

int coli_v4_trace_attention_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size) {
    v4_route_trace_context(weights, position, 1, COLI_EXPERT_PHASE_UNKNOWN);
    return coli_v4_attention_window_token_ref(
        output, state, weights, config, input, position, error, error_size);
}

int coli_v4_trace_attention_window_batch_ref(
    float *outputs, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int start_position, int batch, char *error, size_t error_size) {
    v4_route_trace_context(
        weights, start_position, batch, COLI_EXPERT_PHASE_UNKNOWN);
    return coli_v4_attention_window_batch_ref(
        outputs, state, weights, config, inputs,
        start_position, batch, error, error_size);
}

int coli_v4_trace_route(
    float *weights, int *indices, const float *hidden,
    const float *gate, const float *bias,
    const int *forced_indices, int experts, int dimension,
    int topk, float route_scale) {
    int result = coli_v4_route(
        weights, indices, hidden, gate, bias, forced_indices,
        experts, dimension, topk, route_scale);
    if (!result) v4_route_trace_selected(indices, weights, topk);
    return result;
}

#ifndef COLI_V4_DISABLE_BF16_ROUTE
int coli_v4_trace_route_bf16(
    float *weights, int *indices, const float *hidden,
    const uint16_t *gate, const float *bias,
    const int *forced_indices, int experts, int dimension,
    int topk, float route_scale) {
    int result = coli_v4_route_bf16(
        weights, indices, hidden, gate, bias, forced_indices,
        experts, dimension, topk, route_scale);
    if (!result) v4_route_trace_selected(indices, weights, topk);
    return result;
}
#endif

int coli_v4_trace_expert_lookup(ColiExpertStore *store, ColiExpertKey key,
                                ColiExpertView *view) {
    if (!v4_route_trace_enabled())
        return coli_expert_lookup(store, key, view);
    ColiV4LookupTicket ticket = v4_route_lookup_begin(key);
    uint64_t began = v4_route_now_ns();
    int result = coli_expert_lookup(store, key, view);
    uint64_t elapsed = v4_route_now_ns() - began;
    v4_route_lookup_end(&ticket, result, elapsed, view);
    return result;
}

/* Compile the existing block unit through the wrappers above. Because the
 * internal header is already include-guarded, these macros rewrite only calls
 * inside the block unit. Original implementations remain external and are
 * called by the wrappers defined before these macros. */
#define coli_v4_attention_token_ref coli_v4_trace_attention_token_ref
#define coli_v4_attention_window_token_ref coli_v4_trace_attention_window_token_ref
#define coli_v4_attention_window_batch_ref coli_v4_trace_attention_window_batch_ref
#define coli_v4_route coli_v4_trace_route
#ifndef COLI_V4_DISABLE_BF16_ROUTE
#define coli_v4_route_bf16 coli_v4_trace_route_bf16
#endif
#define coli_expert_lookup coli_v4_trace_expert_lookup
#include "deepseek_v4.c"
