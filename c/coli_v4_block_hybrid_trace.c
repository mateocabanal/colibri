#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * #56 route-context overlay for COLI_V4_UNIT_BLOCK_HYBRID.
 *
 * Keep the large hybrid block implementation in deepseek_v4.c, but intercept
 * the three attention entry points that already carry authoritative layer /
 * token-position identity and the router calls that return authoritative top-k
 * rank/weight. This lets detailed tracing capture every logical routed expert
 * selection without changing the physical expert-load schedule.
 *
 * In particular, batched prefill routes all items before walking the union of
 * selected experts. Tracing at lookup() would lose the many-to-one mapping;
 * tracing at the router boundary preserves one event per (token, layer, rank)
 * even when a single physical lease serves several positions.
 */
#include "deepseek_v4_internal.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
} ColiV4RouteTraceEvent;

typedef struct {
    pthread_mutex_t mutex;
    ColiV4RouteTraceEvent *events;
    size_t count;
    size_t capacity;
    uint64_t dropped;
    uint64_t seq;
    uint64_t request_id;
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
} ColiV4RouteCallContext;

static ColiV4RouteTraceState g_v4_route_trace = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};
static pthread_once_t g_v4_route_trace_once = PTHREAD_ONCE_INIT;
static _Thread_local ColiV4RouteCallContext g_v4_route_call;

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
    fprintf(file,
            "{\"schema\":\"colibri.v4.expert_trace.v2\","
            "\"build\":\"%s\",\"record_bytes\":0,"
            "\"events\":%llu,\"dropped\":%llu,"
            "\"source\":\"route_selected\"}\n",
            COLI_V4_GIT_SHA,
            (unsigned long long)state->count,
            (unsigned long long)state->dropped);
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
        fputs("}\n", file);
    }
    size_t count = state->count;
    uint64_t dropped = state->dropped;
    pthread_mutex_unlock(&state->mutex);
    fclose(file);

    fprintf(stderr,
            "v4_route_trace status=written path=%s events=%llu dropped=%llu\n",
            state->path,
            (unsigned long long)count,
            (unsigned long long)dropped);
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
    int64_t position = g_v4_route_call.start_position + item;
    if (g_v4_route_call.next_item >= g_v4_route_call.batch)
        g_v4_route_call.valid = 0;

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
    }
    pthread_mutex_unlock(&state->mutex);
}

/* These wrappers preserve the exact runtime behavior of the original calls.
 * The only side effect is updating/consuming tracing context when detailed
 * tracing is enabled. */
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

/* Compile the existing block unit through the wrappers above. Because the
 * internal header is already include-guarded, these macros rewrite only calls
 * and the block-local BF16 route prototype inside deepseek_v4.c; the original
 * attention/router implementations remain external and are called by wrappers. */
#define coli_v4_attention_token_ref coli_v4_trace_attention_token_ref
#define coli_v4_attention_window_token_ref coli_v4_trace_attention_window_token_ref
#define coli_v4_attention_window_batch_ref coli_v4_trace_attention_window_batch_ref
#define coli_v4_route coli_v4_trace_route
#ifndef COLI_V4_DISABLE_BF16_ROUTE
#define coli_v4_route_bf16 coli_v4_trace_route_bf16
#endif
#include "deepseek_v4.c"
