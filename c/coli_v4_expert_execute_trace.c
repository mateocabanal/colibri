#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * #56 expert execution lifecycle trace.
 *
 * This object is linked only when Makefile.deepseek-v4 is built with
 * V4_TRACE_EXEC=1. The block unit is then compiled so calls to
 * coli_v4_expert_forward_ref() land here; this wrapper calls the unchanged
 * expert implementation and records execution identity/duration in memory.
 * Default builds do not reference this wrapper at all, so disabled execution
 * tracing adds no function call, clock, pthread_once, allocation, or branch to
 * the normal expert-compute path.
 *
 * Logical request/token/rank identity remains in the schema-v2 route sidecar.
 * This stream deliberately records the physical execution identity carried by
 * ColiExpertView: (layer, expert, lease_generation). Offline tooling joins the
 * two streams and checks that every successful logical selection produces one
 * expert execution even when batch-union prefill shares one physical lookup
 * across several selections.
 *
 * When V4_PROFILE=1, profiled_expert_load_finish() updates the canonical
 * owner-thread COLI_V4_PROF_EXPERT_LOADER_WAIT bucket immediately before the
 * corresponding expert-forward call. Sampling that cumulative bucket here and
 * subtracting the previous execution's sample therefore attributes the exposed
 * owner wait that occurred before this execution without touching the loader
 * implementation or relabeling worker lookup_ns as wall-clock stall. In a
 * batch-union fanout the first execution after a physical lookup receives that
 * lookup's owner wait; later routes sharing the same lease normally receive 0.
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

typedef struct {
    uint64_t seq;
    uint64_t generation;
    uint64_t resident_bytes;
    uint64_t execute_ns;
    uint64_t owner_wait_ns;
    int layer;
    int expert;
    int result;
    int gate_format;
    int down_format;
    int up_format;
    int owner_wait_measured;
    float route_weight;
} ColiV4ExecuteTraceEvent;

typedef struct {
    pthread_mutex_t mutex;
    ColiV4ExecuteTraceEvent *events;
    size_t count;
    size_t capacity;
    uint64_t dropped;
    uint64_t seq;
    uint64_t total_execute_ns;
    uint64_t total_owner_wait_ns;
    uint64_t owner_wait_measured_events;
    char *path;
    int enabled;
} ColiV4ExecuteTraceState;

static ColiV4ExecuteTraceState g_v4_execute_trace = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};
static pthread_once_t g_v4_execute_trace_once = PTHREAD_ONCE_INIT;

/* Expert forward runs on the owner thread in the current V4 pipeline. Keep the
 * previous cumulative wait sample thread-local so a future multi-session owner
 * scheduler cannot make independent execution streams subtract each other. */
static _Thread_local uint64_t g_v4_last_owner_wait_ns;
static _Thread_local int g_v4_owner_wait_sample_valid;

static char *trace_copy(const char *text) {
    if (!text) return NULL;
    size_t n = strlen(text) + 1;
    char *copy = malloc(n);
    if (copy) memcpy(copy, text, n);
    return copy;
}

static char *trace_sidecar(const char *base) {
    static const char suffix[] = ".exec.jsonl";
    if (!base) return NULL;
    size_t n = strlen(base);
    if (n > SIZE_MAX - sizeof(suffix)) return NULL;
    char *path = malloc(n + sizeof(suffix));
    if (!path) return NULL;
    memcpy(path, base, n);
    memcpy(path + n, suffix, sizeof(suffix));
    return path;
}

static uint64_t add_bytes(uint64_t total, size_t value) {
    uint64_t add = (uint64_t)value;
    return UINT64_MAX - total < add ? UINT64_MAX : total + add;
}

static uint64_t expert_resident_bytes(const ColiExpertView *expert) {
    if (!expert) return 0;
    uint64_t bytes = 0;
    bytes = add_bytes(bytes, expert->gate.data_bytes);
    bytes = add_bytes(bytes, expert->gate.scale_bytes);
    bytes = add_bytes(bytes, expert->down.data_bytes);
    bytes = add_bytes(bytes, expert->down.scale_bytes);
    bytes = add_bytes(bytes, expert->up.data_bytes);
    bytes = add_bytes(bytes, expert->up.scale_bytes);
    return bytes;
}

static void owner_wait_sample(uint64_t *wait_ns, int *measured) {
    *wait_ns = 0;
    *measured = 0;
    if (!g_coli_v4_profile_on) {
        g_v4_last_owner_wait_ns = 0;
        g_v4_owner_wait_sample_valid = 0;
        return;
    }

    ColiV4Profile profile = coli_v4_profile_get();
    uint64_t current = profile.ns[COLI_V4_PROF_EXPERT_LOADER_WAIT];
    if (!g_v4_owner_wait_sample_valid || current < g_v4_last_owner_wait_ns)
        *wait_ns = current;
    else
        *wait_ns = current - g_v4_last_owner_wait_ns;
    g_v4_last_owner_wait_ns = current;
    g_v4_owner_wait_sample_valid = 1;
    *measured = 1;
}

static void execute_trace_flush(void) {
    ColiV4ExecuteTraceState *state = &g_v4_execute_trace;
    if (!state->enabled || !state->path) return;

    FILE *file = fopen(state->path, "w");
    if (!file) {
        fprintf(stderr,
                "v4_execute_trace status=error path=%s reason=open\n",
                state->path);
        return;
    }

    pthread_mutex_lock(&state->mutex);
    fprintf(file,
            "{\"schema\":\"colibri.v4.expert_execute_trace.v1\","
            "\"build\":\"%s\",\"source\":\"expert_execute\","
            "\"events\":%llu,\"dropped\":%llu,"
            "\"total_execute_ns\":%llu,"
            "\"owner_wait_measured_events\":%llu,"
            "\"total_owner_wait_ns\":%llu}\n",
            COLI_V4_GIT_SHA,
            (unsigned long long)state->count,
            (unsigned long long)state->dropped,
            (unsigned long long)state->total_execute_ns,
            (unsigned long long)state->owner_wait_measured_events,
            (unsigned long long)state->total_owner_wait_ns);
    for (size_t i = 0; i < state->count; i++) {
        const ColiV4ExecuteTraceEvent *event = &state->events[i];
        fprintf(file,
                "{\"seq\":%llu,\"event\":\"execute\","
                "\"layer\":%d,\"expert\":%d,"
                "\"generation\":%llu,\"resident_bytes\":%llu,"
                "\"route_weight\":",
                (unsigned long long)event->seq,
                event->layer, event->expert,
                (unsigned long long)event->generation,
                (unsigned long long)event->resident_bytes);
        if (isfinite(event->route_weight))
            fprintf(file, "%.9g", (double)event->route_weight);
        else
            fputs("null", file);
        fprintf(file,
                ",\"execute_ns\":%llu,\"result\":%d,"
                "\"owner_wait_measured\":%s,\"owner_wait_ns\":%llu,"
                "\"gate_format\":%d,\"down_format\":%d,"
                "\"up_format\":%d}\n",
                (unsigned long long)event->execute_ns,
                event->result,
                event->owner_wait_measured ? "true" : "false",
                (unsigned long long)event->owner_wait_ns,
                event->gate_format, event->down_format, event->up_format);
    }
    size_t count = state->count;
    uint64_t dropped = state->dropped;
    uint64_t total = state->total_execute_ns;
    uint64_t wait_total = state->total_owner_wait_ns;
    uint64_t measured = state->owner_wait_measured_events;
    pthread_mutex_unlock(&state->mutex);
    fclose(file);

    fprintf(stderr,
            "v4_execute_trace status=written path=%s events=%llu dropped=%llu "
            "execute_ms=%.3f owner_wait_measured=%llu owner_wait_ms=%.3f\n",
            state->path,
            (unsigned long long)count,
            (unsigned long long)dropped,
            total / 1e6,
            (unsigned long long)measured,
            wait_total / 1e6);
}

static void execute_trace_init(void) {
    ColiV4ExecuteTraceState *state = &g_v4_execute_trace;
    const char *explicit_path = getenv("V4_EXEC_TRACE");
    const char *route_path = getenv("V4_ROUTE_TRACE");
    const char *expert_path = getenv("V4_EXPERT_TRACE");
    if ((!explicit_path || !*explicit_path) &&
        (!route_path || !*route_path) &&
        (!expert_path || !*expert_path))
        return;

    if (explicit_path && *explicit_path)
        state->path = trace_copy(explicit_path);
    else if (route_path && *route_path)
        state->path = trace_sidecar(route_path);
    else
        state->path = trace_sidecar(expert_path);
    if (!state->path) return;

    if ((route_path && *route_path && !strcmp(state->path, route_path)) ||
        (expert_path && *expert_path && !strcmp(state->path, expert_path))) {
        fprintf(stderr,
                "v4_execute_trace status=error path=%s reason=trace-path-collision\n",
                state->path);
        free(state->path);
        state->path = NULL;
        return;
    }

    size_t capacity = 65536;
    const char *cap = getenv("V4_EXEC_TRACE_CAP");
    if (!cap || !*cap) cap = getenv("V4_ROUTE_TRACE_CAP");
    if (!cap || !*cap) cap = getenv("V4_EXPERT_TRACE_CAP");
    if (cap && *cap) {
        unsigned long long parsed = strtoull(cap, NULL, 10);
        if (parsed >= 1024 && parsed <= 10000000)
            capacity = (size_t)parsed;
    }

    state->events = calloc(capacity, sizeof(*state->events));
    if (!state->events) {
        free(state->path);
        state->path = NULL;
        return;
    }
    state->capacity = capacity;
    state->enabled = 1;
    atexit(execute_trace_flush);
    fprintf(stderr,
            "v4_execute_trace status=buffering path=%s capacity=%llu "
            "owner_wait=%s\n",
            state->path, (unsigned long long)capacity,
            g_coli_v4_profile_on ? "profiled" : "unmeasured");
}

static int execute_trace_enabled(void) {
    pthread_once(&g_v4_execute_trace_once, execute_trace_init);
    return g_v4_execute_trace.enabled;
}

int coli_v4_trace_expert_forward_ref(float *output,
                                     const ColiExpertView *expert,
                                     const float *input, float route_weight,
                                     float swiglu_limit) {
    if (!execute_trace_enabled())
        return coli_v4_expert_forward_ref(
            output, expert, input, route_weight, swiglu_limit);

    uint64_t owner_wait_ns = 0;
    int owner_wait_measured = 0;
    owner_wait_sample(&owner_wait_ns, &owner_wait_measured);

    uint64_t began = coli_v4_profile_now();
    int result = coli_v4_expert_forward_ref(
        output, expert, input, route_weight, swiglu_limit);
    uint64_t elapsed = coli_v4_profile_now() - began;

    ColiV4ExecuteTraceState *state = &g_v4_execute_trace;
    pthread_mutex_lock(&state->mutex);
    if (UINT64_MAX - state->total_execute_ns < elapsed)
        state->total_execute_ns = UINT64_MAX;
    else
        state->total_execute_ns += elapsed;
    if (owner_wait_measured) {
        state->owner_wait_measured_events++;
        if (UINT64_MAX - state->total_owner_wait_ns < owner_wait_ns)
            state->total_owner_wait_ns = UINT64_MAX;
        else
            state->total_owner_wait_ns += owner_wait_ns;
    }
    if (state->count >= state->capacity) {
        state->dropped++;
        pthread_mutex_unlock(&state->mutex);
        return result;
    }

    ColiV4ExecuteTraceEvent *event = &state->events[state->count++];
    event->seq = ++state->seq;
    event->generation = expert ? expert->lease_generation : 0;
    event->resident_bytes = expert_resident_bytes(expert);
    event->layer = expert ? expert->key.layer : -1;
    event->expert = expert ? expert->key.expert : -1;
    event->result = result;
    event->gate_format = expert ? (int)expert->gate.format : -1;
    event->down_format = expert ? (int)expert->down.format : -1;
    event->up_format = expert ? (int)expert->up.format : -1;
    event->route_weight = route_weight;
    event->execute_ns = elapsed;
    event->owner_wait_measured = owner_wait_measured;
    event->owner_wait_ns = owner_wait_ns;
    pthread_mutex_unlock(&state->mutex);
    return result;
}
