#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * Lightweight default-on logical activation overlay for V4's hybrid block.
 *
 * The router knows logical token->expert multiplicity before
 * v4_moe_batch_union collapses those selections into one physical expert
 * lookup. Capture that multiplicity here and publish it through the shared
 * ExpertStore observer contract before the loader pool begins consuming the
 * physical expert records.
 *
 * Routing runs on the block thread while package-mode expert lookups can run on
 * several loader threads. The completed logical batch is therefore handed off
 * through one small mutex-protected pending buffer. The first physical lookup
 * for that layer publishes the entire batch while the other lanes wait on this
 * short metadata critical section; disk I/O starts only after publication and
 * remains fully parallel.
 *
 * This adapter owns no residency policy and no cache. Stores/runtimes that do
 * not implement observe_activations simply ignore the signal. Allocation or
 * accounting failure must never change inference output.
 */
#include "deepseek_v4_internal.h"
#include "expert_activation.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef COLI_V4_DISABLE_BF16_ROUTE
/* The BF16 route entry point is block-local in the amalgamation. Declare the
 * original symbol before the overlay macros rewrite the embedded unit. */
int coli_v4_route_bf16(float *weights, int *indices, const float *hidden,
                       const uint16_t *gate, const float *bias,
                       const int *forced_indices, int experts, int dimension,
                       int topk, float route_scale);
#endif

typedef struct {
    int prompt_tokens;
    int reused_tokens;
    int active;
} ColiV4ActivationRequest;

typedef struct {
    int valid;
    int layer;
    int64_t start_position;
    int next_item;
    int batch;
    ColiExpertPhase phase;
    uint64_t epoch;

    /* Three phase planes indexed by ColiExpertPhase (UNKNOWN/PREFILL/DECODE).
     * Scratch grows only if a future model/config has more experts than any
     * previous block on this routing thread. */
    uint64_t *counts;
    int *touched;
    size_t capacity;
    size_t touched_count;
    int disabled;
} ColiV4ActivationCall;

typedef struct {
    pthread_mutex_t mutex;
    ColiExpertActivationSample *samples;
    size_t capacity;
    size_t count;
    int layer;
    int active;
    uint64_t dropped_batches;
} ColiV4ActivationPending;

static _Thread_local ColiV4ActivationRequest g_v4_activation_request;
static _Thread_local ColiV4ActivationCall g_v4_activation_call;
static atomic_uint_fast64_t g_v4_activation_epoch = 1;
static ColiV4ActivationPending g_v4_activation_pending = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .layer = -1,
};

void coli_v4_activation_begin_request(int prompt_tokens, int reused_tokens) {
    g_v4_activation_request.prompt_tokens = prompt_tokens;
    g_v4_activation_request.reused_tokens = reused_tokens;
    g_v4_activation_request.active = prompt_tokens >= 0;
}

static void v4_activation_clear_touched(ColiV4ActivationCall *call) {
    if (!call || !call->counts || !call->touched) return;
    for (size_t i = 0; i < call->touched_count; i++) {
        int expert = call->touched[i];
        if (expert < 0 || (size_t)expert >= call->capacity) continue;
        call->counts[(size_t)COLI_EXPERT_PHASE_UNKNOWN * call->capacity +
                     (size_t)expert] = 0;
        call->counts[(size_t)COLI_EXPERT_PHASE_PREFILL * call->capacity +
                     (size_t)expert] = 0;
        call->counts[(size_t)COLI_EXPERT_PHASE_DECODE * call->capacity +
                     (size_t)expert] = 0;
    }
    call->touched_count = 0;
}

static int v4_activation_ensure_capacity(ColiV4ActivationCall *call,
                                         int experts) {
    if (!call || experts <= 0 || call->disabled) return 0;
    if ((size_t)experts <= call->capacity) return 1;
    if ((size_t)experts > SIZE_MAX / (3 * sizeof(uint64_t)) ||
        (size_t)experts > SIZE_MAX / sizeof(int)) {
        call->disabled = 1;
        return 0;
    }
    uint64_t *counts = calloc((size_t)experts * 3, sizeof(*counts));
    int *touched = malloc((size_t)experts * sizeof(*touched));
    if (!counts || !touched) {
        free(counts);
        free(touched);
        call->disabled = 1;
        return 0;
    }
    free(call->counts);
    free(call->touched);
    call->counts = counts;
    call->touched = touched;
    call->capacity = (size_t)experts;
    call->touched_count = 0;
    return 1;
}

static ColiExpertPhase v4_activation_phase(
    const ColiV4ActivationCall *call, int64_t position) {
    if (call->phase != COLI_EXPERT_PHASE_UNKNOWN) return call->phase;
    if (g_v4_activation_request.active) {
        return position < g_v4_activation_request.prompt_tokens
            ? COLI_EXPERT_PHASE_PREFILL : COLI_EXPERT_PHASE_DECODE;
    }
    /* Batch routing is ordinarily prefill. Without request context keep the
     * single-token case UNKNOWN rather than incorrectly calling it decode. */
    return call->batch > 1 ? COLI_EXPERT_PHASE_PREFILL
                           : COLI_EXPERT_PHASE_UNKNOWN;
}

static int v4_activation_pending_reserve_locked(
    ColiV4ActivationPending *pending, size_t wanted) {
    if (!pending || !wanted) return 0;
    if (wanted <= pending->capacity) return 1;
    if (wanted > SIZE_MAX / sizeof(*pending->samples)) return 0;
    ColiExpertActivationSample *samples = realloc(
        pending->samples, wanted * sizeof(*pending->samples));
    if (!samples) return 0;
    pending->samples = samples;
    pending->capacity = wanted;
    return 1;
}

static void v4_activation_publish_pending(ColiV4ActivationCall *call) {
    if (!call || !call->touched_count) return;

    size_t wanted = call->touched_count;
    if (wanted > SIZE_MAX / 3) {
        v4_activation_clear_touched(call);
        return;
    }
    wanted *= 3;

    ColiV4ActivationPending *pending = &g_v4_activation_pending;
    pthread_mutex_lock(&pending->mutex);

    /* Normal V4 block execution consumes every expert lookup for a layer before
     * routing the next layer. If that ordering is ever violated, preserve the
     * older batch rather than attaching new multiplicity to the wrong physical
     * lookups. The signal is advisory, so dropping the newer batch is safer. */
    if (pending->active) {
        pending->dropped_batches++;
        pthread_mutex_unlock(&pending->mutex);
        v4_activation_clear_touched(call);
        return;
    }

    if (!v4_activation_pending_reserve_locked(pending, wanted)) {
        pending->dropped_batches++;
        pthread_mutex_unlock(&pending->mutex);
        v4_activation_clear_touched(call);
        return;
    }

    size_t count = 0;
    for (size_t i = 0; i < call->touched_count; i++) {
        int expert = call->touched[i];
        if (expert < 0 || (size_t)expert >= call->capacity) continue;
        for (int phase = COLI_EXPERT_PHASE_UNKNOWN;
             phase <= COLI_EXPERT_PHASE_DECODE; phase++) {
            uint64_t multiplicity = call->counts[
                (size_t)phase * call->capacity + (size_t)expert];
            if (!multiplicity) continue;
            pending->samples[count++] = (ColiExpertActivationSample){
                .key = {call->layer, expert},
                .phase = (ColiExpertPhase)phase,
                .multiplicity = multiplicity,
                .epoch = call->epoch,
            };
        }
    }

    pending->count = count;
    pending->layer = call->layer;
    pending->active = count > 0;
    pthread_mutex_unlock(&pending->mutex);
    v4_activation_clear_touched(call);
}

static void v4_activation_context(
    const ColiDeepSeekV4LayerWeights *weights,
    int start_position, int batch, ColiExpertPhase phase) {
    ColiV4ActivationCall *call = &g_v4_activation_call;
    v4_activation_clear_touched(call);
    call->valid = weights && start_position >= 0 && batch > 0 && !call->disabled;
    call->layer = weights ? weights->plan.layer : -1;
    call->start_position = start_position;
    call->next_item = 0;
    call->batch = batch;
    call->phase = phase;
    call->epoch = atomic_fetch_add_explicit(
        &g_v4_activation_epoch, UINT64_C(1), memory_order_relaxed);
    if (!call->epoch)
        call->epoch = atomic_fetch_add_explicit(
            &g_v4_activation_epoch, UINT64_C(1), memory_order_relaxed);
}

static void v4_activation_selected(const int *indices, int topk, int experts) {
    ColiV4ActivationCall *call = &g_v4_activation_call;
    if (!call->valid || !indices || topk <= 0 || experts <= 0) return;

    int item = call->next_item++;
    int64_t position = call->start_position + item;
    ColiExpertPhase phase = v4_activation_phase(call, position);
    if (phase < COLI_EXPERT_PHASE_UNKNOWN || phase > COLI_EXPERT_PHASE_DECODE)
        phase = COLI_EXPERT_PHASE_UNKNOWN;

    if (v4_activation_ensure_capacity(call, experts)) {
        for (int rank = 0; rank < topk; rank++) {
            int expert = indices[rank];
            if (expert < 0 || expert >= experts) continue;
            size_t e = (size_t)expert;
            uint64_t sum = call->counts[e] +
                call->counts[call->capacity + e] +
                call->counts[2 * call->capacity + e];
            if (!sum && call->touched_count < call->capacity)
                call->touched[call->touched_count++] = expert;
            size_t offset = (size_t)phase * call->capacity + e;
            call->counts[offset] = coli_expert_activation_sat_add(
                call->counts[offset], UINT64_C(1));
        }
    }

    if (call->next_item >= call->batch) {
        call->valid = 0;
        v4_activation_publish_pending(call);
    }
}

static void v4_activation_flush(ColiExpertStore *store, int layer) {
    ColiV4ActivationPending *pending = &g_v4_activation_pending;
    pthread_mutex_lock(&pending->mutex);
    if (pending->active && pending->layer == layer) {
        /* Keep the pending lock while the store consumes this compact batch.
         * This guarantees every loader lane reaches its physical lookup only
         * after the complete pre-union signal has been published. Observer
         * callbacks are required to be non-failing and must not recurse into
         * V4 routing. */
        if (store && store->ops && store->ops->observe_activations)
            coli_expert_observe_activations(
                store, pending->samples, pending->count);
        pending->count = 0;
        pending->layer = -1;
        pending->active = 0;
    }
    pthread_mutex_unlock(&pending->mutex);
}

int coli_v4_activation_attention_token_ref(
    float *output, const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size) {
    v4_activation_context(weights, position, 1, COLI_EXPERT_PHASE_UNKNOWN);
    return coli_v4_attention_token_ref(
        output, weights, config, input, position, error, error_size);
}

int coli_v4_activation_attention_window_token_ref(
    float *output, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *input, int position,
    char *error, size_t error_size) {
    v4_activation_context(weights, position, 1, COLI_EXPERT_PHASE_UNKNOWN);
    return coli_v4_attention_window_token_ref(
        output, state, weights, config, input, position, error, error_size);
}

int coli_v4_activation_attention_window_batch_ref(
    float *outputs, ColiDeepSeekV4WindowAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int start_position, int batch, char *error, size_t error_size) {
    v4_activation_context(weights, start_position, batch,
                          COLI_EXPERT_PHASE_UNKNOWN);
    return coli_v4_attention_window_batch_ref(
        outputs, state, weights, config, inputs,
        start_position, batch, error, error_size);
}

int coli_v4_activation_route(
    float *weights, int *indices, const float *hidden,
    const float *gate, const float *bias,
    const int *forced_indices, int experts, int dimension,
    int topk, float route_scale) {
    int result = coli_v4_route(
        weights, indices, hidden, gate, bias, forced_indices,
        experts, dimension, topk, route_scale);
    if (!result) v4_activation_selected(indices, topk, experts);
    return result;
}

#ifndef COLI_V4_DISABLE_BF16_ROUTE
int coli_v4_activation_route_bf16(
    float *weights, int *indices, const float *hidden,
    const uint16_t *gate, const float *bias,
    const int *forced_indices, int experts, int dimension,
    int topk, float route_scale) {
    int result = coli_v4_route_bf16(
        weights, indices, hidden, gate, bias, forced_indices,
        experts, dimension, topk, route_scale);
    if (!result) v4_activation_selected(indices, topk, experts);
    return result;
}
#endif

int coli_v4_activation_expert_lookup(ColiExpertStore *store, ColiExpertKey key,
                                     ColiExpertView *view) {
    v4_activation_flush(store, key.layer);
    return coli_expert_lookup(store, key, view);
}

/* Compile the unchanged hybrid block through the lightweight wrappers. */
#define coli_v4_attention_token_ref coli_v4_activation_attention_token_ref
#define coli_v4_attention_window_token_ref coli_v4_activation_attention_window_token_ref
#define coli_v4_attention_window_batch_ref coli_v4_activation_attention_window_batch_ref
#define coli_v4_route coli_v4_activation_route
#ifndef COLI_V4_DISABLE_BF16_ROUTE
#define coli_v4_route_bf16 coli_v4_activation_route_bf16
#endif
#define coli_expert_lookup coli_v4_activation_expert_lookup
#include "deepseek_v4.c"
