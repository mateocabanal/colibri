#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Compile the existing compressor snapshot unit first. It deliberately remaps
 * compressor symbols before including deepseek_v4_internal.h; pre-including
 * that header here would trip its include guard and hide the remapped function
 * declarations from Clang/MSVC-like strict C compilers. */
#include "deepseek_v4.c"
#include "coli_v4_prefix_cache.h"

#include <stdarg.h>

static size_t compressor_count_bytes(size_t count) {
    if (count > (SIZE_MAX - sizeof(ColiV4CompressorSnapshot)) /
                    (2 * sizeof(float)))
        return SIZE_MAX;
    return sizeof(ColiV4CompressorSnapshot) + count * 2 * sizeof(float);
}

size_t coli_v4_compressor_snapshot_bytes(
    const ColiV4CompressorSnapshot *snapshot) {
    return snapshot ? compressor_count_bytes(snapshot->count) : 0;
}

size_t coli_v4_compressor_state_snapshot_bytes(
    const ColiDeepSeekV4CompressorState *state) {
    if (!state) return 0;
    if (state->state_rows < 0 || state->projection_dim < 0) return SIZE_MAX;
    size_t rows = (size_t)state->state_rows;
    size_t projection = (size_t)state->projection_dim;
    if (projection && rows > SIZE_MAX / projection) return SIZE_MAX;
    return compressor_count_bytes(rows * projection);
}

/* Persistent payload: u64 count, then kv_state[count], score_state[count]. */
size_t coli_v4_compressor_snapshot_wire_bytes(
    const ColiV4CompressorSnapshot *snapshot) {
    if (!snapshot) return 0;
    if (snapshot->count > (SIZE_MAX - sizeof(uint64_t)) /
                              (2 * sizeof(float)))
        return SIZE_MAX;
    return sizeof(uint64_t) + snapshot->count * 2 * sizeof(float);
}

int coli_v4_compressor_snapshot_wire_emit(
    const ColiV4CompressorSnapshot *snapshot,
    ColiV4PrefixWireSink sink, void *user_data) {
    if (!snapshot || !sink ||
        (snapshot->count && (!snapshot->kv_state || !snapshot->score_state)))
        return -1;
    if (coli_v4_compressor_snapshot_wire_bytes(snapshot) == SIZE_MAX)
        return -1;
    uint64_t count = (uint64_t)snapshot->count;
    if ((size_t)count != snapshot->count) return -1;
    if (sink(user_data, &count, sizeof(count))) return -1;
    size_t values = snapshot->count * sizeof(float);
    if (values && sink(user_data, snapshot->kv_state, values)) return -1;
    if (values && sink(user_data, snapshot->score_state, values)) return -1;
    return 0;
}

int coli_v4_compressor_snapshot_wire_write(
    const ColiV4CompressorSnapshot *snapshot,
    unsigned char *output, size_t output_size, size_t *written) {
    if (written) *written = 0;
    if (!snapshot || !output || !written ||
        (snapshot->count && (!snapshot->kv_state || !snapshot->score_state)))
        return -1;
    size_t need = coli_v4_compressor_snapshot_wire_bytes(snapshot);
    if (need == SIZE_MAX || need > output_size) return -1;
    uint64_t count = (uint64_t)snapshot->count;
    if ((size_t)count != snapshot->count) return -1;
    memcpy(output, &count, sizeof(count));
    size_t values = snapshot->count * sizeof(float);
    if (values) {
        memcpy(output + sizeof(count), snapshot->kv_state, values);
        memcpy(output + sizeof(count) + values, snapshot->score_state, values);
    }
    *written = need;
    return 0;
}

int coli_v4_compressor_snapshot_wire_read(
    ColiV4CompressorSnapshot **output,
    const unsigned char *input, size_t input_size, size_t *consumed) {
    if (consumed) *consumed = 0;
    if (!output || !input || !consumed || input_size < sizeof(uint64_t))
        return -1;
    *output = NULL;
    uint64_t raw_count = 0;
    memcpy(&raw_count, input, sizeof(raw_count));
    size_t count = (size_t)raw_count;
    if ((uint64_t)count != raw_count) return -1;
    if (count > (SIZE_MAX - sizeof(uint64_t)) / (2 * sizeof(float)))
        return -1;
    size_t need = sizeof(uint64_t) + count * 2 * sizeof(float);
    if (need > input_size) return -1;

    ColiV4CompressorSnapshot *snapshot = calloc(1, sizeof(*snapshot));
    if (!snapshot) return -1;
    snapshot->count = count;
    if (count) {
        size_t values = count * sizeof(float);
        snapshot->kv_state = malloc(values);
        snapshot->score_state = malloc(values);
        if (!snapshot->kv_state || !snapshot->score_state) {
            coli_v4_compressor_snapshot_destroy(snapshot);
            return -1;
        }
        memcpy(snapshot->kv_state, input + sizeof(uint64_t), values);
        memcpy(snapshot->score_state,
               input + sizeof(uint64_t) + values, values);
    }
    *output = snapshot;
    *consumed = need;
    return 0;
}

static int compressor_hydrate_error(char *error, size_t error_size,
                                    const char *format, ...) {
    if (error && error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, error_size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

/* Build only the recurrent compressor state needed by a cached prefix. The
 * dense layer payload is deliberately absent here: target_batch will load that
 * layer exactly once for the fresh tail, and prepare_compressed_state() will
 * bind the real weights through coli_v4_compressor_bind_weights() before the
 * first compressor step. */
int coli_v4_compressor_snapshot_restore_unbound(
    ColiDeepSeekV4CompressorState **output,
    const ColiV4CompressorSnapshot *snapshot,
    const ColiDeepSeekV4Config *config,
    int layer, int compression_ratio,
    const ColiDeepSeekV4CompressorOptions *options,
    char *error, size_t error_size) {
    if (!output || !snapshot || !config || !options || !options->prefix ||
        !options->prefix[0] || layer < 0 || compression_ratio < 1 ||
        options->head_dimension <= 0)
        return compressor_hydrate_error(
            error, error_size, "invalid unbound compressor snapshot geometry");
    if (strlen(options->prefix) >=
        sizeof(((ColiDeepSeekV4CompressorState *)0)->prefix))
        return compressor_hydrate_error(
            error, error_size, "compressor prefix is too long");

    *output = NULL;
    ColiDeepSeekV4CompressorState *state = calloc(1, sizeof(*state));
    if (!state)
        return compressor_hydrate_error(
            error, error_size, "out of memory hydrating compressor snapshot");

    state->weights = NULL;
    state->config = config;
    state->ratio = compression_ratio;
    state->layer = layer;
    state->hidden = config->hidden_size;
    state->head_dim = options->head_dimension;
    state->rotate_fp4 = options->rotate_fp4 != 0;
    memcpy(state->prefix, options->prefix, strlen(options->prefix) + 1);

    int overlap = compression_ratio == 4;
    state->projection_dim = (1 + overlap) * state->head_dim;
    state->state_rows = (1 + overlap) * compression_ratio;
    state->rope_dim = config->qk_rope_head_dim;
    if (state->projection_dim <= 0 || state->state_rows <= 0 ||
        (size_t)state->state_rows > SIZE_MAX / (size_t)state->projection_dim) {
        free(state);
        return compressor_hydrate_error(
            error, error_size, "invalid compressor snapshot dimensions");
    }

    size_t count = (size_t)state->state_rows * (size_t)state->projection_dim;
    if (snapshot->count != count || count > SIZE_MAX / sizeof(float)) {
        free(state);
        return compressor_hydrate_error(
            error, error_size, "compressor snapshot geometry mismatch");
    }
    state->kv_state = malloc(count * sizeof(*state->kv_state));
    state->score_state = malloc(count * sizeof(*state->score_state));
    if (!state->kv_state || !state->score_state) {
        free(state->score_state);
        free(state->kv_state);
        free(state);
        return compressor_hydrate_error(
            error, error_size, "out of memory allocating compressor snapshot state");
    }
    memcpy(state->kv_state, snapshot->kv_state,
           count * sizeof(*state->kv_state));
    memcpy(state->score_state, snapshot->score_state,
           count * sizeof(*state->score_state));
    *output = state;
    return 0;
}
