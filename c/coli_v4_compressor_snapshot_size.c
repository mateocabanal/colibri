#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Compile the existing compressor snapshot unit first. It deliberately remaps
 * compressor symbols before including deepseek_v4_internal.h; pre-including
 * that header here would trip its include guard and hide the remapped function
 * declarations from Clang/MSVC-like strict C compilers. */
#include "deepseek_v4.c"
#include "coli_v4_prefix_cache.h"

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
