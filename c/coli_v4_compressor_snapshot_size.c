#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Compile the existing compressor snapshot unit first. It deliberately remaps
 * compressor symbols before including deepseek_v4_internal.h; pre-including
 * that header here would trip its include guard and hide the remapped function
 * declarations from Clang/MSVC-like strict C compilers. */
#include "deepseek_v4.c"
#include "coli_v4_prefix_cache.h"

size_t coli_v4_compressor_snapshot_bytes(
    const ColiV4CompressorSnapshot *snapshot) {
    if (!snapshot) return 0;
    if (snapshot->count > (SIZE_MAX - sizeof(*snapshot)) /
                              (2 * sizeof(float)))
        return SIZE_MAX;
    return sizeof(*snapshot) + snapshot->count * 2 * sizeof(float);
}
