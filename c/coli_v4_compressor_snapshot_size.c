#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "deepseek_v4_internal.h"
#include "coli_v4_prefix_cache.h"

/* Compile the existing compressor snapshot unit unchanged, then expose exact
 * payload accounting while its private snapshot layout is visible here. */
#include "deepseek_v4.c"

size_t coli_v4_compressor_snapshot_bytes(
    const ColiV4CompressorSnapshot *snapshot) {
    if (!snapshot) return 0;
    if (snapshot->count > (SIZE_MAX - sizeof(*snapshot)) /
                              (2 * sizeof(float)))
        return SIZE_MAX;
    return sizeof(*snapshot) + snapshot->count * 2 * sizeof(float);
}
