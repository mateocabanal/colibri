#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Engine-lifetime hooks for the process-local and persistent #80 prefix tiers.
 * Remap the existing public open/destroy functions, then layer only cache
 * registration/retirement around current-main runtime behavior. */
#define coli_v4_engine_open coli_v4_engine_open_uncached
#define coli_v4_engine_destroy coli_v4_engine_destroy_uncached
#include "deepseek_v4_internal.h"
#undef coli_v4_engine_destroy
#undef coli_v4_engine_open

#include "coli_v4_prefix_cache.h"
#include "coli_v4_prefix_disk.h"

#define coli_v4_engine_open coli_v4_engine_open_uncached
#define coli_v4_engine_destroy coli_v4_engine_destroy_uncached
#include "deepseek_v4.c"
#undef coli_v4_engine_destroy
#undef coli_v4_engine_open

int coli_v4_engine_open(ColiV4Engine **engine,
                        const ColiV4EngineOpenOptions *options,
                        char *error, size_t error_size) {
    int result = coli_v4_engine_open_uncached(engine, options, error, error_size);
    if (!result && engine && *engine)
        coli_v4_prefix_disk_register_engine(*engine);
    return result;
}

void coli_v4_engine_destroy(ColiV4Engine *engine) {
    coli_v4_prefix_cache_forget_engine(engine);
    coli_v4_prefix_disk_forget_engine(engine);
    coli_v4_engine_destroy_uncached(engine);
}
