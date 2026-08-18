#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/* Engine-lifetime hook for #12. Process-local entries use the engine instance
 * as their strict model/config namespace. Retire them before the engine can be
 * destroyed/reallocated at the same address. */
#define coli_v4_engine_destroy coli_v4_engine_destroy_uncached
#include "deepseek_v4_internal.h"
#undef coli_v4_engine_destroy

#include "coli_v4_prefix_cache.h"

#define coli_v4_engine_destroy coli_v4_engine_destroy_uncached
#include "deepseek_v4.c"
#undef coli_v4_engine_destroy

void coli_v4_engine_destroy(ColiV4Engine *engine) {
    coli_v4_prefix_cache_forget_engine(engine);
    coli_v4_engine_destroy_uncached(engine);
}
