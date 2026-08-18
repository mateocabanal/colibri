#ifndef COLIBRI_V4_PREFIX_DISK_H
#define COLIBRI_V4_PREFIX_DISK_H

#include <limits.h>
#include <stdint.h>

#include "deepseek_v4.h"

/* The persistent adapter is deliberately TU-local. The shared SSD framing in
 * prefix_cache_disk.h is header-only, so keeping the V4 glue inline avoids a
 * second build-graph path and makes the ordinary V4 prefix-cache helper targets
 * exercise exactly the production adapter. publish/restore wrappers refresh the
 * engine namespace in their own TU before use, so runtime/generation overlays do
 * not depend on cross-TU registry state. */
#define coli_v4_prefix_disk_register_engine \
    static inline coli_v4_prefix_disk_register_engine
#define coli_v4_prefix_disk_forget_engine \
    static inline coli_v4_prefix_disk_forget_engine
#define coli_v4_prefix_disk_publish_session \
    static inline coli_v4_prefix_disk_publish_session_impl
#define coli_v4_prefix_disk_restore \
    static inline coli_v4_prefix_disk_restore_impl
#include "coli_v4_prefix_disk.c"
#undef coli_v4_prefix_disk_restore
#undef coli_v4_prefix_disk_publish_session
#undef coli_v4_prefix_disk_forget_engine
#undef coli_v4_prefix_disk_register_engine

static inline void coli_v4_prefix_disk_publish_session(ColiV4Session *session) {
    if (session && session->engine)
        coli_v4_prefix_disk_register_engine(session->engine);
    coli_v4_prefix_disk_publish_session_impl(session);
}

static inline int coli_v4_prefix_disk_restore(ColiV4Session *session,
                                               const int *prompt_ids,
                                               int prompt_tokens) {
    if (session && session->engine)
        coli_v4_prefix_disk_register_engine(session->engine);
    return coli_v4_prefix_disk_restore_impl(session, prompt_ids, prompt_tokens);
}

#endif /* COLIBRI_V4_PREFIX_DISK_H */
