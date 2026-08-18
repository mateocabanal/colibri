#ifndef COLIBRI_V4_PREFIX_DISK_H
#define COLIBRI_V4_PREFIX_DISK_H

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#include "deepseek_v4.h"
#include "prefix_cache_disk.h"

static inline int coli_v4_prefix_ascii_ieq(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++, cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return *a == 0 && *b == 0;
}

/* V4 v1 publishes by borrowing the immutable process-RAM end-of-prefill entry.
 * Explicit SSD-only mode has no such entry, so fail closed rather than writing
 * post-decode state. Default `auto` (RAM + SSD) is the fully supported path. */
static inline int coli_v4_prefix_disk_mode_allows_ssd(void) {
    const char *mode = getenv("COLI_PREFIX_CACHE");
    if (mode && *mode && coli_v4_prefix_ascii_ieq(mode, "ssd")) return 0;
    return coli_prefix_disk_mode_allows_ssd();
}

/* The persistent adapter is TU-local. Namespace adapter-private helpers while
 * embedding the implementation so they cannot collide with the shared #92
 * header's static helpers in the same translation unit. */
#define ascii_ieq coli_v4_disk_ascii_ieq
#define coli_prefix_disk_mode_allows_ssd coli_v4_prefix_disk_mode_allows_ssd
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
#undef coli_prefix_disk_mode_allows_ssd
#undef ascii_ieq

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
