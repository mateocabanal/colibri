/* qwen_prefix_cache.h — Qwen prompt-cache policy wrapper.
 *
 * Keep the proven process-local implementation byte-for-byte in
 * qwen_prefix_cache_impl.h.  This wrapper owns serving policy so the cache can
 * be default-on without coupling the snapshot/copy logic to CLI mode parsing.
 */
#ifndef QWEN_PREFIX_CACHE_POLICY_WRAPPER_H
#define QWEN_PREFIX_CACHE_POLICY_WRAPPER_H

#include <stdlib.h>
#include <string.h>

#define QWEN_PREFIX_CACHE_DEFAULT_SERVE_MB 256

/* Pure policy helper used by unit tests.  Explicit values always win, including
 * "0" and "off"; the implementation parser maps non-positive/non-numeric
 * values to a zero-byte cache.  Only an absent value receives the serve-mode
 * default. */
static inline const char *qwen_prefix_cache_policy_value(const char *explicit_value,
                                                          int serving) {
    if (explicit_value && *explicit_value) return explicit_value;
    return serving ? "256" : NULL;
}

/* qwen_prefix_cache_impl.h historically reads QWEN_PREFIX_CACHE_MB through
 * getenv().  Interpose only while compiling that header so every existing call
 * site (budget planning, lazy init, store/restore) observes one policy.  The
 * rest of qwen_moe.c sees the real libc getenv after the include completes. */
static inline char *qpc_policy_getenv(const char *name) {
    char *value = getenv(name);
    if (strcmp(name, "QWEN_PREFIX_CACHE_MB") != 0) return value;
    char *serve = getenv("SERVE");
    return (char *)qwen_prefix_cache_policy_value(value,
                                                   serve && serve[0] == '1');
}

#define getenv qpc_policy_getenv
#include "qwen_prefix_cache_impl.h"
#undef getenv

#endif /* QWEN_PREFIX_CACHE_POLICY_WRAPPER_H */
