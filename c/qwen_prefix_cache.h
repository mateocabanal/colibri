/* qwen_prefix_cache.h — common policy + SSD tier around the proven Qwen RAM cache.
 *
 * qwen_prefix_cache_impl.h is the byte-for-byte implementation currently in
 * main.  This wrapper only owns common policy and persistence, so the validated
 * hybrid snapshot/copy geometry is not forked.
 */
#ifndef QWEN_PREFIX_CACHE_GLOBAL_WRAPPER_H
#define QWEN_PREFIX_CACHE_GLOBAL_WRAPPER_H

#include "prefix_cache_disk.h"

/* Rename only the public policy/store/restore entry points while compiling the
 * mainline implementation. Internal capture/restore helpers remain available
 * for the persistent adapter and therefore share the exact same geometry. */
#define qwen_prefix_cache_budget_from_env qpc_ram_budget_from_env
#define qwen_prefix_cache_budget_for_serve qpc_ram_budget_for_serve
#define qwen_prefix_cache_min_tokens_from_env qpc_ram_min_tokens_from_env
#define qwen_prefix_cache_store qpc_ram_store
#define qwen_prefix_cache_restore qpc_ram_restore
#include "qwen_prefix_cache_impl.h"
#undef qwen_prefix_cache_restore
#undef qwen_prefix_cache_store
#undef qwen_prefix_cache_min_tokens_from_env
#undef qwen_prefix_cache_budget_for_serve
#undef qwen_prefix_cache_budget_from_env

#define QWEN_PREFIX_DISK_STATE_ABI 1u

static inline int qpc_common_mode_allows_ram(void) {
    int mode = coli_prefix_cache_mode();
    return mode != 0;
}

static inline size_t qwen_prefix_cache_budget_from_env(void) {
    if (!qpc_common_mode_allows_ram()) return 0;
    const char *legacy = getenv("QWEN_PREFIX_CACHE_MB");
    if (legacy && *legacy) return qwen_prefix_cache_budget_parse(legacy, 0);
    const char *common = getenv("COLI_PREFIX_CACHE_RAM_MB");
    return qwen_prefix_cache_budget_parse(common, 0);
}

static inline size_t qwen_prefix_cache_budget_for_serve(void) {
    if (!qpc_common_mode_allows_ram()) return 0;
    const size_t fallback = (size_t)QWEN_PREFIX_CACHE_DEFAULT_SERVE_MB * 1024u * 1024u;
    const char *legacy = getenv("QWEN_PREFIX_CACHE_MB");
    if (legacy && *legacy) return qwen_prefix_cache_budget_parse(legacy, fallback);
    return qwen_prefix_cache_budget_parse(getenv("COLI_PREFIX_CACHE_RAM_MB"), fallback);
}

static inline int qwen_prefix_cache_min_tokens_from_env(void) {
    const char *value = getenv("QWEN_PREFIX_CACHE_MIN_TOKENS");
    if (!value || !*value) value = getenv("COLI_PREFIX_CACHE_MIN_TOKENS");
    if (!value || !*value) return QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || parsed < 1) return QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    if (parsed > INT_MAX) return INT_MAX;
    return (int)parsed;
}

static inline void qpc_global_init_if_needed(QwenPrefixCache *c) {
    if (!c || c->initialized) return;
    const char *serve = getenv("SERVE");
    size_t budget = serve && serve[0] == '1'
        ? qwen_prefix_cache_budget_for_serve()
        : qwen_prefix_cache_budget_from_env();
    qwen_prefix_cache_init(c, budget, qwen_prefix_cache_min_tokens_from_env(),
                           getenv("QWEN_PREFIX_LOG") != NULL);
}

typedef struct {
    uint32_t layer_count;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t kv_f16;
    uint64_t gdn_state_elems;
    uint64_t gdn_conv_elems;
} QwenPrefixDiskGeometry;

static inline int qwen_prefix_disk_enabled(void) {
    const char *serve = getenv("SERVE");
    if (!serve || serve[0] != '1') return 0;
    /* Preserve the historical Qwen explicit-off override as a master switch;
     * benchmark cache-off must never silently fall through to SSD. */
    const char *legacy = getenv("QWEN_PREFIX_CACHE_MB");
    if (legacy && *legacy && qwen_prefix_cache_budget_parse(legacy, 0) == 0)
        return 0;
    return coli_prefix_disk_enabled();
}

static inline uint64_t qwen_prefix_disk_namespace(const QwenPrefixStateView *v) {
    if (!v || !qwen_prefix_disk_enabled()) return 0;
    QwenPrefixDiskGeometry g = {
        (uint32_t)v->layer_count, (uint32_t)v->n_kv_heads,
        (uint32_t)v->head_dim, (uint32_t)!!v->kv_f16,
        (uint64_t)v->gdn_state_elems, (uint64_t)v->gdn_conv_elems
    };
    return coli_prefix_disk_namespace("qwen-hybrid", QWEN_PREFIX_DISK_STATE_ABI,
                                      getenv("SNAP"), getenv("COLI_CONFIG"),
                                      &g, sizeof(g));
}

static inline void qwen_prefix_cache_store(QwenPrefixCache *c,
                                            const QwenPrefixStateView *v,
                                            const int *tokens, int token_count) {
    if (!c || !tokens) return;
    qpc_global_init_if_needed(c);
    qpc_ram_store(c, v, tokens, token_count);
    if (token_count < c->min_tokens || !qwen_prefix_disk_enabled()) return;
    uint64_t ns = qwen_prefix_disk_namespace(v);
    if (!ns) return;

    QwenPrefixCacheEntry *e = qpc_find_exact(c, tokens, token_count);
    QwenPrefixCacheEntry *owned = NULL;
    if (!e) {
        size_t bytes, kv_bytes, gdn_elems;
        if (!qwen_prefix_cache_entry_bytes(v, token_count, &bytes, &kv_bytes,
                                           &gdn_elems)) return;
        owned = qpc_capture(v, tokens, token_count, bytes, kv_bytes, gdn_elems);
        e = owned;
    }
    if (e)
        (void)coli_prefix_disk_store(ns, QWEN_PREFIX_DISK_STATE_ABI,
                                     e->tokens, e->token_count,
                                     e->kv, e->kv_bytes,
                                     e->gdn, e->gdn_elems * sizeof(float), c->log);
    qwen_prefix_cache_entry_free(owned);
}

static inline int qwen_prefix_cache_restore(QwenPrefixCache *c,
                                             const QwenPrefixStateView *v,
                                             const int *tokens, int token_count) {
    if (!c || !tokens || token_count <= 1) return 0;
    qpc_global_init_if_needed(c);
    int matched = qpc_ram_restore(c, v, tokens, token_count);
    if (matched || !qwen_prefix_disk_enabled()) return matched;
    uint64_t ns = qwen_prefix_disk_namespace(v);
    if (!ns) return 0;

    ColiPrefixDiskObject d;
    matched = coli_prefix_disk_load_longest(ns, QWEN_PREFIX_DISK_STATE_ABI,
                                             tokens, token_count, &d, c->log);
    if (!matched) return 0;

    size_t expected = 0, kv_bytes = 0, gdn_elems = 0;
    int ok = qwen_prefix_cache_entry_bytes(v, matched, &expected, &kv_bytes,
                                            &gdn_elems) &&
             kv_bytes == d.kv_bytes && gdn_elems * sizeof(float) == d.aux_bytes;
    QwenPrefixCacheEntry e;
    memset(&e, 0, sizeof(e));
    if (ok) {
        e.tokens = d.tokens; e.token_count = d.token_count;
        e.kv = d.kv; e.kv_bytes = d.kv_bytes;
        e.gdn = (float *)d.aux; e.gdn_elems = gdn_elems; e.bytes = expected;
        ok = qpc_restore_entry(&e, v);
    }
    if (!ok) {
        coli_prefix_disk_object_free(&d);
        return 0;
    }

    /* Promote the decoded native boundary into the existing bounded RAM LRU. */
    qpc_ram_store(c, v, d.tokens, d.token_count);
    matched = d.token_count;
    coli_prefix_disk_object_free(&d);
    return matched;
}

static inline ColiPrefixDiskStats qwen_prefix_cache_disk_stats(void) {
    return coli_prefix_disk_stats();
}

#endif
