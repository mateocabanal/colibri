/* qwen_prefix_cache.h — bounded exact-prefix snapshots for Qwen hybrid state.
 *
 * Qwen3.5/3.6/3.7 mixes full GQA layers with Gated DeltaNet layers. Reusing a
 * prompt therefore requires more than the attention KV cache: the GDN
 * recurrence matrix and causal-convolution history are part of the sequence
 * state too. This helper snapshots both representations at the end of prefill.
 *
 * The cache is deliberately model-instance local. Callers provide a state view
 * over one already-loaded Model, so entries cannot accidentally cross model,
 * tokenizer, quantization, or runtime-layout boundaries. Matching is exact on
 * token IDs and only strict prefixes are restorable (the unmatched tail must
 * contain at least one token so the caller receives fresh logits from step()).
 *
 * Admission is hard-budgeted: entries are evicted before allocation, and the
 * byte estimate includes metadata, token IDs, packed KV rows, and all GDN
 * recurrent/conv state. Allocation failure simply disables that admission;
 * prompt caching is an optimization and must never make inference fail.
 */
#ifndef QWEN_PREFIX_CACHE_H
#define QWEN_PREFIX_CACHE_H

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_PREFIX_CACHE_MAX_ENTRIES 64
#define QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS 256

typedef struct {
    int layer_count;
    const int8_t *layer_is_gdn; /* [layer_count], 1=GDN, 0=full attention */
    int n_kv_heads;
    int head_dim;
    int max_t;
    int kv_f16;
    float **K, **V;
    uint16_t **K16, **V16;
    float **gdn_S, **gdn_conv;
    size_t gdn_state_elems;     /* elements per GDN layer */
    size_t gdn_conv_elems;      /* elements per GDN layer */
} QwenPrefixStateView;

typedef struct QwenPrefixCacheEntry {
    int *tokens;
    int token_count;
    unsigned char *kv;          /* packed used-prefix rows, K then V */
    size_t kv_bytes;
    float *gdn;                 /* each GDN layer: recurrence then conv */
    size_t gdn_elems;
    size_t bytes;
    uint64_t last_used;
} QwenPrefixCacheEntry;

typedef struct {
    QwenPrefixCacheEntry *entries[QWEN_PREFIX_CACHE_MAX_ENTRIES];
    size_t count;
    size_t resident_bytes;
    size_t budget_bytes;
    int min_tokens;
    int log;
    int initialized;
    uint64_t clock;
    uint64_t lookups;
    uint64_t hits;
    uint64_t stores;
    uint64_t evictions;
    uint64_t matched_tokens;
    uint64_t restore_bytes;
} QwenPrefixCache;

typedef struct {
    uint64_t lookups, hits, stores, evictions, matched_tokens, restore_bytes;
    size_t entries, resident_bytes, budget_bytes;
} QwenPrefixCacheStats;

static inline int qpc_size_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static inline int qpc_size_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static inline void qwen_prefix_cache_entry_free(QwenPrefixCacheEntry *e) {
    if (!e) return;
    free(e->tokens);
    free(e->kv);
    free(e->gdn);
    free(e);
}

static inline void qwen_prefix_cache_clear(QwenPrefixCache *c) {
    if (!c) return;
    for (size_t i = 0; i < c->count; i++)
        qwen_prefix_cache_entry_free(c->entries[i]);
    memset(c->entries, 0, sizeof(c->entries));
    c->count = 0;
    c->resident_bytes = 0;
}

static inline void qwen_prefix_cache_init(QwenPrefixCache *c,
                                           size_t budget_bytes,
                                           int min_tokens, int log) {
    if (!c) return;
    if (c->initialized) qwen_prefix_cache_clear(c);
    memset(c, 0, sizeof(*c));
    c->budget_bytes = budget_bytes;
    c->min_tokens = min_tokens > 0 ? min_tokens
                                    : QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    c->log = !!log;
    c->initialized = 1;
}

static inline size_t qwen_prefix_cache_budget_from_env(void) {
    const char *value = getenv("QWEN_PREFIX_CACHE_MB");
    if (!value || !*value) return 0;
    char *end = NULL;
    double mib = strtod(value, &end);
    if (end == value || mib <= 0.0) return 0;
    long double bytes = (long double)mib * 1024.0L * 1024.0L;
    if (bytes >= (long double)SIZE_MAX) return SIZE_MAX;
    return (size_t)bytes;
}

static inline int qwen_prefix_cache_min_tokens_from_env(void) {
    const char *value = getenv("QWEN_PREFIX_CACHE_MIN_TOKENS");
    if (!value || !*value) return QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || parsed < 1)
        return QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    if (parsed > INT_MAX) return INT_MAX;
    return (int)parsed;
}

static inline void qwen_prefix_cache_init_env(QwenPrefixCache *c) {
    if (!c || c->initialized) return;
    qwen_prefix_cache_init(c, qwen_prefix_cache_budget_from_env(),
                           qwen_prefix_cache_min_tokens_from_env(),
                           getenv("QWEN_PREFIX_LOG") != NULL);
    if (c->log)
        fprintf(stderr,
                "[QWEN-PREFIX] budget=%.2fMiB min_tokens=%d mode=process-local-exact-hybrid\n",
                (double)c->budget_bytes / (1024.0 * 1024.0), c->min_tokens);
}

static inline int qpc_view_valid(const QwenPrefixStateView *v, int prefix_tokens) {
    if (!v || !v->layer_is_gdn || v->layer_count <= 0 ||
        v->n_kv_heads <= 0 || v->head_dim <= 0 || v->max_t <= 0 ||
        prefix_tokens <= 0 || prefix_tokens > v->max_t)
        return 0;
    for (int l = 0; l < v->layer_count; l++) {
        if (v->layer_is_gdn[l]) {
            if (!v->gdn_S || !v->gdn_conv || !v->gdn_S[l] || !v->gdn_conv[l])
                return 0;
        } else if (v->kv_f16) {
            if (!v->K16 || !v->V16 || !v->K16[l] || !v->V16[l]) return 0;
        } else {
            if (!v->K || !v->V || !v->K[l] || !v->V[l]) return 0;
        }
    }
    return 1;
}

/* Exact allocation geometry for one snapshot. */
static inline int qwen_prefix_cache_entry_bytes(const QwenPrefixStateView *v,
                                                 int token_count,
                                                 size_t *bytes_out,
                                                 size_t *kv_bytes_out,
                                                 size_t *gdn_elems_out) {
    if (!bytes_out || !kv_bytes_out || !gdn_elems_out ||
        !qpc_view_valid(v, token_count)) return 0;

    size_t full_layers = 0, gdn_layers = 0;
    for (int l = 0; l < v->layer_count; l++) {
        if (v->layer_is_gdn[l]) gdn_layers++;
        else full_layers++;
    }

    size_t row_elems, kv_elems, kv_bytes, gdn_per, gdn_elems, gdn_bytes;
    if (!qpc_size_mul((size_t)token_count, (size_t)v->head_dim, &row_elems) ||
        !qpc_size_mul(row_elems, (size_t)v->n_kv_heads, &kv_elems) ||
        !qpc_size_mul(kv_elems, 2, &kv_elems) ||
        !qpc_size_mul(kv_elems, full_layers, &kv_elems) ||
        !qpc_size_mul(kv_elems, v->kv_f16 ? sizeof(uint16_t) : sizeof(float),
                      &kv_bytes) ||
        !qpc_size_add(v->gdn_state_elems, v->gdn_conv_elems, &gdn_per) ||
        !qpc_size_mul(gdn_per, gdn_layers, &gdn_elems) ||
        !qpc_size_mul(gdn_elems, sizeof(float), &gdn_bytes))
        return 0;

    size_t token_bytes, bytes = sizeof(QwenPrefixCacheEntry);
    if (!qpc_size_mul((size_t)token_count, sizeof(int), &token_bytes) ||
        !qpc_size_add(bytes, token_bytes, &bytes) ||
        !qpc_size_add(bytes, kv_bytes, &bytes) ||
        !qpc_size_add(bytes, gdn_bytes, &bytes))
        return 0;
    *bytes_out = bytes;
    *kv_bytes_out = kv_bytes;
    *gdn_elems_out = gdn_elems;
    return 1;
}

static inline size_t qpc_oldest_index(const QwenPrefixCache *c) {
    size_t victim = 0;
    uint64_t oldest = UINT64_MAX;
    for (size_t i = 0; i < c->count; i++) {
        if (c->entries[i]->last_used < oldest) {
            oldest = c->entries[i]->last_used;
            victim = i;
        }
    }
    return victim;
}

static inline void qpc_remove_index(QwenPrefixCache *c, size_t idx, int eviction) {
    QwenPrefixCacheEntry *e = c->entries[idx];
    if (e->bytes <= c->resident_bytes) c->resident_bytes -= e->bytes;
    else c->resident_bytes = 0;
    for (size_t i = idx + 1; i < c->count; i++) c->entries[i - 1] = c->entries[i];
    c->entries[--c->count] = NULL;
    if (eviction) c->evictions++;
    qwen_prefix_cache_entry_free(e);
}

static inline QwenPrefixCacheEntry *qpc_find_exact(QwenPrefixCache *c,
                                                    const int *tokens,
                                                    int token_count) {
    for (size_t i = 0; i < c->count; i++) {
        QwenPrefixCacheEntry *e = c->entries[i];
        if (e->token_count == token_count &&
            !memcmp(e->tokens, tokens, (size_t)token_count * sizeof(int)))
            return e;
    }
    return NULL;
}

static inline QwenPrefixCacheEntry *qpc_capture(const QwenPrefixStateView *v,
                                                 const int *tokens,
                                                 int token_count,
                                                 size_t bytes,
                                                 size_t kv_bytes,
                                                 size_t gdn_elems) {
    QwenPrefixCacheEntry *e = (QwenPrefixCacheEntry *)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->tokens = (int *)malloc((size_t)token_count * sizeof(int));
    e->kv = kv_bytes ? (unsigned char *)malloc(kv_bytes) : NULL;
    e->gdn = gdn_elems ? (float *)malloc(gdn_elems * sizeof(float)) : NULL;
    if (!e->tokens || (kv_bytes && !e->kv) || (gdn_elems && !e->gdn)) {
        qwen_prefix_cache_entry_free(e);
        return NULL;
    }
    memcpy(e->tokens, tokens, (size_t)token_count * sizeof(int));
    e->token_count = token_count;
    e->kv_bytes = kv_bytes;
    e->gdn_elems = gdn_elems;
    e->bytes = bytes;

    size_t kv_off = 0;
    size_t row_elems = (size_t)token_count * (size_t)v->head_dim;
    size_t row_bytes = row_elems * (v->kv_f16 ? sizeof(uint16_t) : sizeof(float));
    for (int l = 0; l < v->layer_count; l++) {
        if (v->layer_is_gdn[l]) continue;
        for (int g = 0; g < v->n_kv_heads; g++) {
            size_t src_off = (size_t)g * (size_t)v->max_t * (size_t)v->head_dim;
            const void *ksrc = v->kv_f16 ? (const void *)(v->K16[l] + src_off)
                                         : (const void *)(v->K[l] + src_off);
            const void *vsrc = v->kv_f16 ? (const void *)(v->V16[l] + src_off)
                                         : (const void *)(v->V[l] + src_off);
            memcpy(e->kv + kv_off, ksrc, row_bytes); kv_off += row_bytes;
            memcpy(e->kv + kv_off, vsrc, row_bytes); kv_off += row_bytes;
        }
    }

    size_t gd_off = 0;
    for (int l = 0; l < v->layer_count; l++) {
        if (!v->layer_is_gdn[l]) continue;
        memcpy(e->gdn + gd_off, v->gdn_S[l], v->gdn_state_elems * sizeof(float));
        gd_off += v->gdn_state_elems;
        memcpy(e->gdn + gd_off, v->gdn_conv[l], v->gdn_conv_elems * sizeof(float));
        gd_off += v->gdn_conv_elems;
    }
    return e;
}

static inline void qwen_prefix_cache_store(QwenPrefixCache *c,
                                            const QwenPrefixStateView *v,
                                            const int *tokens, int token_count) {
    if (!c || !tokens) return;
    qwen_prefix_cache_init_env(c);
    if (!c->budget_bytes || token_count < c->min_tokens ||
        !qpc_view_valid(v, token_count)) return;

    QwenPrefixCacheEntry *duplicate = qpc_find_exact(c, tokens, token_count);
    if (duplicate) {
        duplicate->last_used = ++c->clock;
        return;
    }

    size_t bytes, kv_bytes, gdn_elems;
    if (!qwen_prefix_cache_entry_bytes(v, token_count, &bytes, &kv_bytes,
                                       &gdn_elems) ||
        !bytes || bytes > c->budget_bytes) return;

    while (c->count &&
           (c->count >= QWEN_PREFIX_CACHE_MAX_ENTRIES ||
            c->resident_bytes > c->budget_bytes - bytes))
        qpc_remove_index(c, qpc_oldest_index(c), 1);
    if (c->count >= QWEN_PREFIX_CACHE_MAX_ENTRIES ||
        c->resident_bytes > c->budget_bytes - bytes) return;

    QwenPrefixCacheEntry *e = qpc_capture(v, tokens, token_count, bytes,
                                          kv_bytes, gdn_elems);
    if (!e) return;
    e->last_used = ++c->clock;
    c->entries[c->count++] = e;
    c->resident_bytes += e->bytes;
    c->stores++;
    if (c->log)
        fprintf(stderr,
                "[QWEN-PREFIX] store tokens=%d bytes=%.2fMiB entries=%zu resident=%.2fMiB\n",
                token_count, (double)e->bytes / (1024.0 * 1024.0), c->count,
                (double)c->resident_bytes / (1024.0 * 1024.0));
}

static inline QwenPrefixCacheEntry *qpc_find_longest(QwenPrefixCache *c,
                                                      const int *tokens,
                                                      int token_count) {
    QwenPrefixCacheEntry *best = NULL;
    for (size_t i = 0; i < c->count; i++) {
        QwenPrefixCacheEntry *e = c->entries[i];
        if (e->token_count <= 0 || e->token_count >= token_count ||
            (best && e->token_count <= best->token_count)) continue;
        if (!memcmp(e->tokens, tokens, (size_t)e->token_count * sizeof(int)))
            best = e;
    }
    return best;
}

static inline int qpc_restore_entry(const QwenPrefixCacheEntry *e,
                                    const QwenPrefixStateView *v) {
    if (!e || !qpc_view_valid(v, e->token_count)) return 0;
    size_t expected, kv_bytes, gdn_elems;
    if (!qwen_prefix_cache_entry_bytes(v, e->token_count, &expected,
                                       &kv_bytes, &gdn_elems) ||
        expected != e->bytes || kv_bytes != e->kv_bytes ||
        gdn_elems != e->gdn_elems) return 0;

    size_t kv_off = 0;
    size_t row_elems = (size_t)e->token_count * (size_t)v->head_dim;
    size_t row_bytes = row_elems * (v->kv_f16 ? sizeof(uint16_t) : sizeof(float));
    for (int l = 0; l < v->layer_count; l++) {
        if (v->layer_is_gdn[l]) continue;
        for (int g = 0; g < v->n_kv_heads; g++) {
            size_t dst_off = (size_t)g * (size_t)v->max_t * (size_t)v->head_dim;
            void *kdst = v->kv_f16 ? (void *)(v->K16[l] + dst_off)
                                   : (void *)(v->K[l] + dst_off);
            void *vdst = v->kv_f16 ? (void *)(v->V16[l] + dst_off)
                                   : (void *)(v->V[l] + dst_off);
            memcpy(kdst, e->kv + kv_off, row_bytes); kv_off += row_bytes;
            memcpy(vdst, e->kv + kv_off, row_bytes); kv_off += row_bytes;
        }
    }

    size_t gd_off = 0;
    for (int l = 0; l < v->layer_count; l++) {
        if (!v->layer_is_gdn[l]) continue;
        memcpy(v->gdn_S[l], e->gdn + gd_off, v->gdn_state_elems * sizeof(float));
        gd_off += v->gdn_state_elems;
        memcpy(v->gdn_conv[l], e->gdn + gd_off, v->gdn_conv_elems * sizeof(float));
        gd_off += v->gdn_conv_elems;
    }
    return 1;
}

static inline int qwen_prefix_cache_restore(QwenPrefixCache *c,
                                             const QwenPrefixStateView *v,
                                             const int *tokens,
                                             int token_count) {
    if (!c || !tokens || token_count <= 1) return 0;
    qwen_prefix_cache_init_env(c);
    if (!c->budget_bytes) return 0;
    c->lookups++;
    QwenPrefixCacheEntry *e = qpc_find_longest(c, tokens, token_count);
    if (!e || !qpc_restore_entry(e, v)) return 0;
    e->last_used = ++c->clock;
    c->hits++;
    c->matched_tokens += (uint64_t)e->token_count;
    c->restore_bytes += (uint64_t)e->bytes;
    if (c->log)
        fprintf(stderr,
                "[QWEN-PREFIX] hit matched=%d prompt=%d restore=%.2fMiB\n",
                e->token_count, token_count,
                (double)e->bytes / (1024.0 * 1024.0));
    return e->token_count;
}

static inline void qwen_prefix_cache_stats(const QwenPrefixCache *c,
                                            QwenPrefixCacheStats *s) {
    if (!s) return;
    memset(s, 0, sizeof(*s));
    if (!c) return;
    s->lookups = c->lookups;
    s->hits = c->hits;
    s->stores = c->stores;
    s->evictions = c->evictions;
    s->matched_tokens = c->matched_tokens;
    s->restore_bytes = c->restore_bytes;
    s->entries = c->count;
    s->resident_bytes = c->resident_bytes;
    s->budget_bytes = c->budget_bytes;
}

#endif /* QWEN_PREFIX_CACHE_H */
