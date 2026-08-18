/* qwen_prefix_cache.h — Qwen hybrid-state adapter for the global prefix cache.
 *
 * Qwen owns only the semantic packing of its resumable state:
 *   - used rows of full-attention K/V
 *   - complete Gated DeltaNet recurrent S + causal-conv state
 *
 * Exact-prefix indexing, hard RAM budgeting, LRU eviction and cache telemetry
 * are delegated to prefix_cache.h. The public Qwen API/counters remain stable
 * so qwen_moe.c does not need a parallel cache implementation.
 */
#ifndef QWEN_PREFIX_CACHE_H
#define QWEN_PREFIX_CACHE_H

#include "prefix_cache.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_PREFIX_CACHE_MAX_ENTRIES COLI_PREFIX_CACHE_MAX_ENTRIES
#define QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS 256
#define QWEN_PREFIX_CACHE_DEFAULT_SERVE_MB 256u
#define QWEN_PREFIX_STATE_ABI UINT64_C(0x5157454e00000001)
#define QWEN_PREFIX_SEGMENT_KV 1u
#define QWEN_PREFIX_SEGMENT_GDN 2u

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
    size_t gdn_state_elems;
    size_t gdn_conv_elems;
} QwenPrefixStateView;

/* Retained as a public geometry name for existing tests/tools. The global core
 * owns actual entries; Qwen no longer allocates this object. */
typedef struct QwenPrefixCacheEntry {
    int token_count;
    size_t bytes;
    size_t kv_bytes;
    size_t gdn_elems;
} QwenPrefixCacheEntry;

typedef struct {
    ColiPrefixCache core;
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

typedef struct {
    const QwenPrefixStateView *view;
    int prefix_tokens;
    size_t kv_bytes;
    size_t gdn_elems;
    size_t gdn_bytes;
} QwenPrefixAdapterCtx;

static inline int qpc_size_add(size_t a, size_t b, size_t *out) {
    if (!out || b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static inline int qpc_size_mul(size_t a, size_t b, size_t *out) {
    if (!out || (a && b > SIZE_MAX / a)) return 0;
    *out = a * b;
    return 1;
}

static inline int qpc_ascii_ieq(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        unsigned char ca = (unsigned char)*a++, cb = (unsigned char)*b++;
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return *a == 0 && *b == 0;
}

static inline int qpc_view_valid(const QwenPrefixStateView *v, int prefix_tokens) {
    if (!v || !v->layer_is_gdn || v->layer_count <= 0 ||
        v->n_kv_heads <= 0 || v->head_dim <= 0 || v->max_t <= 0 ||
        prefix_tokens <= 0 || prefix_tokens > v->max_t)
        return 0;
    for (int l = 0; l < v->layer_count; ++l) {
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

static inline int qpc_geometry(const QwenPrefixStateView *v, int token_count,
                               size_t *kv_bytes_out, size_t *gdn_elems_out,
                               size_t *gdn_bytes_out, size_t *segments_out) {
    if (!qpc_view_valid(v, token_count)) return 0;
    size_t full_layers = 0, gdn_layers = 0;
    for (int l = 0; l < v->layer_count; ++l) {
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
    if (kv_bytes_out) *kv_bytes_out = kv_bytes;
    if (gdn_elems_out) *gdn_elems_out = gdn_elems;
    if (gdn_bytes_out) *gdn_bytes_out = gdn_bytes;
    if (segments_out) *segments_out = (kv_bytes ? 1u : 0u) + (gdn_bytes ? 1u : 0u);
    return kv_bytes || gdn_bytes;
}

/* Exact global-core resident geometry for one Qwen snapshot. */
static inline int qwen_prefix_cache_entry_bytes(const QwenPrefixStateView *v,
                                                 int token_count,
                                                 size_t *bytes_out,
                                                 size_t *kv_bytes_out,
                                                 size_t *gdn_elems_out) {
    if (!bytes_out || !kv_bytes_out || !gdn_elems_out) return 0;
    size_t kv_bytes, gdn_elems, gdn_bytes, segments;
    if (!qpc_geometry(v, token_count, &kv_bytes, &gdn_elems, &gdn_bytes,
                      &segments))
        return 0;
    size_t token_bytes, segment_bytes, bytes = sizeof(ColiPrefixCacheEntry);
    if (!qpc_size_mul((size_t)token_count, sizeof(int), &token_bytes) ||
        !qpc_size_mul(segments, sizeof(ColiSequenceSegmentDesc), &segment_bytes) ||
        !qpc_size_add(bytes, token_bytes, &bytes) ||
        !qpc_size_add(bytes, segment_bytes, &bytes) ||
        !qpc_size_add(bytes, kv_bytes, &bytes) ||
        !qpc_size_add(bytes, gdn_bytes, &bytes))
        return 0;
    *bytes_out = bytes;
    *kv_bytes_out = kv_bytes;
    *gdn_elems_out = gdn_elems;
    return 1;
}

static inline uint64_t qpc_hash_mix(uint64_t h, uint64_t value) {
    h ^= value;
    h *= UINT64_C(1099511628211);
    return h;
}

/* The cache object is model-instance local, so this geometry fingerprint only
 * needs to reject accidental layout/view reuse inside that instance. Persistent
 * Qwen caching will replace this with compiled-artifact/tokenizer identity. */
static inline ColiPrefixNamespace qpc_namespace(const QwenPrefixStateView *v) {
    ColiPrefixNamespace ns;
    memset(&ns, 0, sizeof(ns));
    ns.state_abi = QWEN_PREFIX_STATE_ABI;
    uint64_t h = UINT64_C(1469598103934665603);
    h = qpc_hash_mix(h, (uint64_t)v->layer_count);
    h = qpc_hash_mix(h, (uint64_t)v->n_kv_heads);
    h = qpc_hash_mix(h, (uint64_t)v->head_dim);
    h = qpc_hash_mix(h, (uint64_t)v->max_t);
    h = qpc_hash_mix(h, (uint64_t)!!v->kv_f16);
    h = qpc_hash_mix(h, (uint64_t)v->gdn_state_elems);
    h = qpc_hash_mix(h, (uint64_t)v->gdn_conv_elems);
    for (int l = 0; l < v->layer_count; ++l)
        h = qpc_hash_mix(h, (uint64_t)(unsigned char)v->layer_is_gdn[l]);
    for (size_t chunk = 0; chunk < 4; ++chunk) {
        uint64_t x = qpc_hash_mix(h, UINT64_C(0x9e3779b97f4a7c15) * (chunk + 1));
        memcpy(ns.fingerprint + chunk * sizeof(x), &x, sizeof(x));
    }
    return ns;
}

static inline int qpc_describe(void *opaque, uint64_t position,
                               ColiSequenceStateInfo *info,
                               ColiSequenceSegmentDesc *segments,
                               size_t segment_capacity) {
    QwenPrefixAdapterCtx *ctx = (QwenPrefixAdapterCtx *)opaque;
    if (!ctx || !ctx->view || !info || position != (uint64_t)ctx->prefix_tokens ||
        !qpc_view_valid(ctx->view, ctx->prefix_tokens))
        return -1;
    size_t segment_count = (ctx->kv_bytes ? 1u : 0u) + (ctx->gdn_bytes ? 1u : 0u);
    memset(info, 0, sizeof(*info));
    info->state_abi = QWEN_PREFIX_STATE_ABI;
    info->absolute_position = position;
    info->logical_bytes = ctx->kv_bytes + ctx->gdn_bytes;
    info->resident_bytes = info->logical_bytes;
    info->segment_count = segment_count;
    if (!segments) return 0;
    if (segment_capacity < segment_count) return -1;
    size_t out = 0;
    if (ctx->kv_bytes) {
        segments[out++] = (ColiSequenceSegmentDesc){
            .segment_id = QWEN_PREFIX_SEGMENT_KV,
            .kind = COLI_SEQUENCE_SEGMENT_ENGINE_NATIVE,
            .element_bytes = ctx->view->kv_f16 ? sizeof(uint16_t) : sizeof(float),
            .visibility = COLI_SEQUENCE_VIS_CPU | COLI_SEQUENCE_VIS_ACCELERATOR,
            .logical_rows = (uint64_t)ctx->prefix_tokens,
            .row_bytes = 0,
            .snapshot_bytes = ctx->kv_bytes,
            .page_rows = 0,
            .layout_abi = ctx->view->kv_f16 ? 2u : 1u,
        };
    }
    if (ctx->gdn_bytes) {
        segments[out++] = (ColiSequenceSegmentDesc){
            .segment_id = QWEN_PREFIX_SEGMENT_GDN,
            .kind = COLI_SEQUENCE_SEGMENT_RECURRENT_FIXED,
            .element_bytes = sizeof(float),
            .visibility = COLI_SEQUENCE_VIS_CPU | COLI_SEQUENCE_VIS_ACCELERATOR,
            .logical_rows = 1,
            .row_bytes = ctx->gdn_bytes,
            .snapshot_bytes = ctx->gdn_bytes,
            .page_rows = 0,
            .layout_abi = 1u,
        };
    }
    return 0;
}

static inline int qpc_read_kv(const QwenPrefixAdapterCtx *ctx,
                              void *dst, size_t bytes) {
    if (!ctx || !dst || bytes != ctx->kv_bytes) return -1;
    unsigned char *out = (unsigned char *)dst;
    size_t off = 0;
    size_t row_elems = (size_t)ctx->prefix_tokens * (size_t)ctx->view->head_dim;
    size_t row_bytes = row_elems *
        (ctx->view->kv_f16 ? sizeof(uint16_t) : sizeof(float));
    for (int l = 0; l < ctx->view->layer_count; ++l) {
        if (ctx->view->layer_is_gdn[l]) continue;
        for (int g = 0; g < ctx->view->n_kv_heads; ++g) {
            size_t src_off = (size_t)g * (size_t)ctx->view->max_t *
                             (size_t)ctx->view->head_dim;
            const void *ksrc = ctx->view->kv_f16
                ? (const void *)(ctx->view->K16[l] + src_off)
                : (const void *)(ctx->view->K[l] + src_off);
            const void *vsrc = ctx->view->kv_f16
                ? (const void *)(ctx->view->V16[l] + src_off)
                : (const void *)(ctx->view->V[l] + src_off);
            memcpy(out + off, ksrc, row_bytes); off += row_bytes;
            memcpy(out + off, vsrc, row_bytes); off += row_bytes;
        }
    }
    return off == bytes ? 0 : -1;
}

static inline int qpc_write_kv(const QwenPrefixAdapterCtx *ctx,
                               const void *src, size_t bytes) {
    if (!ctx || !src || bytes != ctx->kv_bytes) return -1;
    const unsigned char *in = (const unsigned char *)src;
    size_t off = 0;
    size_t row_elems = (size_t)ctx->prefix_tokens * (size_t)ctx->view->head_dim;
    size_t row_bytes = row_elems *
        (ctx->view->kv_f16 ? sizeof(uint16_t) : sizeof(float));
    for (int l = 0; l < ctx->view->layer_count; ++l) {
        if (ctx->view->layer_is_gdn[l]) continue;
        for (int g = 0; g < ctx->view->n_kv_heads; ++g) {
            size_t dst_off = (size_t)g * (size_t)ctx->view->max_t *
                             (size_t)ctx->view->head_dim;
            void *kdst = ctx->view->kv_f16
                ? (void *)(ctx->view->K16[l] + dst_off)
                : (void *)(ctx->view->K[l] + dst_off);
            void *vdst = ctx->view->kv_f16
                ? (void *)(ctx->view->V16[l] + dst_off)
                : (void *)(ctx->view->V[l] + dst_off);
            memcpy(kdst, in + off, row_bytes); off += row_bytes;
            memcpy(vdst, in + off, row_bytes); off += row_bytes;
        }
    }
    return off == bytes ? 0 : -1;
}

static inline int qpc_read_gdn(const QwenPrefixAdapterCtx *ctx,
                               void *dst, size_t bytes) {
    if (!ctx || !dst || bytes != ctx->gdn_bytes) return -1;
    float *out = (float *)dst;
    size_t off = 0;
    for (int l = 0; l < ctx->view->layer_count; ++l) {
        if (!ctx->view->layer_is_gdn[l]) continue;
        memcpy(out + off, ctx->view->gdn_S[l],
               ctx->view->gdn_state_elems * sizeof(float));
        off += ctx->view->gdn_state_elems;
        memcpy(out + off, ctx->view->gdn_conv[l],
               ctx->view->gdn_conv_elems * sizeof(float));
        off += ctx->view->gdn_conv_elems;
    }
    return off == ctx->gdn_elems ? 0 : -1;
}

static inline int qpc_write_gdn(const QwenPrefixAdapterCtx *ctx,
                                const void *src, size_t bytes) {
    if (!ctx || !src || bytes != ctx->gdn_bytes) return -1;
    const float *in = (const float *)src;
    size_t off = 0;
    for (int l = 0; l < ctx->view->layer_count; ++l) {
        if (!ctx->view->layer_is_gdn[l]) continue;
        memcpy(ctx->view->gdn_S[l], in + off,
               ctx->view->gdn_state_elems * sizeof(float));
        off += ctx->view->gdn_state_elems;
        memcpy(ctx->view->gdn_conv[l], in + off,
               ctx->view->gdn_conv_elems * sizeof(float));
        off += ctx->view->gdn_conv_elems;
    }
    return off == ctx->gdn_elems ? 0 : -1;
}

static inline int qpc_read_segment(void *opaque, uint64_t position,
                                   uint32_t segment_id, uint64_t offset,
                                   void *dst, size_t bytes) {
    QwenPrefixAdapterCtx *ctx = (QwenPrefixAdapterCtx *)opaque;
    if (!ctx || position != (uint64_t)ctx->prefix_tokens || offset != 0) return -1;
    if (segment_id == QWEN_PREFIX_SEGMENT_KV) return qpc_read_kv(ctx, dst, bytes);
    if (segment_id == QWEN_PREFIX_SEGMENT_GDN) return qpc_read_gdn(ctx, dst, bytes);
    return -1;
}

static inline int qpc_write_segment(void *opaque, uint64_t position,
                                    uint32_t segment_id, uint64_t offset,
                                    const void *src, size_t bytes) {
    QwenPrefixAdapterCtx *ctx = (QwenPrefixAdapterCtx *)opaque;
    if (!ctx || position != (uint64_t)ctx->prefix_tokens || offset != 0) return -1;
    if (segment_id == QWEN_PREFIX_SEGMENT_KV) return qpc_write_kv(ctx, src, bytes);
    if (segment_id == QWEN_PREFIX_SEGMENT_GDN) return qpc_write_gdn(ctx, src, bytes);
    return -1;
}

static inline int qpc_finish_restore(void *opaque, uint64_t position) {
    QwenPrefixAdapterCtx *ctx = (QwenPrefixAdapterCtx *)opaque;
    return ctx && position == (uint64_t)ctx->prefix_tokens ? 0 : -1;
}

static const ColiSequenceStateOps qpc_state_ops = {
    qpc_describe,
    qpc_read_segment,
    qpc_write_segment,
    NULL, /* preserve untouched KV tail; only prefix rows are restored */
    qpc_finish_restore,
};

static inline int qpc_adapter_init(QwenPrefixAdapterCtx *ctx,
                                   const QwenPrefixStateView *v,
                                   int prefix_tokens,
                                   ColiSequenceStateAdapter *adapter) {
    if (!ctx || !adapter) return 0;
    memset(ctx, 0, sizeof(*ctx));
    ctx->view = v;
    ctx->prefix_tokens = prefix_tokens;
    size_t segments;
    if (!qpc_geometry(v, prefix_tokens, &ctx->kv_bytes, &ctx->gdn_elems,
                      &ctx->gdn_bytes, &segments))
        return 0;
    adapter->ctx = ctx;
    adapter->ops = &qpc_state_ops;
    return 1;
}

static inline void qpc_sync_compat(QwenPrefixCache *c) {
    if (!c) return;
    ColiPrefixCacheStats s;
    coli_prefix_cache_get_stats(&c->core, &s);
    c->count = s.entries;
    c->resident_bytes = s.ram_resident_bytes;
    c->lookups = s.lookups;
    c->hits = s.hits_ram + s.hits_ssd;
    c->stores = s.stores;
    c->evictions = s.evictions_ram;
    c->matched_tokens = s.matched_tokens;
    c->restore_bytes = s.restore_bytes;
}

static inline void qwen_prefix_cache_clear(QwenPrefixCache *c) {
    if (!c) return;
    coli_prefix_cache_close(&c->core);
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
    (void)coli_prefix_cache_init(&c->core, COLI_PREFIX_CACHE_RAM,
                                 budget_bytes, 0, NULL);
    qpc_sync_compat(c);
}

/* Compatibility helper until #87 replaces engine-local RAM_GB arithmetic. */
static inline int qwen_prefix_cache_ram_cap(const char *value,
                                            size_t prefix_budget_bytes,
                                            int *valid) {
    if (valid) *valid = 0;
    if (!value || !*value) return 0;
    char *end = NULL;
    long double gib = strtold(value, &end);
    if (end == value || !(gib > 0.0L) || gib > 1000000.0L) return 0;
    if (valid) *valid = 1;
    const long double gb = 1000000000.0L;
    long double bytes = gib * gb;
    if (bytes <= (long double)prefix_budget_bytes) return 0;
    long double slots = (bytes - (long double)prefix_budget_bytes) / (2.0L * gb);
    if (slots >= (long double)INT_MAX) return INT_MAX;
    return slots > 0.0L ? (int)slots : 0;
}

static inline size_t qwen_prefix_cache_budget_parse(const char *value,
                                                     size_t fallback_bytes) {
    if (!value || !*value) return fallback_bytes;
    if (qpc_ascii_ieq(value, "off")) return 0;
    char *end = NULL;
    long double mib = strtold(value, &end);
    if (end == value || !(mib > 0.0L)) return 0;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    if (*end) return 0;
    long double bytes = mib * 1024.0L * 1024.0L;
    if (bytes >= (long double)SIZE_MAX) return SIZE_MAX;
    return (size_t)bytes;
}

static inline size_t qwen_prefix_cache_budget_from_env(void) {
    return qwen_prefix_cache_budget_parse(getenv("QWEN_PREFIX_CACHE_MB"), 0);
}

static inline size_t qwen_prefix_cache_budget_for_serve(void) {
    const size_t fallback = (size_t)QWEN_PREFIX_CACHE_DEFAULT_SERVE_MB * 1024u * 1024u;
    return qwen_prefix_cache_budget_parse(getenv("QWEN_PREFIX_CACHE_MB"), fallback);
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
                "[QWEN-PREFIX] budget=%.2fMiB min_tokens=%d mode=global-core-ram-exact-hybrid\n",
                (double)c->budget_bytes / (1024.0 * 1024.0), c->min_tokens);
}

static inline void qwen_prefix_cache_store(QwenPrefixCache *c,
                                            const QwenPrefixStateView *v,
                                            const int *tokens, int token_count) {
    if (!c || !tokens) return;
    qwen_prefix_cache_init_env(c);
    if (!c->budget_bytes || token_count < c->min_tokens ||
        !qpc_view_valid(v, token_count)) return;
    QwenPrefixAdapterCtx ctx;
    ColiSequenceStateAdapter adapter;
    if (!qpc_adapter_init(&ctx, v, token_count, &adapter)) return;
    ColiPrefixNamespace ns = qpc_namespace(v);
    int stored = coli_prefix_cache_store(&c->core, &ns, tokens,
                                         (uint32_t)token_count,
                                         (uint64_t)token_count, &adapter);
    qpc_sync_compat(c);
    if (stored > 0 && c->log)
        fprintf(stderr,
                "[QWEN-PREFIX] store tokens=%d entries=%zu resident=%.2fMiB\n",
                token_count, c->count,
                (double)c->resident_bytes / (1024.0 * 1024.0));
}

static inline int qwen_prefix_cache_restore(QwenPrefixCache *c,
                                             const QwenPrefixStateView *v,
                                             const int *tokens,
                                             int token_count) {
    if (!c || !tokens || token_count <= 1) return 0;
    qwen_prefix_cache_init_env(c);
    if (!c->budget_bytes || !qpc_view_valid(v, token_count - 1)) return 0;

    /* Longest match is not known until the global index chooses it. The Qwen
     * adapter can describe/write any strict prefix <= max_t by changing only
     * its prefix_tokens before the generic restore. Probe the chosen token count
     * first, then invoke the shared restore with the exact semantic boundary. */
    ColiPrefixNamespace ns = qpc_namespace(v);
    ColiPrefixCacheEntry *entry = coli_prefix_find_longest(&c->core, &ns,
                                                           tokens, token_count);
    if (!entry) {
        c->core.stats.lookups++;
        c->core.stats.misses++;
        qpc_sync_compat(c);
        return 0;
    }
    QwenPrefixAdapterCtx ctx;
    ColiSequenceStateAdapter adapter;
    if (!qpc_adapter_init(&ctx, v, (int)entry->token_count, &adapter)) return 0;
    int matched = 0;
    int hit = coli_prefix_cache_restore(&c->core, &ns, tokens, token_count,
                                        &adapter, &matched);
    qpc_sync_compat(c);
    if (hit && c->log)
        fprintf(stderr,
                "[QWEN-PREFIX] hit matched=%d prompt=%d restore=%.2fMiB\n",
                matched, token_count,
                (double)c->restore_bytes / (1024.0 * 1024.0));
    return hit ? matched : 0;
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
