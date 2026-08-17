#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "deepseek_v4_internal.h"
#include "coli_v4_prefix_cache.h"

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLI_V4_PREFIX_CACHE_MAX_ENTRIES 64
#define COLI_V4_PREFIX_CACHE_DEFAULT_MIN_TOKENS 256

typedef struct {
    ColiV4Engine *engine;
    int *tokens;
    int token_count;
    ColiV4AttentionSnapshot **attention;
    int layer_count;
    size_t bytes;
    uint64_t last_used;
    unsigned refs;
    int retired;
} ColiV4PrefixCacheEntry;

typedef struct {
    pthread_mutex_t mutex;
    ColiV4PrefixCacheEntry *entries[COLI_V4_PREFIX_CACHE_MAX_ENTRIES];
    size_t count;
    size_t resident_bytes;
    size_t budget_bytes;
    int min_tokens;
    uint64_t clock;
    uint64_t lookups;
    uint64_t hits;
    uint64_t stores;
    uint64_t evictions;
    uint64_t matched_tokens;
    uint64_t restore_bytes;
    uint64_t restore_ns;
} ColiV4PrefixCache;

static ColiV4PrefixCache g_prefix_cache = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
};
static pthread_once_t g_prefix_cache_once = PTHREAD_ONCE_INIT;

static size_t safe_add_size(size_t a, size_t b) {
    return a > SIZE_MAX - b ? SIZE_MAX : a + b;
}

static size_t cache_budget_from_env(void) {
    const char *value = getenv("V4_PREFIX_CACHE_MB");
    if (!value || !*value) return 0;
    char *end = NULL;
    double mib = strtod(value, &end);
    if (end == value || mib <= 0.0) return 0;
    long double bytes = (long double)mib * 1024.0L * 1024.0L;
    if (bytes >= (long double)SIZE_MAX) return SIZE_MAX;
    return (size_t)bytes;
}

static int cache_min_tokens_from_env(void) {
    const char *value = getenv("V4_PREFIX_CACHE_MIN_TOKENS");
    if (!value || !*value) return COLI_V4_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || parsed < 1) return COLI_V4_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    if (parsed > INT_MAX) return INT_MAX;
    return (int)parsed;
}

static void prefix_cache_init_once(void) {
    g_prefix_cache.budget_bytes = cache_budget_from_env();
    g_prefix_cache.min_tokens = cache_min_tokens_from_env();
    if (getenv("V4_PREFIX_LOG")) {
        fprintf(stderr,
                "[PREFIX-CACHE] budget=%.2fMiB min_tokens=%d mode=process-local-exact\n",
                (double)g_prefix_cache.budget_bytes / (1024.0 * 1024.0),
                g_prefix_cache.min_tokens);
    }
}

static void prefix_cache_init(void) {
    (void)pthread_once(&g_prefix_cache_once, prefix_cache_init_once);
}

size_t coli_v4_prefix_cache_budget_bytes(void) {
    prefix_cache_init();
    return g_prefix_cache.budget_bytes;
}

static void entry_free(ColiV4PrefixCacheEntry *entry) {
    if (!entry) return;
    if (entry->attention) {
        for (int layer = 0; layer < entry->layer_count; layer++)
            coli_v4_attention_snapshot_destroy(entry->attention[layer]);
    }
    free(entry->attention);
    free(entry->tokens);
    free(entry);
}

static void remove_index_locked(size_t index, int eviction) {
    ColiV4PrefixCacheEntry *entry = g_prefix_cache.entries[index];
    if (!entry || entry->refs) return;
    if (entry->bytes <= g_prefix_cache.resident_bytes)
        g_prefix_cache.resident_bytes -= entry->bytes;
    else
        g_prefix_cache.resident_bytes = 0;
    for (size_t item = index + 1; item < g_prefix_cache.count; item++)
        g_prefix_cache.entries[item - 1] = g_prefix_cache.entries[item];
    if (g_prefix_cache.count)
        g_prefix_cache.entries[--g_prefix_cache.count] = NULL;
    if (eviction) g_prefix_cache.evictions++;
    entry_free(entry);
}

static size_t oldest_evictable_locked(void) {
    size_t victim = SIZE_MAX;
    uint64_t oldest = UINT64_MAX;
    for (size_t index = 0; index < g_prefix_cache.count; index++) {
        ColiV4PrefixCacheEntry *entry = g_prefix_cache.entries[index];
        if (!entry || entry->refs) continue;
        if (entry->last_used < oldest) {
            oldest = entry->last_used;
            victim = index;
        }
    }
    return victim;
}

static size_t find_entry_index_locked(const ColiV4PrefixCacheEntry *needle) {
    for (size_t index = 0; index < g_prefix_cache.count; index++)
        if (g_prefix_cache.entries[index] == needle) return index;
    return SIZE_MAX;
}

static void release_entry(ColiV4PrefixCacheEntry *entry) {
    if (!entry) return;
    pthread_mutex_lock(&g_prefix_cache.mutex);
    if (entry->refs) entry->refs--;
    if (entry->retired && !entry->refs) {
        size_t index = find_entry_index_locked(entry);
        if (index != SIZE_MAX) remove_index_locked(index, 0);
    }
    pthread_mutex_unlock(&g_prefix_cache.mutex);
}

static int same_tokens(const ColiV4PrefixCacheEntry *entry,
                       ColiV4Engine *engine, const int *tokens, int count) {
    return entry && !entry->retired && entry->engine == engine &&
           entry->token_count == count && count > 0 &&
           memcmp(entry->tokens, tokens, (size_t)count * sizeof(*tokens)) == 0;
}

static int cache_already_has(ColiV4Engine *engine,
                             const int *tokens, int count) {
    int found = 0;
    pthread_mutex_lock(&g_prefix_cache.mutex);
    for (size_t index = 0; index < g_prefix_cache.count; index++) {
        ColiV4PrefixCacheEntry *entry = g_prefix_cache.entries[index];
        if (same_tokens(entry, engine, tokens, count)) {
            entry->last_used = ++g_prefix_cache.clock;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&g_prefix_cache.mutex);
    return found;
}

static ColiV4PrefixCacheEntry *capture_entry(ColiV4Session *session) {
    if (!session || !session->engine || !session->attention ||
        !session->fed.fed || session->fed.tainted || session->fed.len <= 0)
        return NULL;
    int layers = session->config.num_hidden_layers;
    if (layers <= 0 || layers > COLI_V4_MAX_LAYERS) return NULL;

    ColiV4PrefixCacheEntry *entry = calloc(1, sizeof(*entry));
    if (!entry) return NULL;
    entry->engine = session->engine;
    entry->token_count = session->fed.len;
    entry->layer_count = layers;
    entry->tokens = malloc((size_t)entry->token_count * sizeof(*entry->tokens));
    entry->attention = calloc((size_t)layers, sizeof(*entry->attention));
    if (!entry->tokens || !entry->attention) goto failed;
    memcpy(entry->tokens, session->fed.fed,
           (size_t)entry->token_count * sizeof(*entry->tokens));

    size_t bytes = sizeof(*entry);
    bytes = safe_add_size(bytes, (size_t)entry->token_count * sizeof(*entry->tokens));
    bytes = safe_add_size(bytes, (size_t)layers * sizeof(*entry->attention));
    if (bytes == SIZE_MAX) goto failed;

    for (int layer = 0; layer < layers; layer++) {
        if (coli_v4_attention_snapshot_create(session->attention[layer],
                                               &entry->attention[layer]))
            goto failed;
        size_t snapshot_bytes =
            coli_v4_attention_snapshot_bytes(entry->attention[layer]);
        bytes = safe_add_size(bytes, snapshot_bytes);
        if (bytes == SIZE_MAX || bytes > g_prefix_cache.budget_bytes)
            goto failed;
    }
    entry->bytes = bytes;
    return entry;

failed:
    entry_free(entry);
    return NULL;
}

static int evict_until_fits_locked(size_t bytes) {
    for (;;) {
        size_t used = g_prefix_cache.resident_bytes;
        int bytes_fit = used <= g_prefix_cache.budget_bytes &&
                        bytes <= g_prefix_cache.budget_bytes - used;
        int count_fit = g_prefix_cache.count < COLI_V4_PREFIX_CACHE_MAX_ENTRIES;
        if (bytes_fit && count_fit) return 1;
        size_t victim = oldest_evictable_locked();
        if (victim == SIZE_MAX) return 0;
        remove_index_locked(victim, 1);
    }
}

void coli_v4_prefix_cache_store(ColiV4Session *session) {
    prefix_cache_init();
    if (!g_prefix_cache.budget_bytes || !session || !session->fed.fed ||
        session->fed.tainted || session->fed.len < g_prefix_cache.min_tokens)
        return;
    if (cache_already_has(session->engine, session->fed.fed, session->fed.len))
        return;

    ColiV4PrefixCacheEntry *entry = capture_entry(session);
    if (!entry || !entry->bytes || entry->bytes > g_prefix_cache.budget_bytes) {
        entry_free(entry);
        return;
    }

    int stored_tokens = entry->token_count;
    size_t stored_bytes = entry->bytes;
    pthread_mutex_lock(&g_prefix_cache.mutex);
    if (!evict_until_fits_locked(entry->bytes)) {
        pthread_mutex_unlock(&g_prefix_cache.mutex);
        entry_free(entry);
        return;
    }
    entry->last_used = ++g_prefix_cache.clock;
    g_prefix_cache.entries[g_prefix_cache.count++] = entry;
    g_prefix_cache.resident_bytes += entry->bytes;
    g_prefix_cache.stores++;
    size_t resident = g_prefix_cache.resident_bytes;
    size_t entries = g_prefix_cache.count;
    pthread_mutex_unlock(&g_prefix_cache.mutex);

    if (getenv("V4_PREFIX_LOG"))
        fprintf(stderr,
                "[PREFIX-CACHE] store tokens=%d bytes=%.2fMiB entries=%zu resident=%.2fMiB\n",
                stored_tokens, (double)stored_bytes / (1024.0 * 1024.0),
                entries, (double)resident / (1024.0 * 1024.0));
}

static ColiV4PrefixCacheEntry *find_longest(ColiV4Session *session,
                                            const int *prompt_ids,
                                            int prompt_tokens) {
    ColiV4PrefixCacheEntry *best = NULL;
    pthread_mutex_lock(&g_prefix_cache.mutex);
    g_prefix_cache.lookups++;
    for (size_t index = 0; index < g_prefix_cache.count; index++) {
        ColiV4PrefixCacheEntry *entry = g_prefix_cache.entries[index];
        if (!entry || entry->retired || entry->engine != session->engine ||
            entry->layer_count != session->config.num_hidden_layers ||
            entry->token_count <= 0 || entry->token_count >= prompt_tokens ||
            entry->token_count > session->fed.cap)
            continue;
        if (best && entry->token_count <= best->token_count) continue;
        if (memcmp(entry->tokens, prompt_ids,
                   (size_t)entry->token_count * sizeof(*prompt_ids)) == 0)
            best = entry;
    }
    if (best) {
        best->refs++;
        best->last_used = ++g_prefix_cache.clock;
    }
    pthread_mutex_unlock(&g_prefix_cache.mutex);
    return best;
}

int coli_v4_prefix_cache_restore(ColiV4Session *session,
                                 const int *prompt_ids,
                                 int prompt_tokens) {
    prefix_cache_init();
    if (!g_prefix_cache.budget_bytes || !session || !prompt_ids ||
        prompt_tokens <= 1 || !session->attention || !session->fed.fed)
        return 0;

    ColiV4PrefixCacheEntry *entry = find_longest(session, prompt_ids,
                                                 prompt_tokens);
    if (!entry) return 0;

    uint64_t began = coli_v4_profile_now();
    int ok = 1;
    for (int layer = 0; layer < entry->layer_count; layer++)
        if (coli_v4_attention_snapshot_restore(session->attention[layer],
                                               entry->attention[layer])) {
            ok = 0;
            break;
        }

    if (!ok) {
        for (int layer = 0; layer < session->config.num_hidden_layers; layer++)
            coli_v4_window_attention_reset(session->attention[layer]);
        kv_prefix_clear(&session->fed);
        release_entry(entry);
        return 0;
    }

    memcpy(session->fed.fed, entry->tokens,
           (size_t)entry->token_count * sizeof(*entry->tokens));
    session->fed.len = entry->token_count;
    session->fed.tainted = 0;
    int matched = entry->token_count;
    size_t bytes = entry->bytes;
    uint64_t elapsed = coli_v4_profile_now() - began;

    pthread_mutex_lock(&g_prefix_cache.mutex);
    g_prefix_cache.hits++;
    g_prefix_cache.matched_tokens += (uint64_t)matched;
    g_prefix_cache.restore_bytes += (uint64_t)bytes;
    g_prefix_cache.restore_ns += elapsed;
    pthread_mutex_unlock(&g_prefix_cache.mutex);

    if (getenv("V4_PREFIX_LOG"))
        fprintf(stderr,
                "[PREFIX-CACHE] hit matched=%d prompt=%d restore=%.2fMiB restore_ms=%.3f\n",
                matched, prompt_tokens, (double)bytes / (1024.0 * 1024.0),
                (double)elapsed / 1.0e6);
    release_entry(entry);
    return matched;
}

void coli_v4_prefix_cache_forget_engine(ColiV4Engine *engine) {
    if (!engine) return;
    prefix_cache_init();
    pthread_mutex_lock(&g_prefix_cache.mutex);
    for (size_t index = 0; index < g_prefix_cache.count;) {
        ColiV4PrefixCacheEntry *entry = g_prefix_cache.entries[index];
        if (!entry || entry->engine != engine) {
            index++;
            continue;
        }
        entry->retired = 1;
        if (!entry->refs)
            remove_index_locked(index, 0);
        else
            index++;
    }
    pthread_mutex_unlock(&g_prefix_cache.mutex);
}

void coli_v4_prefix_cache_stats(ColiV4PrefixCacheStats *stats) {
    if (!stats) return;
    prefix_cache_init();
    memset(stats, 0, sizeof(*stats));
    pthread_mutex_lock(&g_prefix_cache.mutex);
    stats->lookups = g_prefix_cache.lookups;
    stats->hits = g_prefix_cache.hits;
    stats->stores = g_prefix_cache.stores;
    stats->evictions = g_prefix_cache.evictions;
    stats->matched_tokens = g_prefix_cache.matched_tokens;
    stats->restore_bytes = g_prefix_cache.restore_bytes;
    stats->restore_ns = g_prefix_cache.restore_ns;
    stats->resident_bytes = g_prefix_cache.resident_bytes;
    stats->budget_bytes = g_prefix_cache.budget_bytes;
    for (size_t index = 0; index < g_prefix_cache.count; index++)
        if (g_prefix_cache.entries[index] && !g_prefix_cache.entries[index]->retired)
            stats->entries++;
    pthread_mutex_unlock(&g_prefix_cache.mutex);
}
