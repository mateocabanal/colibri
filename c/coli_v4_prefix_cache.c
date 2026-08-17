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

int coli_v4_attention_snapshot_restore_fresh(
    ColiDeepSeekV4WindowAttentionState *state,
    const ColiV4AttentionSnapshot *snapshot,
    const ColiDeepSeekV4Config *config, int layer);

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
    /* Snapshot cloning happens outside the cache mutex. Reserve its exact
     * allocation geometry before cloning so concurrent admissions and a full
     * LRU cannot transiently exceed the configured UMA/RAM budget. */
    size_t reserved_bytes;
    size_t reserved_entries;
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
    uint64_t store_bytes;
    uint64_t store_ns;
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

static ColiV4PrefixCacheEntry *find_exact_locked(ColiV4Engine *engine,
                                                  const int *tokens,
                                                  int count) {
    for (size_t index = 0; index < g_prefix_cache.count; index++) {
        ColiV4PrefixCacheEntry *entry = g_prefix_cache.entries[index];
        if (same_tokens(entry, engine, tokens, count)) return entry;
    }
    return NULL;
}

static int cache_already_has(ColiV4Engine *engine,
                             const int *tokens, int count) {
    int found = 0;
    pthread_mutex_lock(&g_prefix_cache.mutex);
    ColiV4PrefixCacheEntry *entry = find_exact_locked(engine, tokens, count);
    if (entry) {
        entry->last_used = ++g_prefix_cache.clock;
        found = 1;
    }
    pthread_mutex_unlock(&g_prefix_cache.mutex);
    return found;
}

/* Allocation-free exact preflight. These bytes mirror capture_entry(): entry
 * metadata, exact token IDs, per-layer snapshot pointers, and each transaction
 * snapshot's requested allocations. */
static size_t entry_snapshot_bytes(const ColiV4Session *session) {
    if (!session || !session->engine || !session->attention ||
        !session->fed.fed || session->fed.tainted || session->fed.len <= 0)
        return SIZE_MAX;
    int layers = session->config.num_hidden_layers;
    if (layers <= 0 || layers > COLI_V4_MAX_LAYERS) return SIZE_MAX;

    size_t token_count = (size_t)session->fed.len;
    if (token_count > SIZE_MAX / sizeof(int) ||
        (size_t)layers > SIZE_MAX / sizeof(ColiV4AttentionSnapshot *))
        return SIZE_MAX;
    size_t bytes = sizeof(ColiV4PrefixCacheEntry);
    bytes = safe_add_size(bytes, token_count * sizeof(int));
    bytes = safe_add_size(bytes,
                          (size_t)layers * sizeof(ColiV4AttentionSnapshot *));
    if (bytes == SIZE_MAX) return SIZE_MAX;

    for (int layer = 0; layer < layers; layer++) {
        size_t snapshot =
            coli_v4_attention_state_snapshot_bytes(session->attention[layer]);
        bytes = safe_add_size(bytes, snapshot);
        if (bytes == SIZE_MAX) return SIZE_MAX;
    }
    return bytes;
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
        if (bytes == SIZE_MAX) goto failed;
    }
    entry->bytes = bytes;
    return entry;

failed:
    entry_free(entry);
    return NULL;
}

static int reserved_geometry_fits_locked(size_t bytes) {
    if (g_prefix_cache.resident_bytes > g_prefix_cache.budget_bytes)
        return 0;
    size_t remaining = g_prefix_cache.budget_bytes - g_prefix_cache.resident_bytes;
    if (g_prefix_cache.reserved_bytes > remaining) return 0;
    remaining -= g_prefix_cache.reserved_bytes;
    if (bytes > remaining) return 0;
    if (g_prefix_cache.count > COLI_V4_PREFIX_CACHE_MAX_ENTRIES ||
        g_prefix_cache.reserved_entries >
            COLI_V4_PREFIX_CACHE_MAX_ENTRIES - g_prefix_cache.count)
        return 0;
    return g_prefix_cache.reserved_entries <
           COLI_V4_PREFIX_CACHE_MAX_ENTRIES - g_prefix_cache.count;
}

static int reserve_snapshot_capacity(size_t bytes) {
    if (!bytes || bytes == SIZE_MAX || bytes > g_prefix_cache.budget_bytes)
        return 0;
    pthread_mutex_lock(&g_prefix_cache.mutex);
    while (!reserved_geometry_fits_locked(bytes)) {
        size_t victim = oldest_evictable_locked();
        if (victim == SIZE_MAX) {
            pthread_mutex_unlock(&g_prefix_cache.mutex);
            return 0;
        }
        remove_index_locked(victim, 1);
    }
    g_prefix_cache.reserved_bytes += bytes;
    g_prefix_cache.reserved_entries++;
    pthread_mutex_unlock(&g_prefix_cache.mutex);
    return 1;
}

static void release_snapshot_reservation_locked(size_t bytes) {
    if (bytes <= g_prefix_cache.reserved_bytes)
        g_prefix_cache.reserved_bytes -= bytes;
    else
        g_prefix_cache.reserved_bytes = 0;
    if (g_prefix_cache.reserved_entries)
        g_prefix_cache.reserved_entries--;
}

static void release_snapshot_reservation(size_t bytes) {
    pthread_mutex_lock(&g_prefix_cache.mutex);
    release_snapshot_reservation_locked(bytes);
    pthread_mutex_unlock(&g_prefix_cache.mutex);
}

void coli_v4_prefix_cache_store(ColiV4Session *session) {
    prefix_cache_init();
    uint64_t began = coli_v4_profile_now();
    if (!g_prefix_cache.budget_bytes || !session || !session->fed.fed ||
        session->fed.tainted || session->fed.len < g_prefix_cache.min_tokens)
        return;
    if (cache_already_has(session->engine, session->fed.fed, session->fed.len))
        return;

    size_t reserved = entry_snapshot_bytes(session);
    if (!reserve_snapshot_capacity(reserved)) return;

    ColiV4PrefixCacheEntry *entry = capture_entry(session);
    if (!entry || !entry->bytes || entry->bytes > reserved) {
        /* Keep the reservation charged until every byte cloned under it is
         * physically gone. Otherwise a concurrent admission can reuse this
         * budget between reservation release and entry_free(), transiently
         * exceeding the hard UMA/RAM cap. */
        entry_free(entry);
        release_snapshot_reservation(reserved);
        return;
    }

    int stored_tokens = entry->token_count;
    size_t stored_bytes = entry->bytes;
    int inserted = 0;
    size_t resident = 0, entries = 0;

    pthread_mutex_lock(&g_prefix_cache.mutex);
    /* Another thread may have admitted the same exact prefix while this clone
     * was being built. Recheck under the insertion lock instead of installing
     * duplicates and wasting the byte budget. */
    ColiV4PrefixCacheEntry *duplicate = find_exact_locked(
        entry->engine, entry->tokens, entry->token_count);
    if (duplicate) {
        duplicate->last_used = ++g_prefix_cache.clock;
    } else {
        /* Evaluate the post-insert geometry while excluding only this clone's
         * own reservation. If accepted, reservation -> resident is one atomic
         * accounting transition under the mutex. Other concurrent clone
         * reservations remain charged throughout. */
        int own_reservation_valid =
            reserved <= g_prefix_cache.reserved_bytes &&
            g_prefix_cache.reserved_entries > 0;
        size_t other_reserved = own_reservation_valid
            ? g_prefix_cache.reserved_bytes - reserved : SIZE_MAX;
        size_t remaining = g_prefix_cache.resident_bytes <= g_prefix_cache.budget_bytes
            ? g_prefix_cache.budget_bytes - g_prefix_cache.resident_bytes : 0;
        if (other_reserved <= remaining)
            remaining -= other_reserved;
        else
            remaining = 0;
        size_t other_reserved_entries = own_reservation_valid
            ? g_prefix_cache.reserved_entries - 1 : SIZE_MAX;
        int count_ok = own_reservation_valid &&
            g_prefix_cache.count < COLI_V4_PREFIX_CACHE_MAX_ENTRIES &&
            other_reserved_entries <
                COLI_V4_PREFIX_CACHE_MAX_ENTRIES - g_prefix_cache.count;
        if (entry->bytes <= remaining && count_ok) {
            release_snapshot_reservation_locked(reserved);
            entry->last_used = ++g_prefix_cache.clock;
            g_prefix_cache.entries[g_prefix_cache.count++] = entry;
            g_prefix_cache.resident_bytes += entry->bytes;
            g_prefix_cache.stores++;
            g_prefix_cache.store_bytes += (uint64_t)entry->bytes;
            g_prefix_cache.store_ns += coli_v4_profile_now() - began;
            inserted = 1;
        }
    }
    resident = g_prefix_cache.resident_bytes;
    entries = g_prefix_cache.count;
    pthread_mutex_unlock(&g_prefix_cache.mutex);

    if (!inserted) {
        /* Duplicate/rejected clones are still covered by their reservation at
         * this point. Free first, then make those bytes available to another
         * admission. */
        entry_free(entry);
        release_snapshot_reservation(reserved);
        return;
    }

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
        if (coli_v4_attention_snapshot_restore_fresh(
                session->attention[layer], entry->attention[layer],
                &session->config, layer)) {
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
    stats->store_bytes = g_prefix_cache.store_bytes;
    stats->store_ns = g_prefix_cache.store_ns;
    stats->resident_bytes = g_prefix_cache.resident_bytes;
    stats->budget_bytes = g_prefix_cache.budget_bytes;
    for (size_t index = 0; index < g_prefix_cache.count; index++)
        if (g_prefix_cache.entries[index] && !g_prefix_cache.entries[index]->retired)
            stats->entries++;
    pthread_mutex_unlock(&g_prefix_cache.mutex);
}
