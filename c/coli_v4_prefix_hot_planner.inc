/*
 * Planner-facing adapter for the validated V4 process-RAM prefix cache.
 * Included in coli_v4_prefix_cache.c after coli_v4_prefix_cache_impl.inc so
 * it can inspect the cache under the same mutex without changing native V4
 * snapshot/wire semantics.
 */
#include "prefix_hot_resource.h"

static uint32_t v4_prefix_hot_id(const ColiV4PrefixCacheEntry *entry) {
    if (!entry || !entry->tokens || entry->token_count <= 0) return 0;
    uint32_t h = UINT32_C(2166136261);
    for (int i = 0; i < entry->token_count; i++) {
        uint32_t token = (uint32_t)entry->tokens[i];
        for (unsigned shift = 0; shift < 32; shift += 8) {
            h ^= (token >> shift) & UINT32_C(0xff);
            h *= UINT32_C(16777619);
        }
    }
    h ^= (uint32_t)entry->token_count;
    h *= UINT32_C(16777619);
    return h ? h : UINT32_C(1);
}

size_t coli_v4_prefix_cache_hot_inventory(ColiPrefixHotEntryInfo *entries,
                                           size_t capacity) {
    prefix_cache_init();
    pthread_mutex_lock(&g_prefix_cache.mutex);
    size_t visible = 0;
    for (size_t i = 0; i < g_prefix_cache.count; i++) {
        ColiV4PrefixCacheEntry *entry = g_prefix_cache.entries[i];
        if (!entry || entry->retired) continue;
        if (entries && visible < capacity) {
            entries[visible] = (ColiPrefixHotEntryInfo){
                .id = v4_prefix_hot_id(entry),
                .resident_bytes = (uint64_t)entry->bytes,
                .token_count = entry->token_count > 0
                    ? (uint64_t)entry->token_count : 0,
                /* The validated V4 cache does not yet carry per-entry hit
                 * counters. Leave this exact/unknown rather than projecting
                 * global cache hits onto individual entries. */
                .hit_count = 0,
                .last_used = entry->last_used,
                .references = entry->refs,
            };
        }
        visible++;
    }
    pthread_mutex_unlock(&g_prefix_cache.mutex);
    return visible;
}

size_t coli_v4_prefix_cache_apply_planner_budget(size_t selected_bytes) {
    prefix_cache_init();
    size_t policy_cap = cache_budget_from_env();
    size_t requested = selected_bytes < policy_cap ? selected_bytes : policy_cap;

    pthread_mutex_lock(&g_prefix_cache.mutex);

    /* Entries being restored and snapshot allocations already reserved outside
     * the mutex are mandatory for this replan. Never invalidate those bytes in
     * order to satisfy a smaller optional target. The next replan may reclaim
     * them after the reference/reservation disappears. */
    size_t mandatory = g_prefix_cache.reserved_bytes;
    for (size_t i = 0; i < g_prefix_cache.count; i++) {
        ColiV4PrefixCacheEntry *entry = g_prefix_cache.entries[i];
        if (entry && entry->refs)
            mandatory = safe_add_size(mandatory, entry->bytes);
    }
    size_t effective = requested > mandatory ? requested : mandatory;
    g_prefix_cache.budget_bytes = effective;

    while (g_prefix_cache.count && g_prefix_cache.resident_bytes > effective) {
        size_t victim = oldest_evictable_locked();
        if (victim == SIZE_MAX) break;
        remove_index_locked(victim, 1);
    }

    pthread_mutex_unlock(&g_prefix_cache.mutex);
    return effective;
}
