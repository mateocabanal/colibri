/*
 * Planner-facing adapter for Qwen's proven process-local hybrid prefix cache.
 *
 * This file is included after qwen_prefix_cache_global_*.inc, so it can inspect
 * the existing QwenPrefixCache without changing the snapshot representation.
 * The global runtime owns valuation/allocation; Qwen remains responsible only
 * for exact hybrid-state capture/restore.
 */
#include "prefix_hot_resource.h"

static inline uint32_t qwen_prefix_hot_id(const QwenPrefixCacheEntry *entry) {
    if (!entry || !entry->tokens || entry->token_count <= 0) return 0;
    /* Stable for the lifetime/content of the entry. This id is used only as the
     * common planner's deterministic tie-break; selection is returned by array
     * position, so a hash collision cannot cause the wrong entry to be freed. */
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

/* Snapshot the currently resident Qwen RAM-hot inventory. Qwen's cache is
 * request-thread local today, so there are no active per-entry references to
 * report. Per-entry hit counters are not yet present in the proven cache
 * representation; hit_count is therefore deliberately zero rather than a
 * fabricated estimate. last_used still gives the shared policy an exact
 * recency observation. */
static inline size_t qwen_prefix_cache_hot_inventory(
        QwenPrefixCache *cache, ColiPrefixHotEntryInfo *entries,
        size_t capacity) {
    if (!cache) return 0;
    qpc_global_init_if_needed(cache);
    size_t count = cache->count;
    if (!entries || !capacity) return count;
    size_t emitted = count < capacity ? count : capacity;
    for (size_t i = 0; i < emitted; i++) {
        const QwenPrefixCacheEntry *entry = cache->entries[i];
        entries[i] = (ColiPrefixHotEntryInfo){
            .id = qwen_prefix_hot_id(entry),
            .resident_bytes = entry ? (uint64_t)entry->bytes : 0,
            .token_count = entry && entry->token_count > 0
                ? (uint64_t)entry->token_count : 0,
            .hit_count = 0,
            .last_used = entry ? entry->last_used : 0,
            .references = 0,
        };
    }
    return count;
}

/* Apply a planner-selected Qwen RAM-hot sub-budget. policy_cap_bytes is the
 * compatibility/user policy ceiling that was resolved at startup; the planner
 * may select less, but never turns an explicit cap into a larger allocation.
 *
 * Qwen's current process-local cache has no concurrent restore leases, so
 * reclamation is immediate LRU. The migration to shared multi-session cache
 * ownership must add ref-pinning before calling this from a concurrent planner.
 */
static inline size_t qwen_prefix_cache_apply_planner_budget(
        QwenPrefixCache *cache, size_t policy_cap_bytes,
        size_t selected_bytes) {
    if (!cache) return 0;
    qpc_global_init_if_needed(cache);
    size_t target = selected_bytes < policy_cap_bytes
        ? selected_bytes : policy_cap_bytes;
    cache->budget_bytes = target;
    while (cache->count && cache->resident_bytes > target)
        qpc_remove_index(cache, qpc_oldest_index(cache), 1);
    return cache->budget_bytes;
}
