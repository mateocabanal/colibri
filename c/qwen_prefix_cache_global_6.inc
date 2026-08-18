        (uint64_t)entry->gdn_elems, (uint64_t)entry->kv_bytes
    };
    ColiPrefixDiskPart parts[3] = {
        { &header, sizeof(header) },
        { entry->gdn, entry->gdn_elems * sizeof(float) },
        { entry->kv, entry->kv_bytes },
    };
    return coli_prefix_disk_store_parts(disk, tokens, entry->token_count,
                                        parts, 3);
}

static inline void qwen_prefix_cache_store(QwenPrefixCache *cache,
                                            const QwenPrefixStateView *view,
                                            const int *tokens,
                                            int token_count) {
    if (!cache || !view || !tokens || token_count <= 0) return;
    qpc_global_init_if_needed(cache);
    qpc_ram_store(cache, view, tokens, token_count);
    ColiPrefixDiskCache *disk = qwen_prefix_disk_cache(
        view, qwen_prefix_cache_min_tokens_from_env(), cache->log);
    if (!disk || token_count < disk->min_tokens) return;
    QwenPrefixCacheEntry *entry = qpc_find_exact(cache, tokens, token_count);
    QwenPrefixCacheEntry *owned = NULL;
    if (!entry) {
        size_t bytes = 0, kv_bytes = 0, gdn_elems = 0;
        if (!qwen_prefix_cache_entry_bytes(view, token_count, &bytes,
                                           &kv_bytes, &gdn_elems))
            return;
        owned = qpc_capture(view, tokens, token_count,
                            bytes, kv_bytes, gdn_elems);
        entry = owned;
    }
    if (entry) (void)qwen_prefix_disk_publish_entry(disk, tokens, entry);
    qwen_prefix_cache_entry_free(owned);
}

static inline int qwen_prefix_cache_restore(QwenPrefixCache *cache,
                                             const QwenPrefixStateView *view,
                                             const int *tokens,
                                             int token_count) {
    if (!cache || !view || !tokens || token_count <= 1) return 0;
    qpc_global_init_if_needed(cache);
    int matched = qpc_ram_restore(cache, view, tokens, token_count);
    if (matched) return matched;
    ColiPrefixDiskCache *disk = qwen_prefix_disk_cache(
