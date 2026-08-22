        view, qwen_prefix_cache_min_tokens_from_env(), cache->log);
    if (!disk) return 0;
    ColiPrefixDiskHit hit;
    matched = coli_prefix_disk_restore_longest(disk, tokens, token_count, &hit);
    if (matched <= 0) return 0;
    if (hit.payload_bytes < sizeof(QwenPrefixDiskPayloadHeader)) {
        coli_prefix_disk_hit_free(&hit);
        return 0;
    }
    QwenPrefixDiskPayloadHeader header;
    memcpy(&header, hit.payload, sizeof(header));
    size_t expected = 0, kv_bytes = 0, gdn_elems = 0;
    int ok = header.version == QWEN_PREFIX_DISK_PAYLOAD_VERSION &&
        qwen_prefix_cache_entry_bytes(view, matched, &expected,
                                      &kv_bytes, &gdn_elems) &&
        header.kv_bytes == kv_bytes && header.gdn_elems == gdn_elems &&
        gdn_elems <= (SIZE_MAX - sizeof(header)) / sizeof(float);
    size_t gdn_bytes = ok ? gdn_elems * sizeof(float) : 0;
    if (ok && (kv_bytes > SIZE_MAX - sizeof(header) - gdn_bytes ||
        hit.payload_bytes != sizeof(header) + gdn_bytes + kv_bytes))
        ok = 0;
    QwenPrefixCacheEntry entry;
    memset(&entry, 0, sizeof(entry));
    if (ok) {
        entry.tokens = hit.tokens;
        entry.token_count = matched;
        entry.gdn = (float *)(void *)(hit.payload + sizeof(header));
        entry.gdn_elems = gdn_elems;
        entry.kv = hit.payload + sizeof(header) + gdn_bytes;
        entry.kv_bytes = kv_bytes;
        entry.bytes = expected;
        ok = qpc_restore_entry(&entry, view);
    }
    /* Keep the public Qwen restore_bytes counter tier-neutral. RAM restore
     * accounts the in-memory snapshot size in qpc_ram_restore; an SSD hit must
     * account the same logical restored state rather than reporting zero. The
     * shared disk store separately tracks physical read_bytes/write_bytes. */
    if (ok) cache->restore_bytes += (uint64_t)expected;
    if (ok && cache->budget_bytes)
        qpc_ram_store(cache, view, tokens, matched);
    coli_prefix_disk_hit_free(&hit);
    return ok ? matched : 0;
}
