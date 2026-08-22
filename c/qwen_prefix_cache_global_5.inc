    const char *legacy = getenv("QWEN_PREFIX_CACHE_MB");
    if (legacy && *legacy && qwen_prefix_cache_budget_parse(legacy, 0) == 0)
        return 0;
    return 1;
}

static inline ColiPrefixDiskCache *qwen_prefix_disk_cache(
    const QwenPrefixStateView *view, int min_tokens, int log) {
    static ColiPrefixDiskCache disk;
    static uint64_t current_key;
    if (!view || !qwen_prefix_disk_master_enabled()) return NULL;
    uint8_t model_fp[COLI_PREFIX_DISK_FINGERPRINT_BYTES];
    char directory[COLI_PREFIX_DISK_PATH_MAX];
    if (!qwen_prefix_disk_model_fingerprint(model_fp) ||
        !coli_prefix_disk_default_directory(directory, sizeof(directory)))
        return NULL;
    uint64_t layout = qwen_prefix_disk_layout_fingerprint(view);
    if (!layout) return NULL;
    size_t budget = coli_prefix_disk_budget_from_env(
        (size_t)QWEN_PREFIX_DISK_DEFAULT_GB * 1024u * 1024u * 1024u);
    uint64_t key = coli_prefix_hash64(model_fp, sizeof(model_fp), layout);
    key = coli_prefix_hash64(directory, strlen(directory) + 1, key);
    key = coli_prefix_hash64(&budget, sizeof(budget), key);
    key = coli_prefix_hash64(&min_tokens, sizeof(min_tokens), key);
    if (key != current_key) {
        ColiPrefixDiskCache fresh;
        if (!coli_prefix_disk_init(&fresh, directory, budget, min_tokens,
                                   QWEN_PREFIX_DISK_ENGINE_ID,
                                   QWEN_PREFIX_DISK_STATE_ABI, model_fp,
                                   layout, log))
            return NULL;
        if (log)
            fprintf(stderr,
                    "[QWEN-PREFIX] persistent_backend=%s compute_abi=%u\n",
                    qwen_prefix_runtime_backend_name(),
                    (unsigned)QWEN_PREFIX_DISK_COMPUTE_ABI);
        disk = fresh;
        current_key = key;
    }
    return disk.enabled ? &disk : NULL;
}

static inline int qwen_prefix_disk_publish_entry(
    ColiPrefixDiskCache *disk, const int *tokens,
    const QwenPrefixCacheEntry *entry) {
    if (!disk || !entry || !tokens || entry->token_count <= 0 ||
        entry->gdn_elems > SIZE_MAX / sizeof(float))
        return 0;
    QwenPrefixDiskPayloadHeader header = {
        QWEN_PREFIX_DISK_PAYLOAD_VERSION, 0,