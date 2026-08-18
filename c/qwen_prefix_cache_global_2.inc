    return coli_prefix_ram_budget_from_env(fallback);
}

static inline int qwen_prefix_cache_min_tokens_from_env(void) {
    const char *value = getenv("QWEN_PREFIX_CACHE_MIN_TOKENS");
    if (!value || !*value) value = getenv("COLI_PREFIX_CACHE_MIN_TOKENS");
    if (!value || !*value) return QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || parsed < 1) return QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    if (*end) return QWEN_PREFIX_CACHE_DEFAULT_MIN_TOKENS;
    if (parsed > INT_MAX) return INT_MAX;
    return (int)parsed;
}

static inline void qpc_global_init_if_needed(QwenPrefixCache *cache) {
    if (!cache || cache->initialized) return;
    const char *serve = getenv("SERVE");
    size_t budget = serve && serve[0] == '1'
        ? qwen_prefix_cache_budget_for_serve()
        : qwen_prefix_cache_budget_from_env();
    qwen_prefix_cache_init(cache, budget,
                           qwen_prefix_cache_min_tokens_from_env(),
                           getenv("QWEN_PREFIX_LOG") != NULL);
}

static inline int qpc_hash_file_full(uint64_t lanes[4],
                                     const char *path,
                                     const char *identity) {
    if (!lanes || !path || !*path || !identity) return 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    for (int lane = 0; lane < 4; lane++)
        lanes[lane] = coli_prefix_hash64(identity, strlen(identity) + 1,
                                         lanes[lane]);
    unsigned char buffer[64 * 1024];
    for (;;) {
        size_t got = fread(buffer, 1, sizeof(buffer), fp);
        if (got)
            for (int lane = 0; lane < 4; lane++)
                lanes[lane] = coli_prefix_hash64(buffer, got, lanes[lane]);
        if (got < sizeof(buffer)) {
            int ok = !ferror(fp);
            fclose(fp);
