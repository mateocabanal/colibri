    memcpy(out, h, sizeof(h));
    return 1;
}

static inline uint64_t qwen_prefix_disk_layout_fingerprint(
    const QwenPrefixStateView *view) {
    if (!view || !view->layer_is_gdn) return 0;
    uint64_t h = UINT64_C(1469598103934665603);
    uint64_t meta[] = {
        QWEN_PREFIX_DISK_STATE_ABI,
        QWEN_PREFIX_DISK_COMPUTE_ABI,
        (uint64_t)qwen_prefix_runtime_backend_id(),
        (uint64_t)(uint32_t)view->layer_count,
        (uint64_t)(uint32_t)view->n_kv_heads,
        (uint64_t)(uint32_t)view->head_dim,
        (uint64_t)(uint32_t)view->kv_f16,
        (uint64_t)view->gdn_state_elems,
        (uint64_t)view->gdn_conv_elems,
    };
    h = coli_prefix_hash64(meta, sizeof(meta), h);
    h = coli_prefix_hash64(view->layer_is_gdn, (size_t)view->layer_count, h);
    const char *chunk = getenv("QWENMOE_CHUNK");
    if (!chunk || !*chunk) chunk = "<default:64>";
    h = coli_prefix_hash64(chunk, strlen(chunk) + 1, h);
    return h;
}

static inline int qwen_prefix_disk_master_enabled(void) {
    const char *serve = getenv("SERVE");
    if (!serve || serve[0] != '1' || !coli_prefix_disk_mode_allows_ssd())
        return 0;