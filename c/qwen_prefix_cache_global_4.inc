    memcpy(out, h, sizeof(h));
    return 1;
}

static inline int qwen_prefix_disk_runtime_safe(void) {
#ifdef COLI_CUDA
    /* Direct CUDA/HIP builds default to GPU compute. Until the resolved backend
     * is exposed at the cache boundary, persist only an explicitly CPU build so
     * a CPU snapshot can never be restored into a GPU execution variant. */
    const char *cuda = getenv("QWEN_CUDA_COMPUTE");
    if (!cuda || atoi(cuda) != 0) return 0;
#endif
#ifdef COLI_METAL
    const char *metal = getenv("QWEN_METAL_COMPUTE");
    if (metal && atoi(metal) != 0) return 0;
#endif
    return 1;
}

static inline uint64_t qwen_prefix_disk_layout_fingerprint(
    const QwenPrefixStateView *view) {
    if (!view || !view->layer_is_gdn) return 0;
    uint64_t h = UINT64_C(1469598103934665603);
    uint64_t meta[] = {
        QWEN_PREFIX_DISK_STATE_ABI,
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
    if (!serve || serve[0] != '1' || !coli_prefix_disk_mode_allows_ssd() ||
        !qwen_prefix_disk_runtime_safe())
        return 0;
