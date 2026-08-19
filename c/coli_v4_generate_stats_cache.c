#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * #12/#80 prefix-cache overlay for COLI_V4_UNIT_GENERATE_STATS.
 *
 * Reuse order is deliberate: same-session continuation first, process-local RAM
 * second, persistent SSD third. Admission remains at the canonical
 * end-of-prefill kv_prefix_record() boundary before decode mutates target state.
 *
 * The same split unit owns V4 speculative generation, so package-only DSpark
 * binds the named COLITENS compatibility source here.  Real safetensors runs
 * continue through the original functions unchanged.
 */
#define coli_v4_session_generate coli_v4_session_generate_uncached
#include "deepseek_v4_internal.h"
#undef coli_v4_session_generate

#include "coli_v4_prefix_cache.h"
#include "coli_v4_prefix_disk.h"
#include "coli_v4_package_tensor_source.h"

#include <stddef.h>
#include <stdio.h>

#ifdef main
#undef main
#define COLI_V4_SKIP_GENERATE_MAIN 1
#define COLI_V4_PREFIX_TEST_SKIP_GENERATE_MAIN 1
#endif

static void v4_ds_reset_history(void);

#ifdef COLI_V4_ADAPTIVE_ACTIVATION
void coli_v4_activation_begin_request(int prompt_tokens, int reused_tokens);
#endif
#ifdef COLI_V4_TRACE_ROUTE
void coli_v4_route_trace_begin_request(int prompt_tokens, int reused_tokens);
#endif

static ColiV4Session *session_from_prefix(const kv_prefix *prefix) {
    if (!prefix) return NULL;
    return (ColiV4Session *)((char *)(uintptr_t)prefix -
                            offsetof(ColiV4Session, fed));
}

static int coli_v4_cached_kv_prefix_reuse(const kv_prefix *prefix,
                                          const int *ids, int count) {
    int reused = kv_prefix_reuse(prefix, ids, count);
    if (!reused) {
        ColiV4Session *session = session_from_prefix(prefix);
        reused = coli_v4_prefix_cache_restore(session, ids, count);
        if (!reused)
            reused = coli_v4_prefix_disk_restore(session, ids, count);
        if (reused && coli_v4_full_dspark_wanted)
            v4_ds_reset_history();
    }
#ifdef COLI_V4_ADAPTIVE_ACTIVATION
    /* The lightweight block overlay uses this only to distinguish fresh prompt
     * positions from decode positions. It does not inspect token values. */
    coli_v4_activation_begin_request(count, reused);
#endif
#ifdef COLI_V4_TRACE_ROUTE
    coli_v4_route_trace_begin_request(count, reused);
#endif
    return reused;
}

static void coli_v4_cached_kv_prefix_record(kv_prefix *prefix,
                                             const int *ids,
                                             int position, int count) {
    kv_prefix_record(prefix, ids, position, count);
    ColiV4Session *session = session_from_prefix(prefix);
    if (!session || !ids || count <= 0 || !session->prompt_ids ||
        session->prompt_count <= 0)
        return;

    /* The end-of-prefill record is uniquely the slice of session->prompt_ids
     * beginning at the reused position, and after kv_prefix_record the exact
     * state coverage equals prompt_count. Decode/speculative records point at
     * generated/input temporaries instead and therefore cannot admit here. */
    if (ids == session->prompt_ids + position &&
        prefix->len == session->prompt_count && !prefix->tainted) {
        coli_v4_prefix_cache_store(session);
        /* If the RAM tier captured the boundary this is a cheap exact-entry
         * check and SSD publication stays post-generation. In SSD-only mode or
         * when RAM admission cannot fit, persist the live state now while it
         * still denotes the canonical pre-decode prompt boundary. */
        coli_v4_prefix_disk_publish_live_prefix(session);
    }
}

/* head.weight is target state, not an MTP ancillary tensor. Keep speculative
 * target verification independent from the generic package compatibility table:
 * the latter is intentionally bounded and optimized for mtp.* discovery, while
 * the target head already has a native COLI record plus a resident BF16 cache.
 *
 * Returning this tiny descriptor lets the existing batched-head implementation
 * reuse that resident allocation directly. No head bytes are read through this
 * descriptor; scalar package decode continues to use the native COLI path. */
static ColiSafetensorsTensor g_coli_v4_package_head_tensor;

static const ColiSafetensorsTensor *coli_v4_generate_st_find(
        const ColiSafetensorsIndex *index, const char *name) {
    if (!coli_v4_package_source_active(index) || !name ||
        strcmp(name, "head.weight"))
        return coli_v4_package_source_find(index, name);

    const ColiExecutor *executor = g_coli_v4_package_tensor_source.executor;
    const ColiRecordInfo *record = executor
        ? coli_executor_record_by_name(executor, "head.weight") : NULL;
    ColiTensorInfo info;
    const ColiPackage *package = executor ? coli_executor_package(executor) : NULL;
    if (!record || !package || record->kind != COLI_CSF_REC_TENSOR ||
        record->math_format != COLI_CSF_MATH_BF16 ||
        coli_package_tensor_info(package, record, &info, NULL, 0) ||
        info.rank != 2 || info.data_stored_bytes > INT64_MAX)
        return NULL;

    int64_t numel = 0;
    if (coli_v4_package_tensor_numel(&info, &numel)) return NULL;
    memset(&g_coli_v4_package_head_tensor, 0,
           sizeof(g_coli_v4_package_head_tensor));
    g_coli_v4_package_head_tensor.name = (char *)record->name;
    g_coli_v4_package_head_tensor.fd = INT_MIN;
    g_coli_v4_package_head_tensor.off = 0;
    g_coli_v4_package_head_tensor.nbytes = (int64_t)info.data_stored_bytes;
    g_coli_v4_package_head_tensor.dtype = COLI_ST_BF16;
    g_coli_v4_package_head_tensor.numel = numel;
    g_coli_v4_package_head_tensor.rank = 2;
    g_coli_v4_package_head_tensor.shape[0] = (int64_t)info.dims[0];
    g_coli_v4_package_head_tensor.shape[1] = (int64_t)info.dims[1];
    return &g_coli_v4_package_head_tensor;
}

static int coli_v4_generate_tensor_shard(
        const ColiSafetensorsIndex *index,
        const ColiSafetensorsTensor *tensor) {
    if (tensor == &g_coli_v4_package_head_tensor &&
        coli_v4_package_source_active(index))
        return -2;
    return coli_v4_package_source_tensor_shard(index, tensor);
}

/* The synthetic package tensor for head.weight keeps its real COLITENS data
 * offset so ordinary named-range reads remain well-defined. The established
 * V4 resident-head cache, however, uses synthetic shard -2 with cache-relative
 * offset zero. Normalize only that synthetic shard at the cache lookup seam so
 * multi-token verification can evaluate one resident head batch instead of
 * falling back to one scalar package head pass per speculative position. */
static const void *coli_v4_package_head_cache_data(
        const ColiV4Engine *engine, int shard,
        uint64_t offset, size_t length) {
    if (shard == -2 && coli_v4_package_source_active(
            engine ? engine->target_index : NULL))
        offset = 0;
    return coli_v4_head_cache_data(engine, shard, offset, length);
}

#define kv_prefix_reuse coli_v4_cached_kv_prefix_reuse
#define kv_prefix_record coli_v4_cached_kv_prefix_record
#define coli_v4_session_generate coli_v4_session_generate_uncached
#define coli_st_find coli_v4_generate_st_find
#define coli_st_read_tensor coli_v4_package_source_read_tensor
#define coli_st_tensor_shard coli_v4_generate_tensor_shard
#define coli_st_read_at coli_v4_package_source_read_at
#define coli_st_read_at_streaming coli_v4_package_source_read_at_streaming
#define st_read_scale_f32 coli_v4_package_read_scale_f32
#define coli_tensor_load_f32 coli_v4_package_tensor_load_f32
#define coli_v4_head_cache_data coli_v4_package_head_cache_data
#include "deepseek_v4.c"
#undef coli_v4_head_cache_data
#undef coli_tensor_load_f32
#undef st_read_scale_f32
#undef coli_st_read_at_streaming
#undef coli_st_read_at
#undef coli_st_tensor_shard
#undef coli_st_read_tensor
#undef coli_st_find
#undef coli_v4_session_generate
#undef kv_prefix_record
#undef kv_prefix_reuse

#ifdef COLI_V4_PREFIX_TEST_SKIP_GENERATE_MAIN
#undef COLI_V4_PREFIX_TEST_SKIP_GENERATE_MAIN
#undef COLI_V4_SKIP_GENERATE_MAIN
#endif

static uint64_t delta_u64(uint64_t after, uint64_t before) {
    return after >= before ? after - before : 0;
}

int coli_v4_session_generate(ColiV4Session *session,
                             const char *prompt, size_t prompt_length,
                             const ColiV4SessionGenerateOptions *options,
                             ColiV4SessionTokenFn on_token, void *user_data,
                             ColiV4SessionGenerateStats *stats,
                             char *error, size_t error_size) {
    ColiV4PrefixCacheStats before = {0};
    ColiV4PrefixCacheStats after = {0};
    coli_v4_prefix_cache_stats(&before);

    if (coli_v4_package_source_bind(
            session && session->engine ? session->engine->coli_static : NULL)) {
        if (error && error_size)
            snprintf(error, error_size,
                     "cannot bind package named-tensor source");
        return -1;
    }

    int result = coli_v4_session_generate_uncached(
        session, prompt, prompt_length, options, on_token, user_data,
        stats, error, error_size);

    /* The process-local entry was captured before decode, so after generation
     * it remains the exact immutable request prefix even though live attention
     * state has advanced. Stream that pinned entry now: SSD I/O is outside TTFT
     * and token streaming, at the cost of delaying return/DONE in this first
     * bounded implementation. If no RAM entry existed, the end-of-prefill hook
     * already used the live one-layer-at-a-time publisher instead. */
    if (!result)
        coli_v4_prefix_disk_publish_session(session);

    coli_v4_prefix_cache_stats(&after);
    if (after.budget_bytes) {
        uint64_t hits = delta_u64(after.hits, before.hits);
        uint64_t matched = delta_u64(after.matched_tokens,
                                     before.matched_tokens);
        uint64_t restore_bytes = delta_u64(after.restore_bytes,
                                           before.restore_bytes);
        uint64_t restore_ns = delta_u64(after.restore_ns,
                                        before.restore_ns);
        uint64_t stores = delta_u64(after.stores, before.stores);
        uint64_t store_bytes = delta_u64(after.store_bytes, before.store_bytes);
        uint64_t store_ns = delta_u64(after.store_ns, before.store_ns);
        uint64_t evictions = delta_u64(after.evictions, before.evictions);
        fprintf(stderr,
                "v4_prefix_cache hit=%llu matched_tokens=%llu restore_bytes=%llu "
                "restore_ms=%.3f stores=%llu store_bytes=%llu store_ms=%.3f "
                "evictions=%llu entries=%zu resident_bytes=%zu budget_bytes=%zu\n",
                (unsigned long long)hits,
                (unsigned long long)matched,
                (unsigned long long)restore_bytes,
                restore_ns / 1.0e6,
                (unsigned long long)stores,
                (unsigned long long)store_bytes,
                store_ns / 1.0e6,
                (unsigned long long)evictions,
                after.entries, after.resident_bytes, after.budget_bytes);
    }
    return result;
}
