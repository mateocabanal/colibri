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
#include <stdlib.h>
#include <string.h>

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
    if (ids == session->prompt_ids + position &&
        prefix->len == session->prompt_count && !prefix->tainted) {
        coli_v4_prefix_cache_store(session);
        coli_v4_prefix_disk_publish_live_prefix(session);
    }
}

/* head.weight is target state, not an MTP ancillary tensor. Package generation
 * has already bound its ColiExecutor before entering the amalgamated generation
 * unit, so target-head lookup must depend on that binding rather than on the
 * contents of the non-owning safetensors sentinel installed by the runtime TU.
 * Keeping this contract executor-based also prevents future sentinel metadata
 * changes from silently disabling package speculative verification. */
static ColiSafetensorsTensor g_coli_v4_package_head_tensor;
static int g_coli_v4_package_head_cache_miss_reported;
static int g_coli_v4_package_head_lookup_miss_reported;
static unsigned g_coli_v4_mtp_trace_count;

static int coli_v4_generate_package_bound(void) {
    return g_coli_v4_package_tensor_source.executor != NULL;
}

static int coli_v4_mtp_name(const char *name) {
    return name && !strncmp(name, "mtp.", 4);
}

static int coli_v4_mtp_trace_enabled(void) {
    const char *value = getenv("V4_MTP_TRACE");
    return value && atoi(value) != 0;
}

static void coli_v4_mtp_trace_tensor(const ColiSafetensorsTensor *tensor,
                                     const char *name) {
    if (!coli_v4_mtp_trace_enabled() || g_coli_v4_mtp_trace_count >= 64) return;
    g_coli_v4_mtp_trace_count++;
    if (!tensor) {
        fprintf(stderr, "[MTP] package tensor lookup failed: %s\n",
                name ? name : "<null>");
        return;
    }
    fprintf(stderr,
            "[MTP] package tensor name=%s dtype=%d rank=%d numel=%lld bytes=%lld",
            name, tensor->dtype, tensor->rank,
            (long long)tensor->numel, (long long)tensor->nbytes);
    for (int i = 0; i < tensor->rank && i < 4; i++)
        fprintf(stderr, "%s%lld", i ? "x" : " shape=",
                (long long)tensor->shape[i]);
    fputc('\n', stderr);
}

static const ColiSafetensorsTensor *coli_v4_generate_st_find(
        const ColiSafetensorsIndex *index, const char *name) {
    if (!name || strcmp(name, "head.weight") ||
        !coli_v4_generate_package_bound()) {
        const ColiSafetensorsTensor *tensor =
            coli_v4_package_source_find(index, name);
        if (coli_v4_generate_package_bound() && coli_v4_mtp_name(name)) {
            if (!tensor)
                fprintf(stderr, "[MTP] package tensor lookup failed: %s\n", name);
            else
                coli_v4_mtp_trace_tensor(tensor, name);
        }
        return tensor;
    }

    const ColiExecutor *executor = g_coli_v4_package_tensor_source.executor;
    const ColiRecordInfo *record =
        coli_executor_record_by_name(executor, "head.weight");
    ColiTensorInfo info;
    memset(&info, 0, sizeof(info));
    const ColiPackage *package = coli_executor_package(executor);
    int info_failed = record && package
        ? coli_package_tensor_info(package, record, &info, NULL, 0)
        : 1;
    if (!record || !package || record->kind != COLI_CSF_REC_TENSOR ||
        record->math_format != COLI_CSF_MATH_BF16 || info_failed ||
        info.rank != 2 || info.data_stored_bytes > INT64_MAX) {
        if (!g_coli_v4_package_head_lookup_miss_reported) {
            g_coli_v4_package_head_lookup_miss_reported = 1;
            fprintf(stderr,
                    "v4_spec_head lookup_failed record=%d package=%d kind=%u math=%u "
                    "tensor_info=%d rank=%u stored=%llu\n",
                    record ? 1 : 0, package ? 1 : 0,
                    record ? (unsigned)record->kind : 0u,
                    record ? (unsigned)record->math_format : 0u,
                    info_failed, (unsigned)info.rank,
                    (unsigned long long)info.data_stored_bytes);
        }
        return NULL;
    }

    int64_t numel = 0;
    if (coli_v4_package_tensor_numel(&info, &numel)) {
        if (!g_coli_v4_package_head_lookup_miss_reported) {
            g_coli_v4_package_head_lookup_miss_reported = 1;
            fprintf(stderr,
                    "v4_spec_head lookup_failed reason=numel rank=%u dims0=%llu dims1=%llu\n",
                    (unsigned)info.rank,
                    (unsigned long long)info.dims[0],
                    (unsigned long long)info.dims[1]);
        }
        return NULL;
    }
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

static int coli_v4_generate_read_tensor(
        const ColiSafetensorsIndex *index,
        const ColiSafetensorsTensor *tensor, void *destination) {
    int result = coli_v4_package_source_read_tensor(index, tensor, destination);
    if (result && tensor && coli_v4_mtp_name(tensor->name))
        fprintf(stderr, "[MTP] package tensor read failed: %s bytes=%lld\n",
                tensor->name, (long long)tensor->nbytes);
    return result;
}

static int64_t coli_v4_generate_read_scale_f32(
        ColiSafetensorsIndex *index, const char *name,
        float *out, int64_t cap, int drop) {
    int64_t result = coli_v4_package_read_scale_f32(index, name, out, cap, drop);
    if (result != cap && coli_v4_mtp_name(name))
        fprintf(stderr,
                "[MTP] package scale read failed: %s got=%lld expected=%lld\n",
                name, (long long)result, (long long)cap);
    return result;
}

static int coli_v4_generate_tensor_load_f32(
        ColiFloatTensor *output, const ColiSafetensorsIndex *index,
        const char *name, char *error, size_t error_size) {
    int result = coli_v4_package_tensor_load_f32(
        output, index, name, error, error_size);
    if (result && coli_v4_mtp_name(name))
        fprintf(stderr, "[MTP] package float load failed: %s reason=%s\n",
                name, error && error[0] ? error : "unknown");
    return result;
}

static int coli_v4_generate_tensor_shard(
        const ColiSafetensorsIndex *index,
        const ColiSafetensorsTensor *tensor) {
    if (tensor == &g_coli_v4_package_head_tensor &&
        coli_v4_generate_package_bound())
        return -2;
    return coli_v4_package_source_tensor_shard(index, tensor);
}

static const void *coli_v4_package_head_cache_data(
        const ColiV4Engine *engine, int shard,
        uint64_t offset, size_t length) {
    if (shard == -2 && coli_v4_generate_package_bound())
        offset = 0;
    const void *data = coli_v4_head_cache_data(engine, shard, offset, length);
    if (!data && shard == -2 && coli_v4_generate_package_bound() &&
        !g_coli_v4_package_head_cache_miss_reported) {
        g_coli_v4_package_head_cache_miss_reported = 1;
        fprintf(stderr,
                "v4_spec_head cache_miss requested_shard=%d requested_offset=%llu "
                "requested_bytes=%zu cache_shard=%d cache_offset=%llu "
                "cache_bytes=%zu cache_data=%d\n",
                shard, (unsigned long long)offset, length,
                engine ? engine->head_cache.shard : 0,
                (unsigned long long)(engine ? engine->head_cache.offset : 0),
                engine ? engine->head_cache.bytes : 0,
                engine && engine->head_cache.data ? 1 : 0);
    }
    return data;
}

#define kv_prefix_reuse coli_v4_cached_kv_prefix_reuse
#define kv_prefix_record coli_v4_cached_kv_prefix_record
#define coli_v4_session_generate coli_v4_session_generate_uncached
#define coli_st_find coli_v4_generate_st_find
#define coli_st_read_tensor coli_v4_generate_read_tensor
#define coli_st_tensor_shard coli_v4_generate_tensor_shard
#define coli_st_read_at coli_v4_package_source_read_at
#define coli_st_read_at_streaming coli_v4_package_source_read_at_streaming
#define st_read_scale_f32 coli_v4_generate_read_scale_f32
#define coli_tensor_load_f32 coli_v4_generate_tensor_load_f32
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
    g_coli_v4_package_head_cache_miss_reported = 0;
    g_coli_v4_package_head_lookup_miss_reported = 0;
    g_coli_v4_mtp_trace_count = 0;

    int result = coli_v4_session_generate_uncached(
        session, prompt, prompt_length, options, on_token, user_data,
        stats, error, error_size);

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
