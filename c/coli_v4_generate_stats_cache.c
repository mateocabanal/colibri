#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * #12/#80 prefix-cache overlay for COLI_V4_UNIT_GENERATE_STATS.
 *
 * Reuse order is deliberate: same-session continuation first, process-local RAM
 * second, persistent SSD third. Admission remains at the canonical
 * end-of-prefill kv_prefix_record() boundary before decode mutates target state.
 */
#define coli_v4_session_generate coli_v4_session_generate_uncached
#include "deepseek_v4_internal.h"
#undef coli_v4_session_generate

#include "coli_v4_prefix_cache.h"
#include "coli_v4_prefix_disk.h"

#include <stddef.h>
#include <stdio.h>

#ifdef main
#undef main
#define COLI_V4_SKIP_GENERATE_MAIN 1
#define COLI_V4_PREFIX_TEST_SKIP_GENERATE_MAIN 1
#endif

static void v4_ds_reset_history(void);

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
        prefix->len == session->prompt_count && !prefix->tainted)
        coli_v4_prefix_cache_store(session);
}

#define kv_prefix_reuse coli_v4_cached_kv_prefix_reuse
#define kv_prefix_record coli_v4_cached_kv_prefix_record
#define coli_v4_session_generate coli_v4_session_generate_uncached
#include "deepseek_v4.c"
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

    int result = coli_v4_session_generate_uncached(
        session, prompt, prompt_length, options, on_token, user_data,
        stats, error, error_size);

    /* The process-local entry was captured before decode, so after generation
     * it remains the exact immutable request prefix even though live attention
     * state has advanced. Stream that pinned entry now: SSD I/O is outside TTFT
     * and token streaming, at the cost of delaying return/DONE in this first
     * bounded implementation. A later shared writer queue can make this fully
     * asynchronous without changing the payload or cache API. */
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
