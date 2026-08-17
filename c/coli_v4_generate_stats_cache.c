#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * #12 process-local prefix-cache overlay for COLI_V4_UNIT_GENERATE_STATS.
 *
 * Keep the large generation implementation in deepseek_v4.c. We intercept its
 * existing single kv_prefix_reuse() decision: same-session continuation wins
 * first; otherwise the process-local cache may restore an exact V4 attention
 * snapshot and return the restored prefix length. The original caller then
 * executes only the fresh prompt tail exactly as it already does today.
 *
 * The public session_generate wrapper stores a snapshot only after successful
 * generation, so a cache entry always describes the same state as session->fed.
 */
#define coli_v4_session_generate coli_v4_session_generate_uncached
#include "deepseek_v4_internal.h"
#undef coli_v4_session_generate

#include "coli_v4_prefix_cache.h"

#include <stddef.h>
#include <stdio.h>

/* Defined by the DSpark implementation included by this generation unit. A
 * cross-session target-state restore must not inherit speculative-drafter
 * history from an unrelated request. Target outputs are verified either way,
 * but resetting keeps draft behavior tied to the restored context. */
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
        if (reused && coli_v4_full_dspark_wanted)
            v4_ds_reset_history();
    }
#ifdef COLI_V4_TRACE_ROUTE
    coli_v4_route_trace_begin_request(count, reused);
#endif
    return reused;
}

/* deepseek_v4_internal.h already included kv_prefix.h, so the helper macro
 * rewrites only call sites in the generation unit, not kv_prefix_reuse's inline
 * definition. Rename the public generation implementation so we can append a
 * successful snapshot store without modifying the amalgamated source. */
#define kv_prefix_reuse coli_v4_cached_kv_prefix_reuse
#define coli_v4_session_generate coli_v4_session_generate_uncached
#include "deepseek_v4.c"
#undef coli_v4_session_generate
#undef kv_prefix_reuse

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
    if (!result) coli_v4_prefix_cache_store(session);

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
        uint64_t evictions = delta_u64(after.evictions, before.evictions);
        fprintf(stderr,
                "v4_prefix_cache hit=%llu matched_tokens=%llu restore_bytes=%llu "
                "restore_ms=%.3f stores=%llu evictions=%llu entries=%zu "
                "resident_bytes=%zu budget_bytes=%zu\n",
                (unsigned long long)hits,
                (unsigned long long)matched,
                (unsigned long long)restore_bytes,
                restore_ns / 1.0e6,
                (unsigned long long)stores,
                (unsigned long long)evictions,
                after.entries, after.resident_bytes, after.budget_bytes);
    }
    return result;
}
