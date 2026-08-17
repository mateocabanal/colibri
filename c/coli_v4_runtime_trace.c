#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

/*
 * #56 request/phase tracing overlay for COLI_V4_UNIT_RUNTIME.
 *
 * `coli_v4_session_generate()` calls kv_prefix_reuse() exactly once after the
 * prompt has been tokenized and immediately before the prompt tail is executed.
 * That is the smallest stable runtime boundary carrying both authoritative
 * prompt length and prefix-reuse length. Intercept only that call and notify
 * the block trace producer; all generation math remains in deepseek_v4.c.
 */
#include "deepseek_v4_internal.h"

/* Implemented by coli_v4_block_hybrid_trace.c. It is a no-op unless detailed
 * route tracing is enabled, so the runtime overlay does not make tracing a
 * semantic dependency of generation. */
void coli_v4_route_trace_begin_request(int prompt_tokens, int reused_tokens);

static int coli_v4_trace_kv_prefix_reuse(const kv_prefix *prefix,
                                         const int *ids, int count) {
    int reused = kv_prefix_reuse(prefix, ids, count);
    coli_v4_route_trace_begin_request(count, reused);
    return reused;
}

/* deepseek_v4_internal.h already pulled in kv_prefix.h, so the macro below
 * rewrites only the runtime unit's call site rather than the inline helper's
 * definition. */
#define kv_prefix_reuse coli_v4_trace_kv_prefix_reuse
#include "deepseek_v4.c"
