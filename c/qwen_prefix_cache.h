/* Common default-on RAM policy + persistent SSD tier around the proven Qwen
 * hybrid snapshot implementation. qwen_prefix_cache_impl.h is byte-identical
 * to the process-local implementation validated in #75/#81. */
#ifndef QWEN_PREFIX_CACHE_GLOBAL_WRAPPER_H
#define QWEN_PREFIX_CACHE_GLOBAL_WRAPPER_H

/* The process-local prefix implementation historically also defined Qwen's
 * 2-decimal-GB-per-slot RAM_GB heuristic. Keep that old symbol private so
 * normal Qwen startup falls back to its safe top-k bootstrap; the adaptive
 * resource adapter below consumes RAM_GB as a TOTAL byte budget through the
 * shared MoE planner once actual expert resident bytes are known. */
#define qwen_prefix_cache_ram_cap qwen_prefix_cache_ram_cap_legacy
#include "qwen_prefix_cache_global_1.inc"
#undef qwen_prefix_cache_ram_cap

static inline int qwen_prefix_cache_ram_cap(const char *value,
                                            size_t prefix_budget_bytes,
                                            int *valid) {
    (void)value;
    (void)prefix_budget_bytes;
    if (valid) *valid = 0;
    return 0;
}

#include "qwen_prefix_cache_global_2.inc"
#include "qwen_prefix_cache_global_3.inc"
#include "qwen_prefix_cache_global_4.inc"
#include "qwen_prefix_cache_global_5.inc"
#include "qwen_prefix_cache_global_6.inc"
#include "qwen_prefix_cache_global_7.inc"

/* qwen_moe.c includes route_trace.h before this wrapper. The adapters therefore
 * wrap the existing semantic/physical seams without changing model math or
 * duplicating residency/resource policy. Prefix-cache-only unit tests include
 * this wrapper without instantiating the engine, so keep their -Werror builds
 * from treating adapter-only static helpers as dead-code failures. */
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "qwen_adaptive_residency_adapter.h"
#include "qwen_adaptive_resource_budget_adapter.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif /* QWEN_PREFIX_CACHE_GLOBAL_WRAPPER_H */
