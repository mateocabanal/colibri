/* Common default-on RAM policy + persistent SSD tier around the proven Qwen
 * hybrid snapshot implementation. qwen_prefix_cache_impl.h is byte-identical
 * to the process-local implementation validated in #75/#81. */
#ifndef QWEN_PREFIX_CACHE_GLOBAL_WRAPPER_H
#define QWEN_PREFIX_CACHE_GLOBAL_WRAPPER_H
#include "qwen_prefix_cache_global_1.inc"
#include "qwen_prefix_cache_global_2.inc"
#include "qwen_prefix_cache_global_3.inc"
#include "qwen_prefix_cache_global_4.inc"
#include "qwen_prefix_cache_global_5.inc"
#include "qwen_prefix_cache_global_6.inc"
#include "qwen_prefix_cache_global_7.inc"

static inline int qwen_adaptive_legacy_placement_requested(void) {
    const char *adaptive = getenv("COLI_ADAPTIVE_RESIDENCY");
    if (adaptive && adaptive[0] && strcmp(adaptive, "0") == 0) return 1;
    const char *hot = getenv("HOT");
    if (hot && hot[0] && atoi(hot) > 0) return 1;
    const char *pin = getenv("PIN");
    if (pin && pin[0] && strcmp(pin, "auto") != 0) return 1;
    return 0;
}

/* The generated/process-local prefix implementation still contains Qwen's
 * historical 2-decimal-GB-per-slot RAM_GB helper for source compatibility.
 * Automatic execution must not consume that heuristic: fall back to the safe
 * top-k bootstrap and let the adaptive resource adapter interpret RAM_GB as a
 * TOTAL byte budget once actual expert resident bytes are known. Explicit
 * legacy/benchmark mode still calls the original helper so A/B controls remain
 * byte-for-byte useful. */
static inline int qwen_adaptive_bootstrap_ram_cap(
    const char *value, size_t prefix_budget_bytes, int *valid) {
    if (qwen_adaptive_legacy_placement_requested())
        return (qwen_prefix_cache_ram_cap)(value, prefix_budget_bytes, valid);
    (void)value;
    (void)prefix_budget_bytes;
    if (valid) *valid = 0;
    return 0;
}
#define qwen_prefix_cache_ram_cap(value, prefix_budget_bytes, valid) \
    qwen_adaptive_bootstrap_ram_cap((value), (prefix_budget_bytes), (valid))

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
