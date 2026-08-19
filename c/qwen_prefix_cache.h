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

/* qwen_moe.c includes route_trace.h before this wrapper. The adapter therefore
 * wraps the existing semantic route seam without changing the model engine's
 * math or duplicating residency policy. */
#include "qwen_adaptive_residency_adapter.h"
#endif /* QWEN_PREFIX_CACHE_GLOBAL_WRAPPER_H */
