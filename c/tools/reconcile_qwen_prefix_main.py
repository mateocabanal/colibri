#!/usr/bin/env python3
"""Reconcile Qwen prompt-cache hooks onto the current optimized main runtime.

Temporary sync helper. Every transform is exact/fail-closed so a changed main
layout cannot silently drop cache correctness or profiling hooks.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "qwen_moe.c"
MAKE = ROOT / "Makefile"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


q = QWEN.read_text()

# The three-way merge retained the state-view/model-field/cache implementation,
# but main won the include conflict.
if '#include "qwen_prefix_cache.h"\n' not in q:
    q = replace_once(
        q,
        '#include "route_trace.h"\n#include "profile.h"\n',
        '#include "route_trace.h"\n#include "qwen_prefix_cache.h"\n#include "profile.h"\n',
        "prefix-cache include",
    )

# Main's qprof prefill hook must remain immediately after the actual unmatched
# tail prefill. Restore and capture accounting stays separate from qprof.
old_prefill = '''    printf("ACCEPT %s %d\\n", q->id, np); fflush(stdout);\n    request_state_reset(m);\n    float *logit = step(m, ids, np, 0);\n    qprof_prefill_end(m, np);\n    int hist_len = np, gen = 0, limited = 1, cancelled = 0;\n'''
new_prefill = '''    printf("ACCEPT %s %d\\n", q->id, np); fflush(stdout);\n    QwenPrefixStateView prefix_view = qwen_prefix_state_view(m);\n    QwenPrefixCacheStats prefix_before;\n    qwen_prefix_cache_stats(&m->prefix_cache, &prefix_before);\n    double prefix_restore_t0 = now_s();\n    int prefix_reused = qwen_prefix_cache_restore(&m->prefix_cache,\n                                                   &prefix_view, ids, np);\n    double prefix_restore_ms = (now_s() - prefix_restore_t0) * 1000.0;\n    if (!prefix_reused) request_state_reset(m);\n    double prefix_prefill_t0 = now_s();\n    float *logit = step(m, ids + prefix_reused, np - prefix_reused,\n                        prefix_reused);\n    double prefix_prefill_ms = (now_s() - prefix_prefill_t0) * 1000.0;\n    qprof_prefill_end(m, np);\n    size_t prefix_snapshot_bytes = 0, prefix_kv_bytes = 0, prefix_gdn_elems = 0;\n    (void)qwen_prefix_cache_entry_bytes(&prefix_view, np,\n                                        &prefix_snapshot_bytes,\n                                        &prefix_kv_bytes,\n                                        &prefix_gdn_elems);\n    int prefix_captured = 0;\n    double prefix_capture_ms = 0.0;\n    int hist_len = np, gen = 0, limited = 1, cancelled = 0;\n'''
if 'QwenPrefixCacheStats prefix_before;' not in q:
    q = replace_once(q, old_prefill, new_prefill, "cache-aware serve prefill")

old_end = '''    logits_free(&logit);\n    qprof_request_end(m, np, gen);\n    double dt = now_s() - t0;\n'''
new_end = '''    logits_free(&logit);\n    /* Defensive fallback: all normal first-token/stop paths already captured\n     * the exact end-of-prefill state before any generated-token step. */\n    qwen_prefix_capture_once(m, &prefix_view, ids, np,\n                             &prefix_captured, &prefix_capture_ms);\n    QwenPrefixCacheStats prefix_after;\n    qwen_prefix_cache_stats(&m->prefix_cache, &prefix_after);\n    if (m->prefix_cache.log) {\n        unsigned long long restore_bytes =\n            (unsigned long long)(prefix_after.restore_bytes - prefix_before.restore_bytes);\n        unsigned long long stores =\n            (unsigned long long)(prefix_after.stores - prefix_before.stores);\n        fprintf(stderr,\n                "[QWEN-PREFIX] request=%s prompt=%d matched=%d restore_bytes=%llu restore_ms=%.3f prefill_ms=%.3f snapshot_bytes=%zu capture_stored=%llu capture_ms=%.3f entries=%zu resident_bytes=%zu budget_bytes=%zu\\n",\n                q->id, np, prefix_reused, restore_bytes, prefix_restore_ms,\n                prefix_prefill_ms, prefix_snapshot_bytes, stores,\n                prefix_capture_ms, prefix_after.entries,\n                prefix_after.resident_bytes, prefix_after.budget_bytes);\n    }\n    qprof_request_end(m, np, gen);\n    double dt = now_s() - t0;\n'''
if 'QwenPrefixCacheStats prefix_after;' not in q:
    q = replace_once(q, old_end, new_end, "prefix request telemetry")

old_cleanup = '''#ifdef COLI_CUDA\n    if (g_cuda_compute) coli_cuda_shutdown();\n#endif\n    tok_free(&T);\n    return rc;\n}\n'''
new_cleanup = '''#ifdef COLI_CUDA\n    if (g_cuda_compute) coli_cuda_shutdown();\n#endif\n    qwen_prefix_cache_clear(&m.prefix_cache);\n    tok_free(&T);\n    return rc;\n}\n'''
if 'qwen_prefix_cache_clear(&m.prefix_cache);' not in q:
    q = replace_once(q, old_cleanup, new_cleanup, "prefix-cache cleanup")

QWEN.write_text(q)

m = MAKE.read_text()
old_target = 'qwen_moe$(EXE): qwen_moe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h profile.h mxfp4_expert.h mxfp4_runtime.h quant.h backend_cuda.h $(QMOE_COLI_OBJS) $(QMOE_GPU_OBJ) $(MIO_OBJ) $(METAL_OBJ)\n'
new_target = 'qwen_moe$(EXE): qwen_moe.c qwen_prefix_cache.h st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h profile.h mxfp4_expert.h mxfp4_runtime.h quant.h backend_cuda.h $(QMOE_COLI_OBJS) $(QMOE_GPU_OBJ) $(MIO_OBJ) $(METAL_OBJ)\n'
if new_target not in m:
    m = replace_once(m, old_target, new_target, "optimized qwen make dependency")
MAKE.write_text(m)

# Structural invariants: retain both stacks and the fixed-total-memory policy.
checks = [
    '#include "qwen_prefix_cache.h"',
    '#include "mxfp4_expert.h"',
    '#include "mxfp4_runtime.h"',
    'QwenPrefixCache prefix_cache;',
    'qwen_prefix_cache_restore(&m->prefix_cache',
    'qwen_prefix_capture_once(m, &prefix_view',
    'qwen_prefix_cache_ram_cap(getenv("RAM_GB"), prefix_budget',
    'qprof_prefill_end(m, np);',
    'qprof_request_end(m, np, gen);',
]
for needle in checks:
    if needle not in q:
        raise SystemExit(f"missing structural invariant: {needle}")

print("reconciled Qwen prompt cache with optimized main runtime")
