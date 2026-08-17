#!/usr/bin/env python3
"""Temporary exact source patcher for the stacked Qwen prefix metrics branch."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "qwen_moe.c"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


q = QWEN.read_text()
q = replace_once(
    q,
    '''static void serve_one(Model *m, Tok *T, SReq *q, int ctx_cap) {\n''',
    '''static void qwen_prefix_capture_once(Model *m,\n'''
    '''                                     const QwenPrefixStateView *view,\n'''
    '''                                     const int *ids, int np,\n'''
    '''                                     int *captured, double *capture_ms) {\n'''
    '''    if (*captured) return;\n'''
    '''    double began = now_s();\n'''
    '''    qwen_prefix_cache_store(&m->prefix_cache, view, ids, np);\n'''
    '''    *capture_ms += (now_s() - began) * 1000.0;\n'''
    '''    *captured = 1;\n'''
    '''}\n\n'''
    '''static void serve_one(Model *m, Tok *T, SReq *q, int ctx_cap) {\n''',
    "capture-once helper",
)
q = replace_once(
    q,
    '''    QwenPrefixStateView prefix_view = qwen_prefix_state_view(m);\n'''
    '''    int prefix_reused = qwen_prefix_cache_restore(&m->prefix_cache,\n'''
    '''                                                   &prefix_view, ids, np);\n'''
    '''    if (!prefix_reused) request_state_reset(m);\n'''
    '''    float *logit = step(m, ids + prefix_reused, np - prefix_reused,\n'''
    '''                        prefix_reused);\n'''
    '''    /* Capture before decode advances the live state past the prompt. */\n'''
    '''    qwen_prefix_cache_store(&m->prefix_cache, &prefix_view, ids, np);\n'''
    '''    int hist_len = np, gen = 0, limited = 1, cancelled = 0;\n''',
    '''    QwenPrefixStateView prefix_view = qwen_prefix_state_view(m);\n'''
    '''    QwenPrefixCacheStats prefix_before;\n'''
    '''    qwen_prefix_cache_stats(&m->prefix_cache, &prefix_before);\n'''
    '''    double prefix_restore_t0 = now_s();\n'''
    '''    int prefix_reused = qwen_prefix_cache_restore(&m->prefix_cache,\n'''
    '''                                                   &prefix_view, ids, np);\n'''
    '''    double prefix_restore_ms = (now_s() - prefix_restore_t0) * 1000.0;\n'''
    '''    if (!prefix_reused) request_state_reset(m);\n'''
    '''    double prefix_prefill_t0 = now_s();\n'''
    '''    float *logit = step(m, ids + prefix_reused, np - prefix_reused,\n'''
    '''                        prefix_reused);\n'''
    '''    double prefix_prefill_ms = (now_s() - prefix_prefill_t0) * 1000.0;\n'''
    '''    size_t prefix_snapshot_bytes = 0, prefix_kv_bytes = 0, prefix_gdn_elems = 0;\n'''
    '''    (void)qwen_prefix_cache_entry_bytes(&prefix_view, np,\n'''
    '''                                        &prefix_snapshot_bytes,\n'''
    '''                                        &prefix_kv_bytes,\n'''
    '''                                        &prefix_gdn_elems);\n'''
    '''    int prefix_captured = 0;\n'''
    '''    double prefix_capture_ms = 0.0;\n'''
    '''    int hist_len = np, gen = 0, limited = 1, cancelled = 0;\n''',
    "restore/prefill timing",
)
q = replace_once(
    q,
    '''        int nt = qwen_pick_token(m, logit, -1);\n'''
    '''        logits_free(&logit);\n'''
    '''        if (is_stop(nt)) { limited = 0; break; }\n'''
    '''        int nb = tok_decode(T, &nt, 1, buf, sizeof(buf)-1);\n'''
    '''        printf("DATA %s %d\\n", q->id, nb);\n'''
    '''        fwrite(buf, 1, (size_t)nb, stdout);\n'''
    '''        fputc('\\n', stdout); fflush(stdout);\n'''
    '''        gen++; hist_len++;\n''',
    '''        int nt = qwen_pick_token(m, logit, -1);\n'''
    '''        logits_free(&logit);\n'''
    '''        if (is_stop(nt)) {\n'''
    '''            /* No DATA will be emitted. Capture while the live state is\n'''
    '''             * still exactly end-of-prefill, then finish the request. */\n'''
    '''            qwen_prefix_capture_once(m, &prefix_view, ids, np,\n'''
    '''                                     &prefix_captured, &prefix_capture_ms);\n'''
    '''            limited = 0; break;\n'''
    '''        }\n'''
    '''        int nb = tok_decode(T, &nt, 1, buf, sizeof(buf)-1);\n'''
    '''        printf("DATA %s %d\\n", q->id, nb);\n'''
    '''        fwrite(buf, 1, (size_t)nb, stdout);\n'''
    '''        fputc('\\n', stdout); fflush(stdout);\n'''
    '''        /* Sampling/decoding does not mutate sequence state. Emit the first\n'''
    '''         * token before copying the potentially 100+ MiB hybrid snapshot,\n'''
    '''         * then capture before the first generated-token step can advance\n'''
    '''         * attention KV or GDN recurrence. This removes capture memcpy from\n'''
    '''         * TTFT without changing the state being cached. */\n'''
    '''        qwen_prefix_capture_once(m, &prefix_view, ids, np,\n'''
    '''                                 &prefix_captured, &prefix_capture_ms);\n'''
    '''        gen++; hist_len++;\n''',
    "move capture after first DATA",
)
q = replace_once(
    q,
    '''    }\n'''
    '''    logits_free(&logit);\n'''
    '''    double dt = now_s() - t0;\n'''
    '''    double tot = (double)(m->hits - h0 + m->miss - m0);\n''',
    '''    }\n'''
    '''    logits_free(&logit);\n'''
    '''    /* Defensive fallback for any future serve path that leaves the loop\n'''
    '''     * without emitting DATA or taking the stop-token branch. */\n'''
    '''    qwen_prefix_capture_once(m, &prefix_view, ids, np,\n'''
    '''                             &prefix_captured, &prefix_capture_ms);\n'''
    '''    QwenPrefixCacheStats prefix_after;\n'''
    '''    qwen_prefix_cache_stats(&m->prefix_cache, &prefix_after);\n'''
    '''    if (m->prefix_cache.log) {\n'''
    '''        unsigned long long restore_bytes =\n'''
    '''            (unsigned long long)(prefix_after.restore_bytes - prefix_before.restore_bytes);\n'''
    '''        unsigned long long stores =\n'''
    '''            (unsigned long long)(prefix_after.stores - prefix_before.stores);\n'''
    '''        fprintf(stderr,\n'''
    '''                "[QWEN-PREFIX] request=%s prompt=%d matched=%d restore_bytes=%llu restore_ms=%.3f prefill_ms=%.3f snapshot_bytes=%zu capture_stored=%llu capture_ms=%.3f entries=%zu resident_bytes=%zu budget_bytes=%zu\\n",\n'''
    '''                q->id, np, prefix_reused, restore_bytes, prefix_restore_ms,\n'''
    '''                prefix_prefill_ms, prefix_snapshot_bytes, stores,\n'''
    '''                prefix_capture_ms, prefix_after.entries,\n'''
    '''                prefix_after.resident_bytes, prefix_after.budget_bytes);\n'''
    '''    }\n'''
    '''    double dt = now_s() - t0;\n'''
    '''    double tot = (double)(m->hits - h0 + m->miss - m0);\n''',
    "request telemetry",
)
QWEN.write_text(q)
print("qwen prefix timing/post-TTFT capture patch applied")
