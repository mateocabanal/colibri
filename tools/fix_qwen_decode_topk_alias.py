#!/usr/bin/env python3
"""Correctness hotfix for Qwen decode when resident expert capacity < top-k.

The current moe_token() gathers Slot* pointers for every routed expert before
applying any of them. expert_get() may LRU-reuse those same Slot objects while
later routed experts are still being gathered. If the per-layer cache capacity
is smaller than top-k (for example cap=4, topk=8 under RAM_GB=10), earlier
pointers silently change identity before compute.

This temporary PR #145 patcher adds an alias-safe fallback: when lc->cap < K,
load/apply/accumulate each routed expert before requesting the next one. It is
correctness-first and intentionally gives up exact-demand overlap in this case.
A proper transient/leased top-k decode wave can restore overlap later.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "c" / "qwen_moe.c"


def replace_once(old: str, new: str, marker: str) -> None:
    text = QWEN.read_text()
    if marker in text:
        print(f"[decode-topk-alias] already patched: {marker}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"c/qwen_moe.c: expected exactly one anchor for {marker}, found {count}\n"
            f"anchor:\n{old[:900]}"
        )
    QWEN.write_text(text.replace(old, new, 1))
    print(f"[decode-topk-alias] patched: {marker}")


old = '''    float *w = malloc(size_mul_or_die((size_t)K, sizeof(float), "router top-k weights"));
    if (!w) { fprintf(stderr, "OOM router weights\\n"); exit(1); }
    for (int i = 0; i < K; i++) w[i] = val[i] / wsum;
    /* Exact-demand async issue: the router has already produced the EXACT
'''
new = '''    float *w = malloc(size_mul_or_die((size_t)K, sizeof(float), "router top-k weights"));
    if (!w) { fprintf(stderr, "OOM router weights\\n"); exit(1); }
    for (int i = 0; i < K; i++) w[i] = val[i] / wsum;

    /* A Slot* returned by expert_get() is not leased: a later miss may LRU-
     * reuse that same object. Gathering K pointers is therefore invalid when
     * the resident cache cannot simultaneously hold all K routed experts.
     * Apply each expert before asking expert_get() for the next one so no
     * pointer can be invalidated underneath us. */
    if (m->cache[layer].cap < K) {
        static int warned_alias_safe = 0;
        if (!warned_alias_safe) {
            fprintf(stderr,
                    "[QWEN-DECODE] resident cap=%d < topk=%d; using alias-safe sequential routed experts\\n",
                    m->cache[layer].cap, K);
            warned_alias_safe = 1;
        }
        if (K <= 64) {
            memcpy(m->last_route, idx, (size_t)K * sizeof(int));
            m->last_route_k = K;
        }
        for (int i = 0; i < K; i++) {
            Slot *s = NULL;
            expert_get(m, layer, idx[i], &s);
            expert_wait_ready(m, s);
            float *y = calloc((size_t)D, sizeof(float));
            if (!y) { fprintf(stderr, "OOM\\n"); exit(1); }
            expert_apply(m, s, x, y);
            for (int d = 0; d < D; d++) acc[d] += y[d] * w[i];
            free(y);
        }
        rt_route(layer, 0, idx, w, K);
        goto routed_experts_done;
    }

    /* Exact-demand async issue: the router has already produced the EXACT
'''
replace_once(old, new, "resident cap=%d < topk=%d; using alias-safe sequential routed experts")

old = '''    /* shared expert: silu(gate(x))*up(x) -> down; * sigmoid(se_g(x)) */
'''
new = '''routed_experts_done:
    ;
    /* shared expert: silu(gate(x))*up(x) -> down; * sigmoid(se_g(x)) */
'''
replace_once(old, new, "routed_experts_done:")

print("[decode-topk-alias] alias-safe decode fallback ready")
print("[decode-topk-alias] next: cd c && make qwen_moe METAL=1")
