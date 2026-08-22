#!/usr/bin/env python3
"""Instrument Qwen hybrid sequence state around prefix-cache boundaries.

Temporary diagnostic for PR #145. Apply after the Apple8 Qwen patchers have
materialized c/qwen_moe.c locally, then rebuild with METAL=1. Set
QWEN_PREFIX_STATE_HASH=1 to log deterministic 64-bit hashes for the used KV
prefix, GDN recurrence matrices and GDN convolution history.

The useful correctness-smoke sequence is cold P+X -> warm P -> restored P+X.
With QWENMOE_CHUNK=48 and the current benchmark prompt, expected state lines are:
  cold:   chunk_end tokens=48, then chunk_end tokens=63
  warm:   chunk_end tokens=48
  cached: restore tokens=48, then chunk_end tokens=63
This distinguishes state-reset nondeterminism from snapshot-copy corruption.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "c" / "qwen_moe.c"


def replace_once(old: str, new: str, marker: str) -> None:
    text = QWEN.read_text()
    if marker in text:
        print(f"[prefix-state-hash] already patched: {marker}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"c/qwen_moe.c: expected exactly one anchor for {marker}, found {count}\n"
            f"anchor:\n{old[:700]}"
        )
    QWEN.write_text(text.replace(old, new, 1))
    print(f"[prefix-state-hash] patched: {marker}")


# Put the helper immediately after qwen_prefix_state_view(), where the Model
# geometry and KV/GDN helpers are all available and before step_batched().
anchor = '''static QwenPrefixStateView qwen_prefix_state_view(Model *m){
    QwenPrefixStateView view = {
        .layer_count = m->c.n_layers,
        .layer_is_gdn = m->c.layer_is_gdn,
        .n_kv_heads = m->c.n_kv_heads,
        .head_dim = m->c.head_dim,
        .max_t = m->max_t,
        .kv_f16 = g_kv_f16,
        .K = m->K, .V = m->V, .K16 = m->K16, .V16 = m->V16,
        .gdn_S = m->gdn_S, .gdn_conv = m->gdn_conv,
        .gdn_state_elems = gdn_state_count(&m->c),
        .gdn_conv_elems = gdn_conv_count(&m->c),
    };
    return view;
}
'''
helper = anchor + r'''
/* Diagnostic only: FNV-1a over exactly the state bytes a prefix snapshot
 * semantically owns. Hash used KV rows rather than stale tail capacity. */
static uint64_t qwen_state_hash_bytes(uint64_t h, const void *ptr, size_t n){
    const unsigned char *p = (const unsigned char *)ptr;
    for (size_t i = 0; i < n; i++) { h ^= (uint64_t)p[i]; h *= UINT64_C(1099511628211); }
    return h;
}

static void qwen_prefix_state_hash(Model *m, int tokens, const char *where){
    if (!getenv("QWEN_PREFIX_STATE_HASH") || tokens <= 0 || tokens > m->max_t) return;
    const uint64_t seed = UINT64_C(1469598103934665603);
    uint64_t hk = seed, hv = seed, hs = seed, hc = seed;
    size_t elem = g_kv_f16 ? sizeof(uint16_t) : sizeof(float);
    size_t row_bytes = (size_t)tokens * (size_t)m->c.head_dim * elem;
    size_t gs_bytes = gdn_state_count(&m->c) * sizeof(float);
    size_t gc_bytes = gdn_conv_count(&m->c) * sizeof(float);
    for (int l = 0; l < m->c.n_layers; l++) {
        if (m->c.layer_is_gdn[l]) {
            hs = qwen_state_hash_bytes(hs, m->gdn_S[l], gs_bytes);
            hc = qwen_state_hash_bytes(hc, m->gdn_conv[l], gc_bytes);
            continue;
        }
        for (int g = 0; g < m->c.n_kv_heads; g++) {
            size_t off = (size_t)g * (size_t)m->max_t * (size_t)m->c.head_dim;
            const void *kp = g_kv_f16 ? (const void *)(m->K16[l] + off)
                                      : (const void *)(m->K[l] + off);
            const void *vp = g_kv_f16 ? (const void *)(m->V16[l] + off)
                                      : (const void *)(m->V[l] + off);
            hk = qwen_state_hash_bytes(hk, kp, row_bytes);
            hv = qwen_state_hash_bytes(hv, vp, row_bytes);
        }
    }
    fprintf(stderr,
            "[QWEN-STATE] where=%s tokens=%d K=%016llx V=%016llx GDN_S=%016llx GDN_CONV=%016llx\n",
            where ? where : "?", tokens,
            (unsigned long long)hk, (unsigned long long)hv,
            (unsigned long long)hs, (unsigned long long)hc);
}
'''
replace_once(anchor, helper, "[QWEN-STATE] where=%s")

# Hash at each completed batched chunk. For the 48-token diagnostic this gives
# us the cold first-48 boundary and the independently recomputed warm P state.
old = '''        if (m->hot_n > 0) {
            m->token_count += cj;
            if (m->token_count >= m->warmup_tokens) pin_hot_experts(m);
        }
    }
    rmsnorm_row(normed, hbuf + (int64_t)(last_cj - 1) * D, m->final_norm, D, c->eps);
'''
new = '''        qwen_prefix_state_hash(m, pos_base + j0 + cj, "chunk_end");
        if (m->hot_n > 0) {
            m->token_count += cj;
            if (m->token_count >= m->warmup_tokens) pin_hot_experts(m);
        }
    }
    rmsnorm_row(normed, hbuf + (int64_t)(last_cj - 1) * D, m->final_norm, D, c->eps);
'''
replace_once(old, new, 'qwen_prefix_state_hash(m, pos_base + j0 + cj, "chunk_end")')

# Hash immediately after a successful restore and before the unmatched tail is
# evaluated. This must equal the warm token-48 hash byte-for-byte.
old = '''    int prefix_reused = qwen_prefix_cache_restore(&m->prefix_cache,
                                                   &prefix_view, ids, np);
    double prefix_restore_ms = (now_s() - prefix_restore_t0) * 1000.0;
    if (!prefix_reused) request_state_reset(m);
'''
new = '''    int prefix_reused = qwen_prefix_cache_restore(&m->prefix_cache,
                                                   &prefix_view, ids, np);
    double prefix_restore_ms = (now_s() - prefix_restore_t0) * 1000.0;
    if (prefix_reused) qwen_prefix_state_hash(m, prefix_reused, "restore");
    if (!prefix_reused) request_state_reset(m);
'''
replace_once(old, new, 'qwen_prefix_state_hash(m, prefix_reused, "restore")')

print("[prefix-state-hash] QWEN_PREFIX_STATE_HASH=1 instrumentation ready")
print("[prefix-state-hash] next: cd c && make qwen_moe METAL=1")
