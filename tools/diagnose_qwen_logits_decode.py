#!/usr/bin/env python3
"""Trace Qwen serve logits and the first generated-token state transition.

Temporary diagnostic for PR #145. Apply after
`tools/diagnose_qwen_prefix_state_hash.py` has already instrumented the local
c/qwen_moe.c. Set QWEN_LOGIT_TRACE=1 together with QWEN_PREFIX_STATE_HASH=1.

For each generated token this logs the exact float-logit byte hash and the
chosen greedy token before DATA is emitted. After each generated-token step it
also emits the existing hybrid state hash. In the correctness-smoke sequence
(cold P+X -> warm P -> cached P+X), this tells us whether divergence starts in
end-of-prefill logits or in decode.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "c" / "qwen_moe.c"


def replace_once(old: str, new: str, marker: str) -> None:
    text = QWEN.read_text()
    if marker in text:
        print(f"[logit-trace] already patched: {marker}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"c/qwen_moe.c: expected exactly one anchor for {marker}, found {count}\n"
            f"anchor:\n{old[:700]}"
        )
    QWEN.write_text(text.replace(old, new, 1))
    print(f"[logit-trace] patched: {marker}")


if "static uint64_t qwen_state_hash_bytes" not in QWEN.read_text():
    raise SystemExit(
        "apply tools/diagnose_qwen_prefix_state_hash.py first; "
        "qwen_state_hash_bytes is missing"
    )

old = '''    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        int nt = qwen_pick_token(m, logit, -1);
        logits_free(&logit);
'''
new = '''    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        int nt = qwen_pick_token(m, logit, -1);
        if (getenv("QWEN_LOGIT_TRACE")) {
            uint64_t lh = qwen_state_hash_bytes(UINT64_C(1469598103934665603),
                                                 logit,
                                                 (size_t)m->c.vocab * sizeof(float));
            fprintf(stderr,
                    "[QWEN-LOGIT] request=%s gen=%d pos=%d hash=%016llx pick=%d\\n",
                    q->id, s, hist_len, (unsigned long long)lh, nt);
        }
        logits_free(&logit);
'''
replace_once(old, new, "[QWEN-LOGIT] request=%s gen=%d")

old = '''        logit = step(m, &nt, 1, hist_len - 1);
    }
'''
new = '''        logit = step(m, &nt, 1, hist_len - 1);
        if (getenv("QWEN_LOGIT_TRACE")) {
            fprintf(stderr,
                    "[QWEN-DECODE] request=%s processed_gen=%d tokens=%d token=%d\\n",
                    q->id, s, hist_len, nt);
            qwen_prefix_state_hash(m, hist_len, "decode_step");
        }
    }
'''
replace_once(old, new, "[QWEN-DECODE] request=%s processed_gen=%d")

print("[logit-trace] QWEN_LOGIT_TRACE=1 instrumentation ready")
print("[logit-trace] next: cd c && make qwen_moe METAL=1")
