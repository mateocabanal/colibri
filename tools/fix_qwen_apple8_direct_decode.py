#!/usr/bin/env python3
"""Hotfix PR #145's first Qwen Apple8-direct integration after M2 decode crash.

The first end-to-end M2 run reached first-token output, then crashed during the
first generated-token decode. Root cause: MetalIO's fixed 256-slot ceiling can
produce a mixed routed set (some direct Apple8 fmt=17, some canonical fmt=7).
The uniform-format fast path then falls back to expert_apply(), which did not
know fmt=17 and dereferenced the f32 slot fields.

This helper is intentionally temporary/anchor-checked. It patches an already
apply_qwen_apple8_direct[_m2].py-modified worktree in place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "c" / "qwen_moe.c"
METALIO = ROOT / "c" / "metalio.mm"


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"[apple8-decode-fix] already patched: {path.relative_to(ROOT)}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path.relative_to(ROOT)}: expected exactly one anchor, found {count}\n"
            f"anchor:\n{old[:500]}"
        )
    path.write_text(text.replace(old, new, 1))
    print(f"[apple8-decode-fix] patched: {path.relative_to(ROOT)}")


# 40 Qwen layers x top-k/cache slots plus the 64-slot prefill wave already
# exceeds the original global 256-slot metadata ceiling. Slots are lazy MTLBuffer
# allocations, so raising the descriptor ceiling does not itself reserve model
# memory; Qwen's RAM/cache budget remains the residency limiter.
replace_once(
    METALIO,
    "#define METALIO_MAX_SLOTS 256\n",
    "#define METALIO_MAX_SLOTS 4096\n",
    "#define METALIO_MAX_SLOTS 4096",
)

# Mixed fmt=17/fmt=7 routed sets are valid while a direct load falls back (or
# while the slot pool is under pressure). Teach the generic per-expert path to
# consume raw Apple8 instead of treating fmt=17 as f32 and dereferencing NULL.
old = '''static void expert_apply(Model *m, Slot *s, const float *x, float *acc){
    Cfg *c = &m->c; int I = c->moe_inter, D = c->hidden;
    if (s->fmt == 7) {
'''
new = '''static void expert_apply(Model *m, Slot *s, const float *x, float *acc){
    Cfg *c = &m->c; int I = c->moe_inter, D = c->hidden;
#ifdef COLI_METALIO
    if (s->fmt == 17) {
        float *y = falloc(D);
        if (!s->apple8_direct ||
            !coli_apple8_metalio_swiglu_slot(s->mio_slot,
                s->apple8_gate_off, s->apple8_gate_bytes,
                s->apple8_up_off, s->apple8_up_bytes,
                s->apple8_down_off, s->apple8_down_bytes,
                x, y, 1, D, I)) {
            fprintf(stderr, "qwen: direct Apple8 mixed-format decode dispatch failed\\n");
            free(y);
            exit(1);
        }
        for (int d = 0; d < D; d++) acc[d] += y[d];
        free(y);
        m->prof_apple8_direct_blocks++;
        m->prof_apple8_direct_experts++;
        return;
    }
#endif
    if (s->fmt == 7) {
'''
replace_once(QWEN, old, new, "direct Apple8 mixed-format decode dispatch failed")

print("[apple8-decode-fix] mixed direct/canonical decode is now safe")
print("[apple8-decode-fix] MetalIO slot descriptor ceiling: 4096")
print("[apple8-decode-fix] next: cd c && make qwen_moe METAL=1")
