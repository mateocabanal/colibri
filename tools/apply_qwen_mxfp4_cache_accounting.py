#!/usr/bin/env python3
from pathlib import Path
import subprocess


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)

p = Path("c/qwen_moe.c")
s = p.read_text()

if "qwen_slot_resident_bytes" not in s:
    anchor = '''static void serve_tiers_emap(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts;
    int filled = 0;
    for (int i = 0; i < c->n_layers; i++) filled += m->cache[i].n;
    int64_t I = c->moe_inter, D = c->hidden;
    /* per-expert resident bytes: int8 gate/up/down + one f32 scale per row */
    int64_t slotb = coli_mode ? 3*I*D*2 : 3*I*D + (2*I+D)*4;
    int64_t total_experts = (int64_t)c->n_layers * E;
    printf("TIERS 0 %d %lld 0.00 %.2f\\n", filled,
           (long long)(total_experts - filled), filled*(double)slotb/1e9);
'''
    replacement = '''static uint64_t qwen_slot_resident_bytes(const Cfg *c, const Slot *s) {
    const uint64_t D = (uint64_t)c->hidden;
    const uint64_t I = (uint64_t)c->moe_inter;
    const uint64_t params = 3u * D * I;
    switch (s->fmt) {
        case 0:  return params * sizeof(float);
        case 16: return params * sizeof(uint16_t);
        case 8:  return params + (2u * I + D) * sizeof(float);
        case 4: {
            const uint64_t weights = 2u * I * ((D + 1u) / 2u) + D * ((I + 1u) / 2u);
            const uint64_t scales = (2u * I * ((D + 63u) / 64u) +
                                     D * ((I + 63u) / 64u)) * sizeof(float);
            return weights + scales;
        }
        case 5: {
            const uint64_t weights = 2u * I * (uint64_t)qm_i3_rowbytes((int)D) +
                                     D * (uint64_t)qm_i3_rowbytes((int)I);
            const uint64_t scales = (2u * I * ((D + 63u) / 64u) +
                                     D * ((I + 63u) / 64u)) * sizeof(float);
            return weights + scales;
        }
        case 7: {
            const uint64_t weights = 2u * I * ((D + 1u) / 2u) + D * ((I + 1u) / 2u);
            const uint64_t scales = 2u * I * ((D + 31u) / 32u) +
                                    D * ((I + 31u) / 32u);
            return weights + scales;
        }
        default: return 0;
    }
}

static void serve_tiers_emap(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts;
    int filled = 0;
    uint64_t ram_bytes = 0;
    for (int i = 0; i < c->n_layers; i++) {
        LCache *lc = &m->cache[i];
        for (int z = 0; z < lc->n; z++) {
            Slot *slot = &lc->slots[z];
            if (slot->eid < 0) continue; /* in-flight loads are not published residency */
            filled++;
            ram_bytes += qwen_slot_resident_bytes(c, slot);
        }
    }
    int64_t total_experts = (int64_t)c->n_layers * E;
    printf("TIERS 0 %d %lld 0.00 %.2f\\n", filled,
           (long long)(total_experts - filled), (double)ram_bytes/1e9);
'''
    s = replace_once(s, anchor, replacement, "serve cache accounting")
p.write_text(s)

p = Path("c/tests/test_qwen_moe.c")
t = p.read_text()
if "test_slot_resident_bytes" not in t:
    test = r'''
static void test_slot_resident_bytes(void){
    Cfg c; Slot s;
    memset(&c, 0, sizeof(c)); memset(&s, 0, sizeof(s));
    c.hidden = 65; c.moe_inter = 33;
    const uint64_t D = 65, I = 33;

    s.fmt = 7;
    uint64_t mx_weights = 2u * I * ((D + 1u) / 2u) + D * ((I + 1u) / 2u);
    uint64_t mx_scales = 2u * I * ((D + 31u) / 32u) + D * ((I + 31u) / 32u);
    CHECK(qwen_slot_resident_bytes(&c, &s) == mx_weights + mx_scales,
          "MXFP4 slot residency bytes mismatch: got %llu want %llu",
          (unsigned long long)qwen_slot_resident_bytes(&c, &s),
          (unsigned long long)(mx_weights + mx_scales));

    s.fmt = 16;
    CHECK(qwen_slot_resident_bytes(&c, &s) == 3u * D * I * 2u,
          "BF16 slot residency bytes mismatch");

    s.fmt = 8;
    CHECK(qwen_slot_resident_bytes(&c, &s) == 3u * D * I + (2u * I + D) * 4u,
          "q8 slot residency bytes mismatch");
}

'''
    t = replace_once(t, "int main(void){\n", test + "int main(void){\n    test_slot_resident_bytes();\n", "Qwen test main")
p.write_text(t)

# Stage our own temporary files for deletion so the workflow's later commit
# contains only the real source/test changes and leaves the feature branch clean.
subprocess.run([
    "git", "rm",
    "tools/apply_qwen_mxfp4_cache_accounting.py",
    ".github/workflows/qwen-mxfp4-accounting-stage.yml",
], check=True)
