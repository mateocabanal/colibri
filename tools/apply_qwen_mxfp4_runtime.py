#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)


qpath = Path("c/qwen_moe.c")
q = qpath.read_text()

if '"mxfp4_expert.h"' not in q:
    q = replace_once(
        q,
        '#include "coli_executor.h"   /* COLLACOLI: COLI package backend */\n',
        '#include "coli_executor.h"   /* COLLACOLI: COLI package backend */\n'
        '#include "mxfp4_expert.h"\n'
        '#include "mxfp4_runtime.h"\n',
        "MXFP4 includes",
    )

if "7=MXFP4" not in q:
    q = replace_once(
        q,
        '    int fmt;                       /* 0=f32, 16=BF16, 8=int8, 4=i4-grouped, 5=int3-g64 */\n',
        '    int fmt;                       /* 0=f32, 16=BF16, 8=int8, 7=MXFP4, 4=i4-grouped, 5=int3-g64 */\n',
        "slot format comment",
    )

if "MXFP4 E2M1 packed gate/up/down weights" not in q:
    q = replace_once(
        q,
        '    float *g4s, *u4s, *d4s;        /* packed: per-64-input-group f32 scales */\n',
        '    float *g4s, *u4s, *d4s;        /* packed: per-64-input-group f32 scales */\n'
        '    uint8_t *mxg, *mxu, *mxd;       /* MXFP4 E2M1 packed gate/up/down weights */\n'
        '    uint8_t *mxgs, *mxus, *mxds;    /* MXFP4 raw E8M0 scales, one per 32 columns */\n',
        "slot MXFP4 pointers",
    )

if "slot_alloc_mxfp4(Slot *s" not in q:
    q = replace_once(
        q,
        'static void slot_alloc_bf16(Slot *s, int64_t ng, int64_t nd); /* exact COLI residency */\n',
        'static void slot_alloc_bf16(Slot *s, int64_t ng, int64_t nd); /* exact COLI residency */\n'
        'static void slot_alloc_mxfp4(Slot *s, const ColiMxfp4ExpertLayout *layout);\n',
        "MXFP4 allocator declaration",
    )

if "read MXFP4 expert" not in q:
    start = q.index("static void load_expert_coli(Model *m, int layer, int eid, Slot *s){")
    end = q.index("/* ============================================ COLLACOLI end", start)
    new_loader = r'''static void load_expert_coli(Model *m, int layer, int eid, Slot *s){
    Cfg *cc = &m->c;
    int64_t ng = (int64_t)cc->moe_inter * cc->hidden;
    int64_t nd = (int64_t)cc->hidden * cc->moe_inter;
    const ColiRecordInfo *erec = coli_executor_expert(m->coli, layer, eid);
    ColiExpertInfo ei;
    if (!erec || erec->codec != COLI_CSF_CODEC_NONE ||
        coli_executor_expert_info(m->coli, layer, eid, &ei, NULL, 0)) {
        fprintf(stderr, "qwen coli: missing/invalid expert (%d,%d)\n", layer, eid); exit(1);
    }
    int gi = coli_emx(&ei, 1), ui = coli_emx(&ei, 2), di = coli_emx(&ei, 3);
    if (gi < 0 || ui < 0 || di < 0) {
        fprintf(stderr, "qwen coli: expert (%d,%d) is missing gate/up/down roles\n", layer, eid); exit(1);
    }

    /* Keep MXFP4 compressed in the streamed cache. The generic loader validates
     * the compiler/runtime ABI and reads only the six executable E2M1/E8M0 spans. */
    if (ei.matrices[gi].math_format == COLI_CSF_MATH_MXFP4_E2M1 ||
        ei.matrices[ui].math_format == COLI_CSF_MATH_MXFP4_E2M1 ||
        ei.matrices[di].math_format == COLI_CSF_MATH_MXFP4_E2M1) {
        ColiMxfp4ExpertLayout layout;
        char err[512] = {0};
        if (coli_mxfp4_expert_validate_info(&ei, cc->hidden, cc->moe_inter,
                                            &layout, err, sizeof(err))) {
            fprintf(stderr, "qwen coli: expert (%d,%d) MXFP4 layout broken: %s\n",
                    layer, eid, err[0] ? err : "validation failed");
            exit(1);
        }
        slot_alloc_mxfp4(s, &layout);
        ColiMxfp4ExpertBuffers buffers = {
            .gate_weights = s->mxg, .gate_weight_capacity = layout.gate_weight_bytes,
            .gate_scales = s->mxgs, .gate_scale_capacity = layout.gate_scale_bytes,
            .up_weights = s->mxu, .up_weight_capacity = layout.up_weight_bytes,
            .up_scales = s->mxus, .up_scale_capacity = layout.up_scale_bytes,
            .down_weights = s->mxd, .down_weight_capacity = layout.down_weight_bytes,
            .down_scales = s->mxds, .down_scale_capacity = layout.down_scale_bytes,
        };
        const ColiPackage *pkg = coli_executor_package(m->coli);
        if (coli_mxfp4_expert_load(pkg, erec, cc->hidden, cc->moe_inter,
                                   &buffers, &layout, err, sizeof(err))) {
            fprintf(stderr, "qwen coli: read MXFP4 expert (%d,%d) failed: %s\n",
                    layer, eid, err[0] ? err : "read failed");
            exit(1);
        }
        s->fmt = 7;
        s->pinned = 0;
        return;
    }

    if (!coli_exact_bf16_matrix(&ei.matrices[gi], cc->moe_inter, cc->hidden) ||
        !coli_exact_bf16_matrix(&ei.matrices[ui], cc->moe_inter, cc->hidden) ||
        !coli_exact_bf16_matrix(&ei.matrices[di], cc->hidden, cc->moe_inter)) {
        fprintf(stderr, "qwen coli: expert (%d,%d) exact BF16 layout broken\n", layer, eid); exit(1);
    }
    slot_alloc_bf16(s, ng, nd); s->fmt = 16; s->pinned = 0;
    const ColiPackage *pkg = coli_executor_package(m->coli);
    if (coli_package_read_range(pkg, erec, ei.matrices[gi].weight_offset,
                                s->bgu, (size_t)ng * 2, NULL, 0) ||
        coli_package_read_range(pkg, erec, ei.matrices[ui].weight_offset,
                                s->bgu + ng, (size_t)ng * 2, NULL, 0) ||
        coli_package_read_range(pkg, erec, ei.matrices[di].weight_offset,
                                s->bd, (size_t)nd * 2, NULL, 0)) {
        fprintf(stderr, "qwen coli: read expert (%d,%d) failed\n", layer, eid); exit(1);
    }
}
'''
    q = q[:start] + new_loader + q[end:]

if "static void slot_alloc_mxfp4(Slot *s, const ColiMxfp4ExpertLayout *layout){" not in q:
    allocator = r'''static void slot_alloc_mxfp4(Slot *s, const ColiMxfp4ExpertLayout *layout){
    if (s->mxg) return;
    size_t weight_bytes = 0, scale_bytes = 0;
    if (!size_add_ok(layout->gate_weight_bytes, layout->up_weight_bytes, &weight_bytes) ||
        !size_add_ok(weight_bytes, layout->down_weight_bytes, &weight_bytes) ||
        !size_add_ok(layout->gate_scale_bytes, layout->up_scale_bytes, &scale_bytes) ||
        !size_add_ok(scale_bytes, layout->down_scale_bytes, &scale_bytes)) {
        fprintf(stderr, "expert MXFP4 allocation overflows\n"); exit(1);
    }
    s->mxg = malloc(weight_bytes);
    s->mxgs = malloc(scale_bytes);
    if ((!s->mxg && weight_bytes) || (!s->mxgs && scale_bytes)) {
        fprintf(stderr, "OOM MXFP4 expert (%zu weight + %zu scale bytes)\n",
                weight_bytes, scale_bytes);
        exit(1);
    }
    s->mxu = s->mxg + layout->gate_weight_bytes;
    s->mxd = s->mxu + layout->up_weight_bytes;
    s->mxus = s->mxgs + layout->gate_scale_bytes;
    s->mxds = s->mxus + layout->up_scale_bytes;
    s->pinned = 0;
    s->fmt = 7;
}

'''
    anchor = "static void slot_alloc_q8(Model *m, Slot *s){"
    pos = q.index(anchor)
    q = q[:pos] + allocator + q[pos:]

if "coli_mxfp4_swiglu_expert(acc, x" not in q:
    old = '''static void expert_apply(Model *m, Slot *s, const float *x, float *acc){
    Cfg *c = &m->c; int I = c->moe_inter, D = c->hidden;
    if (s->fmt == 4 || s->fmt == 5) {
'''
    new = '''static void expert_apply(Model *m, Slot *s, const float *x, float *acc){
    Cfg *c = &m->c; int I = c->moe_inter, D = c->hidden;
    if (s->fmt == 7) {
        float *gate = falloc(I), *up = falloc(I), *h = falloc(I), *y = falloc(D);
        coli_mxfp4_swiglu_expert(acc, x,
                                 s->mxg, s->mxgs, s->mxu, s->mxus,
                                 s->mxd, s->mxds, 1, D, I, 1.0f,
                                 gate, up, h, y);
        free(gate); free(up); free(h); free(y);
    } else if (s->fmt == 4 || s->fmt == 5) {
'''
    q = replace_once(q, old, new, "decode MXFP4 dispatch")

if "coli_mxfp4_matmul(gate, xscratch" not in q:
    batch_start = q.index("static void moe_batch(")
    anchor = "            if (s->fmt == 4 || s->fmt == 5) {\n"
    pos = q.index(anchor, batch_start)
    branch = '''            if (s->fmt == 7) {
                float *gate = falloc((int64_t)st * I), *up = falloc((int64_t)st * I);
                float *h = falloc((int64_t)st * I);
                coli_mxfp4_matmul(gate, xscratch, s->mxg, s->mxgs, st, D, I);
                coli_mxfp4_matmul(up, xscratch, s->mxu, s->mxus, st, D, I);
                for (int r = 0; r < st; r++)
                    for (int i = 0; i < I; i++)
                        h[(int64_t)r * I + i] = silu(gate[(int64_t)r * I + i]) * up[(int64_t)r * I + i];
                coli_mxfp4_matmul(yscratch, h, s->mxd, s->mxds, st, I, D);
                free(gate); free(up); free(h);
            } else if (s->fmt == 4 || s->fmt == 5) {
'''
    q = q[:pos] + branch + q[pos + len(anchor):]

# Transient arena slots own MXFP4 slabs just like the existing pread-backed formats.
if "free(s->mxg); free(s->mxgs);" not in q:
    q = q.replace(
        '        free(s->gu); free(s->bgu); free(s->g); free(s->g4); free(s->g4s);\n'
        '        s->gu = NULL; s->bgu = NULL; s->bd = NULL; s->g = NULL; s->g4 = NULL; s->g4s = NULL;\n',
        '        free(s->gu); free(s->bgu); free(s->g); free(s->g4); free(s->g4s);\n'
        '        free(s->mxg); free(s->mxgs);\n'
        '        s->gu = NULL; s->bgu = NULL; s->bd = NULL; s->g = NULL; s->g4 = NULL; s->g4s = NULL;\n'
        '        s->mxg = s->mxu = s->mxd = NULL; s->mxgs = s->mxus = s->mxds = NULL;\n',
        1,
    )
    q = q.replace(
        '            free(s->gu); free(s->bgu); free(s->g); free(s->gs); free(s->g4); free(s->g4s);\n',
        '            free(s->gu); free(s->bgu); free(s->g); free(s->gs); free(s->g4); free(s->g4s);\n'
        '            free(s->mxg); free(s->mxgs);\n',
        1,
    )

qpath.write_text(q)

mpath = Path("c/Makefile")
m = mpath.read_text()
if "mxfp4_expert.o mxfp4_runtime.o" not in m:
    m = replace_once(
        m,
        "QMOE_COLI_OBJS = coli_executor.o coli_format.o\n",
        "QMOE_COLI_OBJS = coli_executor.o coli_format.o mxfp4_expert.o mxfp4_runtime.o\n",
        "Qwen COLI objects",
    )
if "route_trace.h mxfp4_expert.h mxfp4_runtime.h quant.h $(QMOE_COLI_OBJS)" not in m:
    m = replace_once(
        m,
        "qwen_moe$(EXE): qwen_moe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h $(QMOE_COLI_OBJS) $(MIO_OBJ) $(METAL_OBJ)\n",
        "qwen_moe$(EXE): qwen_moe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h mxfp4_expert.h mxfp4_runtime.h quant.h $(QMOE_COLI_OBJS) $(MIO_OBJ) $(METAL_OBJ)\n",
        "Qwen target dependencies",
    )
if "test_qwen_moe.c qwen_moe.c st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h sample.h omp_tune.h route_trace.h mxfp4_expert.h" not in m:
    m = replace_once(
        m,
        "tests/test_qwen_moe$(EXE): tests/test_qwen_moe.c qwen_moe.c st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h sample.h omp_tune.h route_trace.h $(QMOE_COLI_OBJS)\n",
        "tests/test_qwen_moe$(EXE): tests/test_qwen_moe.c qwen_moe.c st.h json.h tok.h tok_unicode.h tok_unicode_o200k.h compat.h sample.h omp_tune.h route_trace.h mxfp4_expert.h mxfp4_runtime.h quant.h $(QMOE_COLI_OBJS)\n",
        "Qwen test dependencies",
    )
mpath.write_text(m)

tpath = Path("c/tests/test_qwen_moe.c")
t = tpath.read_text()
if "test_mxfp4_expert_dispatch" not in t:
    test = r'''
static void test_mxfp4_expert_dispatch(void){
    Model m;
    Slot s;
    ColiMxfp4ExpertLayout layout = {
        .gate_weight_bytes = 2, .gate_scale_bytes = 2,
        .up_weight_bytes = 2, .up_scale_bytes = 2,
        .down_weight_bytes = 2, .down_scale_bytes = 2,
        .resident_bytes = 12,
    };
    memset(&m, 0, sizeof(m));
    memset(&s, 0, sizeof(s));
    m.c.hidden = 2;
    m.c.moe_inter = 2;
    slot_alloc_mxfp4(&s, &layout);
    CHECK(s.fmt == 7, "MXFP4 slot format is %d, want 7", s.fmt);
    CHECK(s.mxu == s.mxg + 2 && s.mxd == s.mxg + 4,
          "MXFP4 weight spans are not contiguous");
    CHECK(s.mxus == s.mxgs + 2 && s.mxds == s.mxgs + 4,
          "MXFP4 scale spans are not contiguous");

    /* E2M1 code 2 is +1.0; low nibble is the even column. Three identity
     * matrices make expert_apply(x) = silu(x) * x elementwise. */
    const uint8_t ident[2] = { 0x02, 0x20 };
    memcpy(s.mxg, ident, 2); memcpy(s.mxu, ident, 2); memcpy(s.mxd, ident, 2);
    memset(s.mxgs, 127, 2); memset(s.mxus, 127, 2); memset(s.mxds, 127, 2);
    float x[2] = { 1.f, 2.f }, out[2] = { 0.f, 0.f };
    expert_apply(&m, &s, x, out);
    float want0 = silu(1.f), want1 = silu(2.f) * 2.f;
    CHECK(fabsf(out[0] - want0) < 1e-5f,
          "MXFP4 dispatch out[0]=%g want %g", out[0], want0);
    CHECK(fabsf(out[1] - want1) < 1e-5f,
          "MXFP4 dispatch out[1]=%g want %g", out[1], want1);
    free(s.mxg); free(s.mxgs);
}

'''
    t = replace_once(t, "int main(void){\n", test + "int main(void){\n    test_mxfp4_expert_dispatch();\n", "Qwen unit-test main")
tpath.write_text(t)
