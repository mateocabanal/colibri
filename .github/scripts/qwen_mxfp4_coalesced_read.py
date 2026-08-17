from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    s = p.read_text()
    if old not in s:
        raise SystemExit(f"{path}: expected text not found: {old[:120]!r}")
    if s.count(old) != 1:
        raise SystemExit(f"{path}: expected exactly one match, got {s.count(old)}")
    p.write_text(s.replace(old, new, 1))


# Generic MXFP4 layout exposes a coalesced on-disk span view. Existing six-buffer
# loading remains unchanged and is the fallback for sparse/unusual record layouts.
replace_once(
    "c/mxfp4_expert.h",
    '''    size_t down_weight_bytes;\n    size_t down_scale_bytes;\n    size_t resident_bytes;\n} ColiMxfp4ExpertLayout;\n''',
    '''    size_t down_weight_bytes;\n    size_t down_scale_bytes;\n    size_t resident_bytes;\n\n    /* Smallest record-relative span covering all six executable regions.\n     * The *_span_offset fields are relative to record_span_offset. A runtime\n     * may read record_span_bytes once when padding overhead is acceptable;\n     * the generic six-span loader remains valid for arbitrary layouts. */\n    uint64_t record_span_offset;\n    size_t record_span_bytes;\n    size_t gate_weight_span_offset;\n    size_t gate_scale_span_offset;\n    size_t up_weight_span_offset;\n    size_t up_scale_span_offset;\n    size_t down_weight_span_offset;\n    size_t down_scale_span_offset;\n} ColiMxfp4ExpertLayout;\n''')

replace_once(
    "c/mxfp4_expert.c",
    '''    layout->resident_bytes = total;\n\n    if (info->logical_bytes != (uint64_t)total)\n''',
    '''    layout->resident_bytes = total;\n\n    /* Build a record-relative coalesced span view for runtimes that want one\n     * physical read. coli_package_expert_info() already validates record bounds\n     * and overlap; keep overflow checks here because validate_info() is also a\n     * public helper and can be exercised on caller-constructed metadata. */\n    const uint64_t span_offsets[6] = {\n        gate->weight_offset, gate->scale_offset, up->weight_offset,\n        up->scale_offset, down->weight_offset, down->scale_offset,\n    };\n    const size_t span_sizes[6] = {\n        layout->gate_weight_bytes, layout->gate_scale_bytes,\n        layout->up_weight_bytes, layout->up_scale_bytes,\n        layout->down_weight_bytes, layout->down_scale_bytes,\n    };\n    uint64_t span_lo = UINT64_MAX, span_hi = 0;\n    for (size_t i = 0; i < 6; ++i) {\n        if ((uint64_t)span_sizes[i] > UINT64_MAX - span_offsets[i])\n            return fail(error, error_size, "MXFP4 executable span overflows u64");\n        const uint64_t end = span_offsets[i] + (uint64_t)span_sizes[i];\n        if (span_offsets[i] < span_lo) span_lo = span_offsets[i];\n        if (end > span_hi) span_hi = end;\n    }\n    if (span_hi < span_lo || span_hi - span_lo > (uint64_t)SIZE_MAX)\n        return fail(error, error_size, "MXFP4 coalesced span does not fit size_t");\n    layout->record_span_offset = span_lo;\n    layout->record_span_bytes = (size_t)(span_hi - span_lo);\n#define SET_REL(field, absolute) \\\n    do { \\\n        uint64_t delta = (absolute) - span_lo; \\\n        if (delta > (uint64_t)SIZE_MAX) \\\n            return fail(error, error_size, "MXFP4 span offset does not fit size_t"); \\\n        layout->field = (size_t)delta; \\\n    } while (0)\n    SET_REL(gate_weight_span_offset, gate->weight_offset);\n    SET_REL(gate_scale_span_offset, gate->scale_offset);\n    SET_REL(up_weight_span_offset, up->weight_offset);\n    SET_REL(up_scale_span_offset, up->scale_offset);\n    SET_REL(down_weight_span_offset, down->weight_offset);\n    SET_REL(down_scale_span_offset, down->scale_offset);\n#undef SET_REL\n\n    if (info->logical_bytes != (uint64_t)total)\n''')

# Give the validator fixture realistic compiler-style offsets and assert the
# coalesced view, including shuffled descriptor order.
replace_once(
    "c/tests/test_mxfp4_expert.c",
    '''    info.matrices[0] = matrix(COLI_MXFP4_EXPERT_ROLE_DOWN, 64, 32);\n    info.matrices[1] = matrix(COLI_MXFP4_EXPERT_ROLE_GATE, 32, 64);\n    info.matrices[2] = matrix(COLI_MXFP4_EXPERT_ROLE_UP, 32, 64);\n    info.logical_bytes = 3264;\n''',
    '''    info.matrices[0] = matrix(COLI_MXFP4_EXPERT_ROLE_DOWN, 64, 32);\n    info.matrices[1] = matrix(COLI_MXFP4_EXPERT_ROLE_GATE, 32, 64);\n    info.matrices[2] = matrix(COLI_MXFP4_EXPERT_ROLE_UP, 32, 64);\n    /* Compiler order is gate W/S, up W/S, down W/S; descriptor order is\n     * intentionally shuffled above to prove roles, not indices, drive it. */\n    info.matrices[1].weight_offset = 448;\n    info.matrices[1].scale_offset = 1472;\n    info.matrices[2].weight_offset = 1536;\n    info.matrices[2].scale_offset = 2560;\n    info.matrices[0].weight_offset = 2624;\n    info.matrices[0].scale_offset = 3648;\n    info.logical_bytes = 3264;\n''')
replace_once(
    "c/tests/test_mxfp4_expert.c",
    '''        layout.down_weight_bytes != 1024 || layout.down_scale_bytes != 64 ||\n        layout.resident_bytes != 3264) {\n''',
    '''        layout.down_weight_bytes != 1024 || layout.down_scale_bytes != 64 ||\n        layout.resident_bytes != 3264 ||\n        layout.record_span_offset != 448 || layout.record_span_bytes != 3264 ||\n        layout.gate_weight_span_offset != 0 ||\n        layout.gate_scale_span_offset != 1024 ||\n        layout.up_weight_span_offset != 1088 ||\n        layout.up_scale_span_offset != 2112 ||\n        layout.down_weight_span_offset != 2176 ||\n        layout.down_scale_span_offset != 3200) {\n''')

# Qwen owns one slab for compiler-like contiguous MXFP4 records. Interior
# weight/scale pointers are still exactly what the kernels expect.
replace_once(
    "c/qwen_moe.c",
    '''    uint8_t *mxg, *mxu, *mxd;       /* MXFP4 E2M1 packed gate/up/down weights */\n    uint8_t *mxgs, *mxus, *mxds;    /* MXFP4 raw E8M0 scales, one per 32 columns */\n    uint64_t used;\n''',
    '''    uint8_t *mxg, *mxu, *mxd;       /* MXFP4 E2M1 packed gate/up/down weights */\n    uint8_t *mxgs, *mxus, *mxds;    /* MXFP4 raw E8M0 scales, one per 32 columns */\n    uint8_t *mxbase;                 /* owner for coalesced on-disk MXFP4 slab, else NULL */\n    uint64_t used;\n''')

old_alloc = '''static void slot_alloc_mxfp4(Slot *s, const ColiMxfp4ExpertLayout *layout){\n    if (s->mxg) return;\n    size_t weight_bytes = 0, scale_bytes = 0;\n    if (!size_add_ok(layout->gate_weight_bytes, layout->up_weight_bytes, &weight_bytes) ||\n        !size_add_ok(weight_bytes, layout->down_weight_bytes, &weight_bytes) ||\n        !size_add_ok(layout->gate_scale_bytes, layout->up_scale_bytes, &scale_bytes) ||\n        !size_add_ok(scale_bytes, layout->down_scale_bytes, &scale_bytes)) {\n        fprintf(stderr, "expert MXFP4 allocation overflows\\n"); exit(1);\n    }\n    size_t weight_alloc = moe_slab_bytes(weight_bytes);\n    size_t scale_alloc = moe_slab_bytes(scale_bytes);\n    s->mxg = moe_slab_alloc(weight_alloc);\n    s->mxgs = moe_slab_alloc(scale_alloc);\n    s->mxu = s->mxg + layout->gate_weight_bytes;\n    s->mxd = s->mxu + layout->up_weight_bytes;\n    s->mxus = s->mxgs + layout->gate_scale_bytes;\n    s->mxds = s->mxus + layout->up_scale_bytes;\n    s->pinned = 0;\n    s->fmt = 7;\n#ifdef COLI_METAL\n    if (g_metal_compute) {\n        coli_metal_register(s->mxg, weight_alloc);\n        coli_metal_register(s->mxgs, scale_alloc);\n    }\n#endif\n}\n'''
new_alloc = '''static void slot_alloc_mxfp4(Slot *s, const ColiMxfp4ExpertLayout *layout){\n    if (s->mxg) return;\n\n    /* colic currently emits the six executable regions almost contiguously\n     * (16-byte alignment only). Preserve that physical layout in RAM so one\n     * pread can populate the whole expert and one Metal registration covers\n     * every interior pointer. Keep a conservative padding guard so arbitrary\n     * valid COLIEXPT records still use the generic two-slab/six-read path. */\n    if (layout->record_span_bytes >= layout->resident_bytes &&\n        layout->record_span_bytes - layout->resident_bytes <= 4096) {\n        size_t span_alloc = moe_slab_bytes(layout->record_span_bytes);\n        s->mxbase = moe_slab_alloc(span_alloc);\n        s->mxg  = s->mxbase + layout->gate_weight_span_offset;\n        s->mxgs = s->mxbase + layout->gate_scale_span_offset;\n        s->mxu  = s->mxbase + layout->up_weight_span_offset;\n        s->mxus = s->mxbase + layout->up_scale_span_offset;\n        s->mxd  = s->mxbase + layout->down_weight_span_offset;\n        s->mxds = s->mxbase + layout->down_scale_span_offset;\n        s->pinned = 0;\n        s->fmt = 7;\n#ifdef COLI_METAL\n        if (g_metal_compute) coli_metal_register(s->mxbase, span_alloc);\n#endif\n        return;\n    }\n\n    size_t weight_bytes = 0, scale_bytes = 0;\n    if (!size_add_ok(layout->gate_weight_bytes, layout->up_weight_bytes, &weight_bytes) ||\n        !size_add_ok(weight_bytes, layout->down_weight_bytes, &weight_bytes) ||\n        !size_add_ok(layout->gate_scale_bytes, layout->up_scale_bytes, &scale_bytes) ||\n        !size_add_ok(scale_bytes, layout->down_scale_bytes, &scale_bytes)) {\n        fprintf(stderr, "expert MXFP4 allocation overflows\\n"); exit(1);\n    }\n    size_t weight_alloc = moe_slab_bytes(weight_bytes);\n    size_t scale_alloc = moe_slab_bytes(scale_bytes);\n    s->mxg = moe_slab_alloc(weight_alloc);\n    s->mxgs = moe_slab_alloc(scale_alloc);\n    s->mxu = s->mxg + layout->gate_weight_bytes;\n    s->mxd = s->mxu + layout->up_weight_bytes;\n    s->mxus = s->mxgs + layout->gate_scale_bytes;\n    s->mxds = s->mxus + layout->up_scale_bytes;\n    s->pinned = 0;\n    s->fmt = 7;\n#ifdef COLI_METAL\n    if (g_metal_compute) {\n        coli_metal_register(s->mxg, weight_alloc);\n        coli_metal_register(s->mxgs, scale_alloc);\n    }\n#endif\n}\n'''
replace_once("c/qwen_moe.c", old_alloc, new_alloc)

old_load = '''        slot_alloc_mxfp4(s, &layout);\n        ColiMxfp4ExpertBuffers buffers = {\n            .gate_weights = s->mxg, .gate_weight_capacity = layout.gate_weight_bytes,\n            .gate_scales = s->mxgs, .gate_scale_capacity = layout.gate_scale_bytes,\n            .up_weights = s->mxu, .up_weight_capacity = layout.up_weight_bytes,\n            .up_scales = s->mxus, .up_scale_capacity = layout.up_scale_bytes,\n            .down_weights = s->mxd, .down_weight_capacity = layout.down_weight_bytes,\n            .down_scales = s->mxds, .down_scale_capacity = layout.down_scale_bytes,\n        };\n        const ColiPackage *pkg = coli_executor_package(m->coli);\n        uint32_t read_flags = g_expert_drop ? COLI_CSF_READ_UNCACHED : COLI_CSF_READ_DEFAULT;\n        if (coli_mxfp4_expert_load_ex(pkg, erec, cc->hidden, cc->moe_inter,\n                                      &buffers, &layout, read_flags, err, sizeof(err))) {\n            fprintf(stderr, "qwen coli: read MXFP4 expert (%d,%d) failed: %s\\n",\n                    layer, eid, err[0] ? err : "read failed");\n            exit(1);\n        }\n'''
new_load = '''        slot_alloc_mxfp4(s, &layout);\n        const ColiPackage *pkg = coli_executor_package(m->coli);\n        uint32_t read_flags = g_expert_drop ? COLI_CSF_READ_UNCACHED : COLI_CSF_READ_DEFAULT;\n        int load_rc = 0;\n        if (s->mxbase) {\n            /* Fast path: compiler-style layout, one direct record-range read. */\n            load_rc = coli_package_read_range_ex(pkg, erec, layout.record_span_offset,\n                                                  s->mxbase, layout.record_span_bytes,\n                                                  read_flags, err, sizeof(err));\n        } else {\n            ColiMxfp4ExpertBuffers buffers = {\n                .gate_weights = s->mxg, .gate_weight_capacity = layout.gate_weight_bytes,\n                .gate_scales = s->mxgs, .gate_scale_capacity = layout.gate_scale_bytes,\n                .up_weights = s->mxu, .up_weight_capacity = layout.up_weight_bytes,\n                .up_scales = s->mxus, .up_scale_capacity = layout.up_scale_bytes,\n                .down_weights = s->mxd, .down_weight_capacity = layout.down_weight_bytes,\n                .down_scales = s->mxds, .down_scale_capacity = layout.down_scale_bytes,\n            };\n            load_rc = coli_mxfp4_expert_load_ex(pkg, erec, cc->hidden, cc->moe_inter,\n                                                 &buffers, &layout, read_flags,\n                                                 err, sizeof(err));\n        }\n        if (load_rc) {\n            fprintf(stderr, "qwen coli: read MXFP4 expert (%d,%d) failed: %s\\n",\n                    layer, eid, err[0] ? err : "read failed");\n            exit(1);\n        }\n'''
replace_once("c/qwen_moe.c", old_load, new_load)

p = Path("c/qwen_moe.c")
s = p.read_text()
old_unreg = 'coli_metal_unregister(s->mxg); coli_metal_unregister(s->mxgs);'
new_unreg = 'if (s->mxbase) coli_metal_unregister(s->mxbase); else { coli_metal_unregister(s->mxg); coli_metal_unregister(s->mxgs); }'
if old_unreg not in s:
    raise SystemExit("c/qwen_moe.c: MXFP4 unregister pattern missing")
s = s.replace(old_unreg, new_unreg)
old_free = 'free(s->mxg); free(s->mxgs);'
new_free = 'if (s->mxbase) free(s->mxbase); else { free(s->mxg); free(s->mxgs); }'
if old_free not in s:
    raise SystemExit("c/qwen_moe.c: MXFP4 free pattern missing")
s = s.replace(old_free, new_free)
old_reset = 's->mxg = s->mxu = s->mxd = NULL; s->mxgs = s->mxus = s->mxds = NULL;'
new_reset = 's->mxg = s->mxu = s->mxd = NULL; s->mxgs = s->mxus = s->mxds = NULL; s->mxbase = NULL;'
s = s.replace(old_reset, new_reset)
p.write_text(s)

print("coalesced MXFP4 read patch staged")
