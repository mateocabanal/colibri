#!/usr/bin/env python3
from pathlib import Path

p = Path('c/qwen_moe.c')
s = p.read_text()

def one(old, new, label):
    global s
    if new in s:
        return
    if old not in s:
        raise SystemExit(f'missing anchor: {label}')
    s = s.replace(old, new, 1)

one(
    '    uint64_t prefetch_misses;      /* loads triggered by lookahead prefetch */\n',
    '    uint64_t prefetch_misses;      /* loads triggered by lookahead prefetch */\n'
    '    uint64_t prof_expert_requests; /* logical routed expert applications */\n'
    '    uint64_t prof_expert_loads;    /* physical expert loads from storage */\n',
    'model profile counters')

one(
'''    QPROF_METAL_KERNEL,\n    QPROF_COUNT\n};\nenum {\n    QPC_EXPERT_REQUESTS = 0,\n    QPC_EXPERT_HITS,\n    QPC_EXPERT_MISSES,\n    QPC_PREFETCH_MISSES,\n    QPC_COUNT\n};\n''',
'''    QPROF_METAL_KERNEL,\n    QPROF_METAL_MOE_SETUP,\n    QPROF_METAL_MOE_WAIT,\n    QPROF_METAL_MOE_KERNEL,\n    QPROF_METAL_MOE_SCATTER,\n    QPROF_COUNT\n};\nenum {\n    QPC_ROUTED_EXPERT_REQUESTS = 0,\n    QPC_EXPERT_LOADS,\n    QPC_CACHE_HITS,\n    QPC_CACHE_MISSES,\n    QPC_PREFETCH_MISSES,\n    QPC_METAL_MOE_BLOCKS,\n    QPC_METAL_MOE_FALLBACKS,\n    QPC_METAL_MOE_EXPERTS,\n    QPC_COUNT\n};\n''',
    'profile enums')

one(
'''    [QPROF_METAL_KERNEL]  = {"metal_kernel", 0},\n};\nstatic const ColiProfileCounterDef qprof_counters[QPC_COUNT] = {\n    [QPC_EXPERT_REQUESTS] = {"expert_requests"},\n    [QPC_EXPERT_HITS] = {"expert_hits"},\n    [QPC_EXPERT_MISSES] = {"expert_misses"},\n    [QPC_PREFETCH_MISSES] = {"prefetch_misses"},\n};\n''',
'''    [QPROF_METAL_KERNEL]  = {"metal_kernel", 0},\n    [QPROF_METAL_MOE_SETUP]   = {"metal_moe_setup", 0},\n    [QPROF_METAL_MOE_WAIT]    = {"metal_moe_wait", 0},\n    [QPROF_METAL_MOE_KERNEL]  = {"metal_moe_kernel", 0},\n    [QPROF_METAL_MOE_SCATTER] = {"metal_moe_scatter", 0},\n};\nstatic const ColiProfileCounterDef qprof_counters[QPC_COUNT] = {\n    [QPC_ROUTED_EXPERT_REQUESTS] = {"routed_expert_requests"},\n    [QPC_EXPERT_LOADS] = {"expert_loads"},\n    [QPC_CACHE_HITS] = {"cache_hits"},\n    [QPC_CACHE_MISSES] = {"cache_misses"},\n    [QPC_PREFETCH_MISSES] = {"prefetch_misses"},\n    [QPC_METAL_MOE_BLOCKS] = {"metal_moe_blocks"},\n    [QPC_METAL_MOE_FALLBACKS] = {"metal_moe_fallbacks"},\n    [QPC_METAL_MOE_EXPERTS] = {"metal_moe_experts"},\n};\n''',
    'profile definitions')

one(
'''    coli_profile_counter_set(&g_qprof, QPC_EXPERT_REQUESTS, m->hits + m->miss);\n    coli_profile_counter_set(&g_qprof, QPC_EXPERT_HITS, m->hits);\n    coli_profile_counter_set(&g_qprof, QPC_EXPERT_MISSES, m->miss);\n    coli_profile_counter_set(&g_qprof, QPC_PREFETCH_MISSES, m->prefetch_misses);\n#ifdef COLI_METAL\n    uint64_t encode = 0, submit = 0, wait = 0, kernel = 0;\n    coli_metal_profile_get(&encode, &submit, &wait, &kernel);\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_ENCODE, encode);\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_SUBMIT, submit);\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_WAIT, wait);\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_KERNEL, kernel);\n#endif\n''',
'''    coli_profile_counter_set(&g_qprof, QPC_ROUTED_EXPERT_REQUESTS, m->prof_expert_requests);\n    coli_profile_counter_set(&g_qprof, QPC_EXPERT_LOADS, m->prof_expert_loads);\n    coli_profile_counter_set(&g_qprof, QPC_CACHE_HITS, m->hits);\n    coli_profile_counter_set(&g_qprof, QPC_CACHE_MISSES, m->miss);\n    coli_profile_counter_set(&g_qprof, QPC_PREFETCH_MISSES, m->prefetch_misses);\n#ifdef COLI_METAL\n    uint64_t encode = 0, submit = 0, wait = 0, kernel = 0;\n    coli_metal_profile_get(&encode, &submit, &wait, &kernel);\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_ENCODE, encode);\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_SUBMIT, submit);\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_WAIT, wait);\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_KERNEL, kernel);\n\n    double moe_setup = 0.0, moe_wait = 0.0, moe_scatter = 0.0;\n    uint64_t moe_ok = 0, moe_fb = 0, moe_experts = 0;\n    coli_metal_moe_times(&moe_setup, &moe_wait, &moe_scatter);\n    coli_metal_moe_counts(&moe_ok, &moe_fb, &moe_experts);\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_MOE_SETUP, qprof_s_to_ns(moe_setup));\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_MOE_WAIT, qprof_s_to_ns(moe_wait));\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_MOE_KERNEL,\n                           qprof_s_to_ns(coli_metal_moe_kernel_time()));\n    coli_profile_phase_set(&g_qprof, QPROF_METAL_MOE_SCATTER, qprof_s_to_ns(moe_scatter));\n    coli_profile_counter_set(&g_qprof, QPC_METAL_MOE_BLOCKS, moe_ok);\n    coli_profile_counter_set(&g_qprof, QPC_METAL_MOE_FALLBACKS, moe_fb);\n    coli_profile_counter_set(&g_qprof, QPC_METAL_MOE_EXPERTS, moe_experts);\n#endif\n''',
    'qprof sync')

one(
'''static void load_expert(Model *m, int layer, int eid, Slot *s){\n    if (coli_mode) { load_expert_coli(m, layer, eid, s); return; }\n''',
'''static void load_expert(Model *m, int layer, int eid, Slot *s){\n    m->prof_expert_loads++;\n    if (coli_mode) { load_expert_coli(m, layer, eid, s); return; }\n''',
    'physical load counter')

one(
'''static void moe_token(Model *m, Layer *l, int layer, const float *x, float *out){\n    Cfg *c = &m->c; int E = c->n_experts, K = c->topk, D = c->hidden;\n''',
'''static void moe_token(Model *m, Layer *l, int layer, const float *x, float *out){\n    Cfg *c = &m->c; int E = c->n_experts, K = c->topk, D = c->hidden;\n    m->prof_expert_requests += (uint64_t)K;\n''',
    'decode logical requests')

one(
'''static void moe_batch(Model *m, Layer *l, int layer, const float *xs, int C, float *out){\n    Cfg *c = &m->c; int E = c->n_experts, K = c->topk, D = c->hidden;\n    double t0 = now_s();\n''',
'''static void moe_batch(Model *m, Layer *l, int layer, const float *xs, int C, float *out){\n    Cfg *c = &m->c; int E = c->n_experts, K = c->topk, D = c->hidden;\n    m->prof_expert_requests += (uint64_t)C * (uint64_t)K;\n    double t0 = now_s();\n''',
    'prefill logical requests')

p.write_text(s)
