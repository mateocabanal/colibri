#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "qwen_moe.c"
HEADER = ROOT / "qwen_prefix_cache.h"
TEST = ROOT / "tests" / "test_qwen_prefix_cache.c"


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one match, found {n}")
    return text.replace(old, new, 1)


h = HEADER.read_text()
h = replace_once(
    h,
    '''static inline size_t qwen_prefix_cache_budget_from_env(void) {\n''',
    '''/* Qwen's legacy RAM_GB policy treats one expert slot per sparse layer as\n'''
    ''' * roughly 2 decimal GB. Reserve the prompt-cache bytes before applying\n'''
    ''' * that same heuristic so an implicit RAM budget remains a total budget.\n'''
    ''' * Explicit CACHE/positional caps are intentionally outside this helper. */\n'''
    '''static inline int qwen_prefix_cache_ram_cap(const char *value,\n'''
    '''                                            size_t prefix_budget_bytes,\n'''
    '''                                            int *valid) {\n'''
    '''    if (valid) *valid = 0;\n'''
    '''    if (!value || !*value) return 0;\n'''
    '''    char *end = NULL;\n'''
    '''    long double gib = strtold(value, &end);\n'''
    '''    /* Keep RAM_GB's historical permissive trailing-text behavior while\n'''
    '''     * rejecting NaN/inf/negative values before any integer conversion. */\n'''
    '''    if (end == value || !(gib > 0.0L) || gib > 1000000.0L) return 0;\n'''
    '''    if (valid) *valid = 1;\n'''
    '''    const long double gb = 1000000000.0L;\n'''
    '''    long double bytes = gib * gb;\n'''
    '''    if (bytes <= (long double)prefix_budget_bytes) return 0;\n'''
    '''    long double slots = (bytes - (long double)prefix_budget_bytes) / (2.0L * gb);\n'''
    '''    if (slots >= (long double)INT_MAX) return INT_MAX;\n'''
    '''    return slots > 0.0L ? (int)slots : 0;\n'''
    '''}\n\n'''
    '''static inline size_t qwen_prefix_cache_budget_from_env(void) {\n''',
    "RAM cap helper",
)
HEADER.write_text(h)

q = QWEN.read_text()
q = replace_once(
    q,
    '''    const char *snap;\n'''
    '''    int cap;\n'''
    '''    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {\n''',
    '''    const char *snap;\n'''
    '''    int cap;\n'''
    '''    int serving = getenv("SERVE") && getenv("SERVE")[0] == '1';\n'''
    '''    if (serving) {\n''',
    "serve flag",
)
q = replace_once(
    q,
    '''    if (cap == 0) {\n'''
    '''        /* Default: ONLY the experts currently in use stay resident — one\n'''
    '''         * cache slot per selected expert per layer (topk). Everything else\n'''
    '''         * is streamed from disk on demand. Raise CACHE/RAM_GB for more. */\n'''
    '''        if (getenv("RAM_GB") && atoi(getenv("RAM_GB")) > 0) {\n'''
    '''            cap = atoi(getenv("RAM_GB")) / 2;\n'''
    '''        } else {\n'''
    '''            Cfg cfg0;\n'''
    '''            if (getenv("COLI_CONFIG")) load_cfg(&cfg0, getenv("COLI_CONFIG"));\n'''
    '''            else load_cfg(&cfg0, snap);   /* config-only pre-pass for topk */\n'''
    '''            cap = cfg0.topk;\n'''
    '''            free(cfg0.layer_is_gdn);\n'''
    '''        }\n'''
    '''    }\n''',
    '''    int explicit_cap = cap != 0;\n'''
    '''    size_t prefix_budget = serving ? qwen_prefix_cache_budget_from_env() : 0;\n'''
    '''    if (cap == 0) {\n'''
    '''        /* Default: ONLY the experts currently in use stay resident — one\n'''
    '''         * cache slot per selected expert per layer (topk). Everything else\n'''
    '''         * is streamed from disk on demand. Raise CACHE/RAM_GB for more.\n'''
    '''         *\n'''
    '''         * RAM_GB is a total host/UMA budget. Its existing expert-residency\n'''
    '''         * heuristic is 2 decimal GB per cache slot/layer, so reserve the\n'''
    '''         * opt-in prompt cache before deriving that cap. */\n'''
    '''        int ram_valid = 0;\n'''
    '''        cap = qwen_prefix_cache_ram_cap(getenv("RAM_GB"), prefix_budget, &ram_valid);\n'''
    '''        if (ram_valid) {\n'''
    '''            if (prefix_budget)\n'''
    '''                fprintf(stderr,\n'''
    '''                        "[QWEN-PREFIX] RAM_GB reserves %.2fMiB for prompt snapshots; expert cap=%d under 2GB/slot heuristic\\n",\n'''
    '''                        (double)prefix_budget / (1024.0 * 1024.0), cap);\n'''
    '''        } else {\n'''
    '''            Cfg cfg0;\n'''
    '''            if (getenv("COLI_CONFIG")) load_cfg(&cfg0, getenv("COLI_CONFIG"));\n'''
    '''            else load_cfg(&cfg0, snap);   /* config-only pre-pass for topk */\n'''
    '''            cap = cfg0.topk;\n'''
    '''            free(cfg0.layer_is_gdn);\n'''
    '''        }\n'''
    '''    } else if (serving && explicit_cap && prefix_budget) {\n'''
    '''        fprintf(stderr,\n'''
    '''                "[QWEN-PREFIX] explicit expert cap=%d; %.2fMiB prompt-cache budget is additive to expert residency\\n",\n'''
    '''                cap, (double)prefix_budget / (1024.0 * 1024.0));\n'''
    '''    }\n''',
    "RAM budget reservation",
)
QWEN.write_text(q)

t = TEST.read_text()
t = replace_once(
    t,
    '''static void test_hard_budget(void) {\n''',
    '''static void test_ram_cap_reservation(void) {\n'''
    '''    int valid = 0;\n'''
    '''    assert(qwen_prefix_cache_ram_cap(NULL, 0, &valid) == 0 && !valid);\n'''
    '''    assert(qwen_prefix_cache_ram_cap("garbage", 0, &valid) == 0 && !valid);\n'''
    '''    assert(qwen_prefix_cache_ram_cap("4", 0, &valid) == 2 && valid);\n'''
    '''    assert(qwen_prefix_cache_ram_cap("4", 1, &valid) == 1 && valid);\n'''
    '''    assert(qwen_prefix_cache_ram_cap("4.5", 500000000, &valid) == 2 && valid);\n'''
    '''    assert(qwen_prefix_cache_ram_cap("1", 2ULL * 1000000000ULL, &valid) == 0 && valid);\n'''
    '''}\n\n'''
    '''static void test_hard_budget(void) {\n''',
    "RAM helper unit",
)
t = replace_once(
    t,
    '''    test_restore(0);\n'''
    '''    test_hard_budget();\n''',
    '''    test_restore(0);\n'''
    '''    test_ram_cap_reservation();\n'''
    '''    test_hard_budget();\n''',
    "call RAM helper unit",
)
TEST.write_text(t)
print("qwen RAM-aware prefix budget patch applied")
