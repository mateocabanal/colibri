#!/usr/bin/env python3
"""Apply the small qwen_moe.c/Makefile integration edits fail-closed.

Temporary branch-only helper: the GitHub connector replaces whole files, so
using exact source transforms avoids round-tripping qwen_moe.c's ~166 KiB body.
The workflow that invokes this script removes both itself and this file before
committing the real change.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "qwen_moe.c"
MAKE = ROOT / "Makefile"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


q = QWEN.read_text()
q = replace_once(
    q,
    '#include "route_trace.h"\n',
    '#include "route_trace.h"\n#include "qwen_prefix_cache.h"\n',
    "include qwen_prefix_cache",
)
q = replace_once(
    q,
    '    float **gdn_conv;              /* [n_layers][conv_dim*(k-1)] conv state */\n',
    '    float **gdn_conv;              /* [n_layers][conv_dim*(k-1)] conv state */\n'
    '    QwenPrefixCache prefix_cache;  /* exact end-of-prefill hybrid snapshots */\n',
    "Model prefix cache field",
)
q = replace_once(
    q,
    'static int g_kv_f16 = 1;             /* QWEN_KV_F16=0 disables (f32 KV) */\n',
    '''static int g_kv_f16 = 1;             /* QWEN_KV_F16=0 disables (f32 KV) */\n\n'''
    '''static QwenPrefixStateView qwen_prefix_state_view(Model *m){\n'''
    '''    QwenPrefixStateView view = {\n'''
    '''        .layer_count = m->c.n_layers,\n'''
    '''        .layer_is_gdn = m->c.layer_is_gdn,\n'''
    '''        .n_kv_heads = m->c.n_kv_heads,\n'''
    '''        .head_dim = m->c.head_dim,\n'''
    '''        .max_t = m->max_t,\n'''
    '''        .kv_f16 = g_kv_f16,\n'''
    '''        .K = m->K, .V = m->V, .K16 = m->K16, .V16 = m->V16,\n'''
    '''        .gdn_S = m->gdn_S, .gdn_conv = m->gdn_conv,\n'''
    '''        .gdn_state_elems = gdn_state_count(&m->c),\n'''
    '''        .gdn_conv_elems = gdn_conv_count(&m->c),\n'''
    '''    };\n'''
    '''    return view;\n'''
    '''}\n''',
    "state view helper",
)
q = replace_once(
    q,
    ''' * v1 scope, same as olmoe: one request in flight, full re-prefill every turn,\n'''
    ''' * no cross-request KV reuse. The payload arrives already rendered by\n'''
    ''' * openai_server.py's render_chat_qwen — this engine tokenizes it as-is.\n'''
    ''' * Expert-cache contents, route counts, and immutable weights persist.  Before\n'''
    ''' * each prefill we clear GDN recurrence/conv state; KV storage need not be\n'''
    ''' * cleared because position-zero prefill overwrites every position attention\n'''
    ''' * reads for the new request. */\n''',
    ''' * One request is in flight at a time. With QWEN_PREFIX_CACHE_MB>0, exact\n'''
    ''' * end-of-prefill snapshots can reuse the longest strict token prefix across\n'''
    ''' * requests; snapshots include both full-attention KV and GDN recurrence/conv\n'''
    ''' * state. A miss keeps the old full re-prefill behavior. The payload arrives\n'''
    ''' * already rendered by openai_server.py's render_chat_qwen and is tokenized\n'''
    ''' * as-is. Expert-cache contents, route counts, and immutable weights persist. */\n''',
    "serve cache comment",
)
q = replace_once(
    q,
    '''    request_state_reset(m);\n'''
    '''    float *logit = step(m, ids, np, 0);\n''',
    '''    QwenPrefixStateView prefix_view = qwen_prefix_state_view(m);\n'''
    '''    int prefix_reused = qwen_prefix_cache_restore(&m->prefix_cache,\n'''
    '''                                                   &prefix_view, ids, np);\n'''
    '''    if (!prefix_reused) request_state_reset(m);\n'''
    '''    float *logit = step(m, ids + prefix_reused, np - prefix_reused,\n'''
    '''                        prefix_reused);\n'''
    '''    /* Capture before decode advances the live state past the prompt. */\n'''
    '''    qwen_prefix_cache_store(&m->prefix_cache, &prefix_view, ids, np);\n''',
    "serve restore/store hook",
)
q = replace_once(
    q,
    '''    tok_free(&T);\n'''
    '''    return rc;\n'''
    '''}\n''',
    '''    qwen_prefix_cache_clear(&m.prefix_cache);\n'''
    '''    tok_free(&T);\n'''
    '''    return rc;\n'''
    '''}\n''',
    "cache cleanup",
)
QWEN.write_text(q)

m = MAKE.read_text()
m = replace_once(
    m,
    'qwen_moe$(EXE): qwen_moe.c st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h $(QMOE_COLI_OBJS) $(MIO_OBJ) $(METAL_OBJ)\n',
    'qwen_moe$(EXE): qwen_moe.c qwen_prefix_cache.h st.h json.h compat.h sample.h tok.h tok_unicode.h tok_unicode_o200k.h omp_tune.h route_trace.h $(QMOE_COLI_OBJS) $(MIO_OBJ) $(METAL_OBJ)\n',
    "qwen make dependency",
)
MAKE.write_text(m)

print("qwen prefix cache integration patch applied")
