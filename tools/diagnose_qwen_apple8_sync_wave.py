#!/usr/bin/env python3
"""Add a temporary Qwen MetalIO wave-pipeline diagnostic switch.

Usage after the PR #145 Qwen patchers have been applied:

    python3 tools/diagnose_qwen_apple8_sync_wave.py
    cd c && make qwen_moe METAL=1

Then set QWEN_METAL_IO_PIPELINE=0 to keep the same direct Apple8 Metal compute
and persistent MetalIO wave slots while making expert loads synchronous.  This
isolates async slot pipelining from prefix-cache/state correctness.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "c" / "qwen_moe.c"


def replace_once(old: str, new: str, marker: str) -> None:
    text = QWEN.read_text()
    if marker in text:
        print(f"[apple8-sync-wave] already patched: {marker}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"c/qwen_moe.c: expected exactly one anchor for {marker}, found {count}\n"
            f"anchor:\n{old[:500]}"
        )
    QWEN.write_text(text.replace(old, new, 1))
    print(f"[apple8-sync-wave] patched: {marker}")


replace_once(
    "static int g_mio_async_issue = 0;   /* 1 = exact-demand issue: enqueue, no wait */\n",
    "static int g_mio_async_issue = 0;   /* 1 = exact-demand issue: enqueue, no wait */\n"
    "static int g_mio_wave_pipeline = 1; /* diagnostic: 0 = sync wave loads */\n",
    "g_mio_wave_pipeline",
)

replace_once(
    "    if (pool) g_mio_async_issue = 1;\n",
    "    if (pool && g_mio_wave_pipeline) g_mio_async_issue = 1;\n",
    "pool && g_mio_wave_pipeline) g_mio_async_issue = 1",
)

replace_once(
    "                if (wb >= QWEN_ARENA_CAP) {\n",
    "                if (g_mio_wave_pipeline && wb >= QWEN_ARENA_CAP) {\n",
    "g_mio_wave_pipeline && wb >= QWEN_ARENA_CAP",
)

replace_once(
    "            if (pool && wb + QWEN_ARENA_CAP < nuniq) {\n",
    "            if (pool && g_mio_wave_pipeline && wb + QWEN_ARENA_CAP < nuniq) {\n",
    "pool && g_mio_wave_pipeline && wb + QWEN_ARENA_CAP < nuniq",
)

old = '''    g_arena_wave = getenv("QWEN_ARENA_WAVE") ? atoi(getenv("QWEN_ARENA_WAVE")) : QWEN_ARENA_CAP;
    if (g_arena_wave < 8 || g_arena_wave > QWEN_ARENA_CAP_MAX) g_arena_wave = QWEN_ARENA_CAP;
'''
new = old + '''#ifdef COLI_METALIO
    g_mio_wave_pipeline = getenv("QWEN_METAL_IO_PIPELINE")
        ? atoi(getenv("QWEN_METAL_IO_PIPELINE")) != 0 : 1;
    if (!g_mio_wave_pipeline)
        fprintf(stderr, "[QWEN-METALIO] async prefill wave pipeline disabled (sync diagnostic)\\n");
#endif
'''
replace_once(old, new, "async prefill wave pipeline disabled (sync diagnostic)")

print("[apple8-sync-wave] QWEN_METAL_IO_PIPELINE=0 is ready")
print("[apple8-sync-wave] next: cd c && make qwen_moe METAL=1")
