#!/usr/bin/env python3
"""Fix Qwen MetalIO prefill-wave bounds on partial final waves.

The async arena pipeline used a wave-level condition to decide whether a next
wave existed, then indexed the next-wave expert for every slot in the current
wave. When the next wave was partial, later slots read past nuniq and could
turn uninitialized memory into impossible expert IDs (for example 19894).

This patch is anchor-checked and idempotent. It also uses the runtime WAVE
value instead of the legacy QWEN_ARENA_CAP constant so QWEN_ARENA_WAVE remains
correct when overridden.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "c" / "qwen_moe.c"


def replace_once(old: str, new: str, marker: str) -> None:
    text = QWEN.read_text()
    if marker in text:
        print(f"[prefill-wave-bounds] already patched: {marker}")
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"c/qwen_moe.c: expected exactly one anchor for {marker}, found {count}\n"
            f"anchor:\n{old[:900]}"
        )
    QWEN.write_text(text.replace(old, new, 1))
    print(f"[prefill-wave-bounds] patched: {marker}")


replace_once(
    "                if (wb >= QWEN_ARENA_CAP) {\n",
    "                if (wb >= WAVE) {\n",
    "if (wb >= WAVE)",
)

old = '''            if (pool && wb + QWEN_ARENA_CAP < nuniq) {
                /* pipeline the next wave's expert into the SAME buffer right
                 * after the apply finished reading it; persist the full
                 * descriptor so the next wave's skip path finds the
                 * pointers + mio_resident state. */
                int ne = uniq[wb + QWEN_ARENA_CAP + a];
'''
new = '''            int next_i = wb + WAVE + a;
            if (pool && next_i < nuniq) {
                /* Pipeline only when THIS physical slot has a corresponding
                 * expert in the next wave. A partial final wave has fewer
                 * experts than the current wave, so a wave-level existence
                 * check is insufficient and would read uniq[] out of bounds. */
                int ne = uniq[next_i];
'''
replace_once(old, new, "int next_i = wb + WAVE + a;")

print("[prefill-wave-bounds] partial-wave pipeline bounds are safe")
print("[prefill-wave-bounds] next: cd c && make qwen_moe METAL=1")
