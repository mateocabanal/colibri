#!/usr/bin/env python3
"""Install the validated Qwen Apple8 runtime and make it the macOS default.

This is the production promotion step for PR #145. It composes the already
hardware-validated integration patchers, then flips the Apple runtime defaults
from opt-in to default-on while preserving explicit =0 escape hatches.

Validated target: Qwen3.6-35B-A3B.Apple8.raw.coli on 16 GB M2 macOS.
Unsupported layouts/codecs/backends keep the existing canonical/CPU fallbacks.
"""
from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "c" / "qwen_moe.c"

# Compose only the correctness/runtime patches. Diagnostic hash/logit/sync-wave
# instrumentation is deliberately not installed here.
for script in (
    "apply_qwen_apple8_direct_m2.py",
    "fix_qwen_apple8_direct_decode.py",
    "fix_qwen_decode_topk_alias.py",
):
    runpy.run_path(str(ROOT / "tools" / script), run_name="__main__")

text = QWEN.read_text()

replacements = (
    (
        'g_metal_compute = getenv("QWEN_METAL_COMPUTE") ? atoi(getenv("QWEN_METAL_COMPUTE")) : 0;',
        'g_metal_compute = getenv("QWEN_METAL_COMPUTE") ? atoi(getenv("QWEN_METAL_COMPUTE")) : 1;',
        "Metal compute default-on",
    ),
    (
        'g_metal_io = getenv("QWEN_METAL_IO") ? atoi(getenv("QWEN_METAL_IO")) : 0;',
        'g_metal_io = getenv("QWEN_METAL_IO") ? atoi(getenv("QWEN_METAL_IO")) : 1;',
        "MetalIO default-on",
    ),
    (
        'g_apple8_direct = getenv("QWEN_APPLE8_DIRECT") ? atoi(getenv("QWEN_APPLE8_DIRECT")) : 0;',
        'g_apple8_direct = getenv("QWEN_APPLE8_DIRECT") ? atoi(getenv("QWEN_APPLE8_DIRECT")) : 1;',
        "direct Apple8 default-on",
    ),
    (
        'g_chunk = getenv("QWENMOE_CHUNK") ? atoi(getenv("QWENMOE_CHUNK")) : 64;',
        'g_chunk = getenv("QWENMOE_CHUNK") ? atoi(getenv("QWENMOE_CHUNK")) : 48;',
        "validated Apple8 prefill chunk default",
    ),
)

changed = False
for old, new, label in replacements:
    if new in text:
        print(f"[qwen-apple8-default] already enabled: {label}")
        continue
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"c/qwen_moe.c: expected exactly one anchor for {label}, found {count}"
        )
    text = text.replace(old, new, 1)
    changed = True
    print(f"[qwen-apple8-default] enabled: {label}")

if changed:
    QWEN.write_text(text)

print("[qwen-apple8-default] installed")
print("[qwen-apple8-default] defaults: Metal=on MetalIO=on Apple8-direct=on chunk=48")
print("[qwen-apple8-default] opt-outs: QWEN_METAL_COMPUTE=0 QWEN_METAL_IO=0 QWEN_APPLE8_DIRECT=0")
print("[qwen-apple8-default] next: cd c && make qwen_moe METAL=1")
