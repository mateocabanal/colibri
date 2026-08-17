#!/usr/bin/env python3
from pathlib import Path

path = Path('c/tools/benchmark_v4_prefix_cache.py')
text = path.read_text(encoding='utf-8')
old = '''    paired_saved = [float(pair["ttft_saved_sec"]) for pair in pairs]\n    paired_speedup = [float(pair["ttft_speedup"]) for pair in pairs]\n    first_baseline = pairs[0]["cache_off"]  # type: ignore[assignment]\n'''
new = '''    paired_saved = [float(pair["ttft_saved_sec"]) for pair in pairs]\n    paired_speedup = [float(pair["ttft_speedup"]) for pair in pairs]\n    cache_restore_ms = [float(pair["cache_on"]["restore_ms"]) for pair in pairs]  # type: ignore[index]\n    cache_first_store_ms = [float(pair["cache_on"]["first_store_ms"]) for pair in pairs]  # type: ignore[index]\n    cache_second_store_ms = [float(pair["cache_on"]["second_store_ms"]) for pair in pairs]  # type: ignore[index]\n    cache_first_store_bytes = [int(pair["cache_on"]["first_store_bytes"]) for pair in pairs]  # type: ignore[index]\n    cache_second_store_bytes = [int(pair["cache_on"]["second_store_bytes"]) for pair in pairs]  # type: ignore[index]\n    first_baseline = pairs[0]["cache_off"]  # type: ignore[assignment]\n'''
if text.count(old) != 1:
    raise SystemExit('summary source anchor changed')
text = text.replace(old, new)
old = '''        "paired_ttft_saved_median_sec": statistics.median(paired_saved),\n        "paired_ttft_speedup_median": statistics.median(paired_speedup),\n        "pairs": pairs,\n'''
new = '''        "paired_ttft_saved_median_sec": statistics.median(paired_saved),\n        "paired_ttft_speedup_median": statistics.median(paired_speedup),\n        "cache_restore_ms_median": statistics.median(cache_restore_ms),\n        "cache_first_store_ms_median": statistics.median(cache_first_store_ms),\n        "cache_second_store_ms_median": statistics.median(cache_second_store_ms),\n        "cache_first_store_bytes_median": statistics.median(cache_first_store_bytes),\n        "cache_second_store_bytes_median": statistics.median(cache_second_store_bytes),\n        "pairs": pairs,\n'''
if text.count(old) != 1:
    raise SystemExit('result anchor changed')
path.write_text(text.replace(old, new), encoding='utf-8')
