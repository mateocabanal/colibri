#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "coli_v4_prefix_cache.h"
CACHE = ROOT / "coli_v4_prefix_cache.c"
GEN = ROOT / "coli_v4_generate_stats_cache.c"
BENCH = ROOT / "tools" / "benchmark_v4_prefix_cache.c"


def rep(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match in {path}, found {count}")
    path.write_text(text.replace(old, new, 1))


rep(HEADER,
'''    uint64_t restore_bytes;\n    uint64_t restore_ns;\n''',
'''    uint64_t restore_bytes;\n    uint64_t restore_ns;\n    uint64_t store_bytes;\n    uint64_t store_ns;\n''',
"public store metrics")

rep(CACHE,
'''    uint64_t restore_bytes;\n    uint64_t restore_ns;\n} ColiV4PrefixCache;\n''',
'''    uint64_t restore_bytes;\n    uint64_t restore_ns;\n    uint64_t store_bytes;\n    uint64_t store_ns;\n} ColiV4PrefixCache;\n''',
"cache store metrics")

rep(CACHE,
'''void coli_v4_prefix_cache_store(ColiV4Session *session) {\n    prefix_cache_init();\n''',
'''void coli_v4_prefix_cache_store(ColiV4Session *session) {\n    prefix_cache_init();\n'''
'''    uint64_t began = coli_v4_profile_now();\n''',
"store timer start")

rep(CACHE,
'''            g_prefix_cache.resident_bytes += entry->bytes;\n            g_prefix_cache.stores++;\n            inserted = 1;\n''',
'''            g_prefix_cache.resident_bytes += entry->bytes;\n            g_prefix_cache.stores++;\n'''
'''            g_prefix_cache.store_bytes += (uint64_t)entry->bytes;\n'''
'''            g_prefix_cache.store_ns += coli_v4_profile_now() - began;\n'''
'''            inserted = 1;\n''',
"successful store accounting")

rep(CACHE,
'''    stats->restore_bytes = g_prefix_cache.restore_bytes;\n    stats->restore_ns = g_prefix_cache.restore_ns;\n    stats->resident_bytes = g_prefix_cache.resident_bytes;\n''',
'''    stats->restore_bytes = g_prefix_cache.restore_bytes;\n    stats->restore_ns = g_prefix_cache.restore_ns;\n    stats->store_bytes = g_prefix_cache.store_bytes;\n    stats->store_ns = g_prefix_cache.store_ns;\n    stats->resident_bytes = g_prefix_cache.resident_bytes;\n''',
"stats export")

rep(GEN,
'''        uint64_t stores = delta_u64(after.stores, before.stores);\n        uint64_t evictions = delta_u64(after.evictions, before.evictions);\n''',
'''        uint64_t stores = delta_u64(after.stores, before.stores);\n        uint64_t store_bytes = delta_u64(after.store_bytes, before.store_bytes);\n        uint64_t store_ns = delta_u64(after.store_ns, before.store_ns);\n        uint64_t evictions = delta_u64(after.evictions, before.evictions);\n''',
"generate store deltas")

rep(GEN,
'''                "restore_ms=%.3f stores=%llu evictions=%llu entries=%zu "\n                "resident_bytes=%zu budget_bytes=%zu\\n",\n''',
'''                "restore_ms=%.3f stores=%llu store_bytes=%llu store_ms=%.3f "\n                "evictions=%llu entries=%zu resident_bytes=%zu budget_bytes=%zu\\n",\n''',
"generate store log format")

rep(GEN,
'''                restore_ns / 1.0e6,\n                (unsigned long long)stores,\n                (unsigned long long)evictions,\n''',
'''                restore_ns / 1.0e6,\n                (unsigned long long)stores,\n                (unsigned long long)store_bytes,\n                store_ns / 1.0e6,\n                (unsigned long long)evictions,\n''',
"generate store log args")

rep(BENCH,
'''    uint64_t restore_ns = cache2.restore_ns >= cache1.restore_ns\n        ? cache2.restore_ns - cache1.restore_ns : 0;\n\n''',
'''    uint64_t restore_ns = cache2.restore_ns >= cache1.restore_ns\n        ? cache2.restore_ns - cache1.restore_ns : 0;\n    uint64_t first_store_bytes = cache1.store_bytes >= cache0.store_bytes\n        ? cache1.store_bytes - cache0.store_bytes : 0;\n    uint64_t first_store_ns = cache1.store_ns >= cache0.store_ns\n        ? cache1.store_ns - cache0.store_ns : 0;\n    uint64_t second_store_bytes = cache2.store_bytes >= cache1.store_bytes\n        ? cache2.store_bytes - cache1.store_bytes : 0;\n    uint64_t second_store_ns = cache2.store_ns >= cache1.store_ns\n        ? cache2.store_ns - cache1.store_ns : 0;\n\n''',
"benchmark store deltas")

rep(BENCH,
'''           "\\\"restore_bytes\\\":%llu,\\\"restore_ms\\\":%.6f,"\n           "\\\"cache_resident_bytes\\\":%zu,\\\"cache_budget_bytes\\\":%zu,"\n''',
'''           "\\\"restore_bytes\\\":%llu,\\\"restore_ms\\\":%.6f,"\n           "\\\"first_store_bytes\\\":%llu,\\\"first_store_ms\\\":%.6f,"\n           "\\\"second_store_bytes\\\":%llu,\\\"second_store_ms\\\":%.6f,"\n           "\\\"cache_resident_bytes\\\":%zu,\\\"cache_budget_bytes\\\":%zu,"\n''',
"benchmark JSON store fields")

rep(BENCH,
'''           (unsigned long long)restore_bytes, restore_ns / 1.0e6,\n           cache2.resident_bytes, cache2.budget_bytes,\n''',
'''           (unsigned long long)restore_bytes, restore_ns / 1.0e6,\n           (unsigned long long)first_store_bytes, first_store_ns / 1.0e6,\n           (unsigned long long)second_store_bytes, second_store_ns / 1.0e6,\n           cache2.resident_bytes, cache2.budget_bytes,\n''',
"benchmark JSON store args")

print("V4 prefix snapshot store timing metrics applied")
