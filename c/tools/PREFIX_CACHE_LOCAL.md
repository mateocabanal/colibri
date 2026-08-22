# Local Qwen prefix-cache validation

`benchmark_qwen_prefix_tiers.py` is the local acceptance path for the Qwen
process-RAM prefix cache and persistent SSD prefix cache. It does not depend on
GitHub Actions.

The benchmark runs four phases:

1. cache disabled: cold `P+X`
2. RAM-only: cold `P`, then a same-process `P+X` hit
3. SSD-only: cold `P` to publish a persistent snapshot
4. SSD-only: restart `qwen_moe`, then run `P+X` and require a persistent hit

It checks that the hit matches exactly the seeded prefix-token count and that
RAM-hit and SSD-hit output bytes equal the cache-disabled output. Each run gets a
fresh SSD directory, so the SSD seed starts cold and the final SSD hit cannot be
an old object from an earlier benchmark.

## macOS / Apple Silicon

Build the local engine first:

```sh
cd c
make qwen_moe
```

Inspect the planned run without starting inference:

```sh
COLI_CONFIG=/path/to/Qwen-config \
python3 tools/benchmark_qwen_prefix_tiers.py \
  --binary ./qwen_moe \
  --model /path/to/model.coli \
  --memory-gb 10 \
  --plan
```

Run the short local acceptance case:

```sh
COLI_CONFIG=/path/to/Qwen-config \
python3 tools/benchmark_qwen_prefix_tiers.py \
  --binary ./qwen_moe \
  --model /path/to/model.coli \
  --memory-gb 10 \
  --prefix-tokens 256 \
  --ram-cache-mb 256 \
  --disk-cache-gb 1 \
  --min-prefix-tokens 128 \
  --show-engine-stderr \
  --output ~/qwen-prefix-tiers.json
```

The tool currently forces `QWEN_METAL_COMPUTE=0`. Qwen persistent prefix-state
restore intentionally rejects explicit Metal compute today, so this benchmark
validates the storage/state mechanism on the common CPU-safe execution variant.
Do not use these results as the final Metal performance number.

A passing run prints a table for `cold/off`, `RAM hit`, and `SSD hit/restart` and
writes the complete request telemetry to JSON. The SSD cache is kept under the
platform cache directory by default so failed runs can be inspected; pass
`--cleanup-cache` to delete the run-specific directory after the benchmark.

For a larger context after the smoke case passes, increase `--prefix-tokens` or
use `--prefix-file`. The engine telemetry's actual `prompt`/`matched` token count
is authoritative; `--prefix-tokens` only controls generated text size
approximately.
