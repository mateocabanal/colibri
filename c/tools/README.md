# Tools

These scripts support model preparation and offline engineering work. They are
not runtime dependencies of the C engine.

- `convert_fp8_to_int4.py`, `download_glm52.py`: model preparation
- `convert_fmt4_to_fmt2.py`: fmt=4 (grouped int4) -> fmt=2 (per-row int4)
  re-quant of a GLM-5.2 container, for Metal-backend compatibility
  (see `docs/METAL-M1ULTRA-FMT2-REPORT.md`)
- `repack_fp8_passthrough.py`: fmt=8 repack (byte-preserved FP8, resident kinds only;
  see the module docstring -- synthetic-fixture-tested only, no real-shard runs yet)
- `make_glm_oracle.py`, `make_glm_bench_model.py`: deterministic fixtures
- `benchmark_cuda_fixture.py`, `eval_glm.py`, `fetch_benchmarks.py`: benchmarks
- `benchmark_v4_perf.py`: DeepSeek-V4 end-to-end benchmark runner. The default
  `quick` profile is **one** short-prompt, 8-token decode run with a hard 120 s
  subprocess timeout, so the normal edit/build/benchmark loop stays around the
  1-minute scale on the current M2 baseline and can never exceed 2 minutes.
  `standard` adds 512/2k prefill cases (one output token each); `full` restores
  sustained 32/128-token decode plus 8k prefill and has no default timeout. The
  runner parses stable V4 CLI diagnostics and emits JSONL/CSV with TTFT,
  generated throughput, expert-cache traffic, wall time, git SHA, platform, and
  relevant environment.
- `gen_unicode.py`: tokenizer table generation

Run them from `c/`, for example:

```sh
python3 tools/convert_fp8_to_int4.py --selftest
python3 tools/make_glm_bench_model.py --output /tmp/colibri-bench
python3 tools/benchmark_v4_perf.py --selftest

# Default developer loop: one decode8 run, hard-capped at 120 s total.
python3 tools/benchmark_v4_perf.py --model /path/to/v4 \
  --engine ./deepseek_v4 --memory-gb 10 \
  --keep-logs /tmp/v4-perf --output /tmp/v4-perf.jsonl

# Broader but still bounded profile.
python3 tools/benchmark_v4_perf.py --profile standard --model /path/to/v4 \
  --engine ./deepseek_v4 --memory-gb 10 --output /tmp/v4-standard.jsonl

# Long regression/sustained matrix; intentionally not time-bounded.
python3 tools/benchmark_v4_perf.py --profile full --model /path/to/v4 \
  --engine ./deepseek_v4 --memory-gb 10 --output /tmp/v4-full.jsonl

# Target a specific path while retaining a hard wall-time cap.
python3 tools/benchmark_v4_perf.py --model /path/to/v4 \
  --cases prefill2k --timeout-sec 120
```

The V4 runner forces `--no-dspark` so results measure the exact target path.
It does **not** claim to flush the OS page cache; repeated trials are therefore
reported as trials rather than synthetic "cold"/"warm" labels. A later
engine-instrumentation slice will add phase-local CPU/I/O/Metal buckets and a
controlled same-process warm-cache mode.
