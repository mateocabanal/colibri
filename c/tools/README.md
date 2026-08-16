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
- `benchmark_v4_perf.py`: canonical DeepSeek-V4 end-to-end benchmark matrix. It
  parses the V4 CLI's stable `prompt_tokens`, `v4_tokens`, `timing`, and
  `TUNE decode` diagnostics and emits JSONL/CSV records with TTFT, generated
  throughput, expert-cache traffic, wall time, git SHA, platform, and relevant
  runtime environment. Long-prompt case names target ~512/2k/8k tokens; the
  actual tokenizer count from the engine is authoritative and is recorded.
- `gen_unicode.py`: tokenizer table generation

Run them from `c/`, for example:

```sh
python3 tools/convert_fp8_to_int4.py --selftest
python3 tools/make_glm_bench_model.py --output /tmp/colibri-bench
python3 tools/benchmark_v4_perf.py --selftest
python3 tools/benchmark_v4_perf.py --model /path/to/v4 \
  --engine ./deepseek_v4 --memory-gb 10 --memory-gb 14 \
  --cases decode1,decode32,decode128,prompt512,prompt2k,prompt8k \
  --trials 2 --keep-logs /tmp/v4-perf --output /tmp/v4-perf.jsonl
```

The V4 runner forces `--no-dspark` so results measure the exact target path.
It does **not** claim to flush the OS page cache; repeated trials are therefore
reported as trials rather than as synthetic "cold"/"warm" labels. A later
engine-instrumentation slice will add phase-local CPU/I/O/Metal buckets and a
controlled same-process warm-cache mode.
