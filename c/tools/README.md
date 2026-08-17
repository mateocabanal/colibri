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
- `analyze_v4_expert_trace.py`: offline routed-expert analysis. The physical
  lifecycle trace at `V4_EXPERT_TRACE` remains schema v1 and reports request,
  hit/inflight/load/evict/release activity with stable `(layer, expert, tier,
  generation)` identity. When detailed tracing is enabled, the V4 block runtime
  also emits an explicit logical-route schema-v2 sidecar at
  `<V4_EXPERT_TRACE>.routes.jsonl` (or `V4_ROUTE_TRACE=<path>`). That sidecar
  records every top-k route selection before batch-union lookups collapse many
  logical selections into one physical lease. It includes request id, token
  position, prefill/decode phase, layer, expert, route rank/weight, and the
  physical lookup correlation (`lookup_id`, lookup duration, fanout, result,
  lease generation). Both schemas are accepted by the analyzer. It reports
  activation skew, exact LRU stack/reuse distance, hypothetical cache curves,
  co-routing, unique logical experts/token and adjacent-token overlap. Old v1
  traces still fall back to layer-wrap token inference; explicit v2 route traces
  use authoritative request/token/phase identity.
- `analyze_v4_request_overlap.py`: schema-v2 serving-locality view for #56.
  It keeps analysis linear in request count by comparing consecutive generation
  requests instead of doing an O(requests²) all-pairs sweep. It reports shared
  logical experts, Jaccard overlap, previous-working-set retention, next-request
  reuse, and per-layer adjacent-request overlap. Duplicate route selections do
  not inflate the working set.
- `validate_v4_trace_pair.py`: exact-COLI identity/lifecycle closure gate. It
  verifies every logical v2 lease joins the physical v1 lifecycle by
  `(layer, expert, generation)`, checks many-to-one lookup fanout, rejects
  dropped/corrupt traces, and can additionally validate a compile-gated expert
  execution trace with `--execution`.
- `analyze_v4_residency_value.py`: offline policy-reference tool for #3. It
  compares deterministic residency and global hot-expert capacity using
  estimated **exposed I/O milliseconds avoided per resident GiB**. Repeated
  `--dense-point` inputs can also produce a finite-difference marginal dense
  frontier, but those points must be from the same workload and differ only in
  residency budget.
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

# Capture physical residency lifecycle plus explicit logical route identity.
# The second file is created automatically as /tmp/v4-experts.jsonl.routes.jsonl.
V4_EXPERT_TRACE=/tmp/v4-experts.jsonl ./deepseek_v4 /path/to/model.coli hi

# Analyze physical cache/load behavior (schema v1).
python3 tools/analyze_v4_expert_trace.py /tmp/v4-experts.jsonl \
  --capacities 1,2,4,8,16,32,64,128 \
  --persistent-capacities 1,2,4,8 \
  --prompt-tokens 5

# Analyze authoritative logical route selections (schema v2).
python3 tools/analyze_v4_expert_trace.py \
  /tmp/v4-experts.jsonl.routes.jsonl --top 10

# For a multi-request/serve trace, quantify how much of each request's logical
# expert working set is reused by the next request.
python3 tools/analyze_v4_request_overlap.py \
  /tmp/v4-experts.jsonl.routes.jsonl --top-layers 10

# Validate physical-v1 <-> logical-v2 lease identity on an exact COLI run.
python3 tools/validate_v4_trace_pair.py \
  /tmp/v4-experts.jsonl /tmp/v4-experts.jsonl.routes.jsonl

# Optional execution lifecycle capture. This is COMPILE-GATED so normal builds
# do not put a tracing wrapper on expert-forward at all. The execution stream is
# still buffered/bounded and written only at process teardown.
make deepseek-v4-clean
make deepseek-v4 V4_TRACE_EXEC=1
V4_EXPERT_TRACE=/tmp/v4-experts.jsonl \
V4_EXEC_TRACE=/tmp/v4-exec.jsonl \
V4_EXEC_TRACE_CAP=131072 \
./deepseek_v4 /path/to/model.coli hi
python3 tools/validate_v4_trace_pair.py \
  /tmp/v4-experts.jsonl /tmp/v4-experts.jsonl.routes.jsonl \
  --execution /tmp/v4-exec.jsonl

# Optional explicit route-sidecar controls. Request ids are assigned
# monotonically at the session-generate boundary. V4_ROUTE_REQUEST_ID changes
# the first id in that sequence (useful for deterministic fixtures), rather than
# pinning every generation request to one id.
V4_ROUTE_TRACE=/tmp/v4-routes.jsonl \
V4_ROUTE_TRACE_CAP=131072 \
V4_ROUTE_REQUEST_ID=42 \
./deepseek_v4 /path/to/model.coli hi

# Compare residency classes in the #3 policy metric. This example uses one
# dense point; repeat --dense-point with SAME-WORKLOAD fixed-memory A/B points
# to obtain a marginal dense frontier.
python3 tools/analyze_v4_residency_value.py /tmp/v4-experts.jsonl \
  --dense-point 3.16,69.58,68.27,39612.6 \
  --expert-read-gib 70.68 \
  --expert-wait-ms 7776.5 \
  --expert-capacities 43,86,172,344
```

The V4 runner forces `--no-dspark` so results measure the exact target path.
It does **not** claim to flush the OS page cache; use `--warm-cache` for an
explicit cold/warm pair (trial 2 = page-cache warm).

Detailed route tracing is bounded in memory and writes the JSONL only at normal
process teardown. The v2 header reports `dropped`, `physical_lookups`,
`correlation_misses`, and `uncorrelated_routes`; a clean trace should have all
three loss/error counters at zero. One physical lookup can legitimately have
`lookup_routes > 1` during batch-union prefill. All logical routes in that group
share the same `lookup_id`, lookup duration, result and `lease_generation`.
The `(layer, expert, lease_generation)` triple is the stable join key back to the
physical v1 lifecycle and, for a `V4_TRACE_EXEC=1` build, to the expert execution
stream. The execution stream reports one `execute` record per routed expert
forward with resident bytes, tensor execution formats, duration, result and
lease generation. Its buffer obeys `V4_EXEC_TRACE_CAP` (falling back to the route
or expert trace cap) and reports drops rather than blocking inference.

`lookup_ns` is physical lookup latency and may overlap other pipeline work; do
not relabel it as exposed token-wall wait without an owner-thread wait
measurement. Likewise, the sum of per-expert execution durations is diagnostic:
backend work can overlap or include accelerator waits already represented by the
canonical #1 phase profiler.

Phase telemetry: by default the runner sets `V4_PROFILE=1`, and the engine
emits one `v4_phases scope=startup|run|prompt|decode` line per scope with
per-phase ms and byte buckets (dense/expert/head read, expert lookup/read
work/wait/compute, shared expert, router, HC pre/post, attention projection,
compressor, indexer, sparse attention, head, and Metal encode/submit/wait).
The runner parses these into `result.phases` and checks the reconciliation
gate: run-scope `unaccounted_ms <= max(5 ms, 1% of wall)` →
`phases_reconcile`. Expert read work and Metal kernel time are reported as
overlapping diagnostics and are deliberately excluded from the accounted sum
(lanes/GPU run concurrently). `--no-profile` disables telemetry for a pure
wall-clock comparison.

## Interpreting residency traces

The policy metric for #3 is now **expected recurring exposed I/O time avoided
per resident byte**. Raw bytes avoided/resident byte remains a useful stable
fallback, but it can mis-rank expert and deterministic residency because expert
storage reads are often hidden behind concurrent compute/read lanes while dense
reads are much more exposed to token wall time.

The short zero-copy M2 run kept 3.70 GiB of deterministic tensors resident and
avoided 33.28 GiB of rereads. The longer 33-token run kept 3.16 GiB resident and
avoided 69.58 GiB. The latter also showed that persistent-expert frequency value
is generation-horizon dependent: a global perfect-hot 43-slot oracle reached
about 19.5 bytes avoided/resident byte. That does **not** mean the runtime should
allocate those slots by default: in the same run dense I/O exposed about
39.61 s for 68.27 GiB read while expert I/O exposed only about 7.78 s for
70.68 GiB read.

For planner decisions, compare the **marginal** dense tail against an expert
capacity band, not cache-wide averages. Use fixed-memory, same-workload A/Bs to
measure the dense marginal frontier; auto-detected free RAM can shift the dense
budget enough to confound small performance differences.

Do not confuse storage reduction with end-to-end speedup. PR #64 established
that zero-copy deterministic residency can materially reduce decode I/O while
four global transient expert slots are enough for the measured three-lane M2
pipeline (`peak_inflight=3`, `slot_waits=0`). Persistent expert capacity remains
a trace-driven optional tier rather than a fixed per-layer allocation.
