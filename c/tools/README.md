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
  subprocess timeout. `standard` adds 512/2k prefill cases; `full` restores
  sustained 32/128-token decode plus 8k prefill. The runner emits JSONL/CSV with
  TTFT, throughput, expert-cache traffic, wall time, git SHA, platform and
  relevant environment.
- `analyze_v4_expert_trace.py`: physical/logical routed-expert analysis. Physical
  `V4_EXPERT_TRACE` data is schema v1 and records residency lifecycle with stable
  `(layer, expert, tier, generation)` identity. A detailed route build can also
  emit schema v2 logical selections carrying request id, token position, phase,
  route rank/weight and physical lookup correlation. The analyzer reports
  activation skew, reuse distance, cache curves, co-routing and adjacent-token
  overlap.
- `analyze_v4_request_overlap.py`: adjacent-request working-set overlap for v2
  traces, including Jaccard/retention/reuse and per-layer overlap in O(requests).
- `validate_v4_trace_pair.py`: exact-COLI identity/lifecycle closure gate. It
  verifies logical v2 leases join physical v1 lifecycle by
  `(layer, expert, generation)`, checks lookup fanout, and optionally validates
  one expert execution per routed selection.
- `analyze_v4_expert_stalls.py`: combines physical v1, logical v2 and execution
  traces to compare worker-side lookup time with the canonical owner-thread
  expert-loader wait. It reports cold-vs-resident lookup groups, cold exposure,
  phase totals, and top experts/layers by exposed stall.
- `analyze_v4_residency_value.py`: offline policy-reference tool for #3. It
  compares deterministic residency and global hot-expert capacity using
  estimated **exposed I/O milliseconds avoided per resident GiB**. Repeated
  `--dense-point` inputs produce a finite-difference marginal dense frontier
  only when the points are the same workload with different residency budgets.
- `gen_unicode.py`: tokenizer table generation

## V4 tracing modes

Detailed routing/execution instrumentation is **compile-gated**. The ordinary
binary compiles the original block and generation units directly, so it has no
logical-route wrapper and no expert-forward trace wrapper on the hot path.
Physical store counters and the #1 phase profiler remain available normally.

```sh
# Normal inference / aggregate profiling build: detailed wrappers absent.
make deepseek-v4

# Logical route + physical lookup correlation, no execution wrapper.
make deepseek-v4-clean
make deepseek-v4 V4_TRACE_ROUTE=1

# Full #56 build: route tracing + expert execution/owner-wait trace.
# V4_TRACE_EXEC=1 enables V4_TRACE_ROUTE by default.
make deepseek-v4-clean
make deepseek-v4 V4_TRACE_EXEC=1
```

`V4_EXPERT_TRACE` itself is runtime-gated in the expert store. In a normal build
it can still capture the physical v1 lifecycle. The automatic logical
`<V4_EXPERT_TRACE>.routes.jsonl` sidecar only exists in a build compiled with
`V4_TRACE_ROUTE=1` (or `V4_TRACE_EXEC=1`).

## Capture and analyze

```sh
# Physical v1 only; works in a normal build.
V4_EXPERT_TRACE=/tmp/v4-experts.jsonl \
./deepseek_v4 /path/to/model.coli hi

python3 tools/analyze_v4_expert_trace.py /tmp/v4-experts.jsonl \
  --capacities 1,2,4,8,16,32,64,128 \
  --persistent-capacities 1,2,4,8

# Full detailed lifecycle build.
make deepseek-v4-clean
make deepseek-v4 V4_TRACE_EXEC=1
V4_PROFILE=1 \
V4_EXPERT_TRACE=/tmp/v4-experts.jsonl \
V4_EXEC_TRACE=/tmp/v4-exec.jsonl \
V4_EXPERT_TRACE_CAP=131072 \
V4_EXEC_TRACE_CAP=131072 \
./deepseek_v4 /path/to/model.coli hi

# Logical v2 sidecar is automatic for a route-enabled build.
python3 tools/analyze_v4_expert_trace.py \
  /tmp/v4-experts.jsonl.routes.jsonl --top 10

# Exact physical-v1 <-> logical-v2 <-> execution identity/lifecycle check.
python3 tools/validate_v4_trace_pair.py \
  /tmp/v4-experts.jsonl /tmp/v4-experts.jsonl.routes.jsonl \
  --execution /tmp/v4-exec.jsonl

# Cold expert storage work vs actually exposed owner-thread stall.
python3 tools/analyze_v4_expert_stalls.py \
  /tmp/v4-experts.jsonl /tmp/v4-experts.jsonl.routes.jsonl \
  /tmp/v4-exec.jsonl --stall-threshold-ms 1

# Cross-request locality from a serving/multi-generation v2 trace.
python3 tools/analyze_v4_request_overlap.py \
  /tmp/v4-experts.jsonl.routes.jsonl --top-layers 10

# Optional explicit logical-sidecar controls.
V4_ROUTE_TRACE=/tmp/v4-routes.jsonl \
V4_ROUTE_TRACE_CAP=131072 \
V4_ROUTE_REQUEST_ID=42 \
./deepseek_v4 /path/to/model.coli hi
```

Request ids are allocated monotonically at the session-generation boundary.
`V4_ROUTE_REQUEST_ID` changes the first id in that sequence for deterministic
fixtures; it does not pin all generations to the same id.

Detailed buffers never write synchronously on the routing/execution hot path.
They flush at normal process teardown and report `dropped` instead of blocking
when a configured cap is exhausted. Logical v2 additionally reports
`physical_lookups`, `correlation_misses`, and `uncorrelated_routes`.

One physical lookup can serve multiple logical prompt positions during
batch-union prefill. Those routes share `lookup_id`, `lookup_ns`, result,
fanout and `lease_generation`. The stable
`(layer, expert, lease_generation)` triple joins logical v2 selections to the
physical residency lifecycle and to the optional execution stream.

`lookup_ns` is worker-side physical lookup latency and can overlap other work.
With `V4_PROFILE=1`, the execution stream also records `owner_wait_ns`, derived
from the canonical `COLI_V4_PROF_EXPERT_LOADER_WAIT` bucket immediately before
that expert executes. That is the exposed owner-thread stall measurement. Do not
substitute `lookup_ns` for it. `analyze_v4_expert_stalls.py` keeps the two
separate and treats `1 - owner_wait/lookup` only as an overlap estimate.

The execution stream also records resident bytes, tensor execution formats,
route weight, duration, result and exact lease generation. Its event count is
validated against logical selections, including many-to-one prefill fanout.

## CI correctness gates

The generated tiny V4 target has two complementary gates:

1. Linux runs the source fixture token-exact against the Transformers-derived
   oracle with `V4_TRACE_EXEC=1`, checking request/token/rank/weight identity,
   lookup fanout, execution count and owner-wait measurement.
2. An Apple-Silicon macOS job compiles that fixture with
   `colic compile --target native --quant exact --codec none`, runs the
   package-only COLI runtime, and validates the physical v1 + logical v2 +
   execution streams together. The first green exact-COLI gate observed 54
   logical routes, 17 loaded lease generations, 0.289 ms of exposed owner wait
   and a 31.18% cold lookup exposure ratio while preserving exact output.

The ordinary Linux/macOS/Windows `make check` jobs keep detailed tracing compiled
out, so the default hot path is independently exercised without those wrappers.

## Performance telemetry and residency analysis

The V4 benchmark runner forces `--no-dspark` so results measure the exact target
path. It does **not** claim to flush the OS page cache; use `--warm-cache` for an
explicit cold/warm pair.

By default the runner sets `V4_PROFILE=1`, and the engine emits one
`v4_phases scope=startup|run|prompt|decode` line per scope with dense/expert/head
read time+bytes, expert read-work/wait/compute, router/shared-expert, HC,
attention/compressor/indexer/sparse-attention, head, and Metal timing. Worker I/O
and Metal kernel time are overlapping diagnostics and are excluded from the
owner-thread accounted sum. `--no-profile` disables this telemetry.

The policy metric for #3 is **expected recurring exposed I/O time avoided per
resident byte**. Raw bytes avoided/resident byte is a stable fallback but can
mis-rank residency classes because expert reads often overlap compute while
dense reads are more exposed. Compare the marginal dense tail against an expert
capacity band using fixed-memory, same-workload A/B runs rather than cache-wide
averages or auto-detected free RAM.

```sh
python3 tools/analyze_v4_residency_value.py /tmp/v4-experts.jsonl \
  --dense-point 3.16,69.58,68.27,39612.6 \
  --expert-read-gib 70.68 \
  --expert-wait-ms 7776.5 \
  --expert-capacities 43,86,172,344
```
