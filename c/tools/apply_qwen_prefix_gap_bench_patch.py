#!/usr/bin/env python3
from pathlib import Path

P = Path(__file__).resolve().with_name("benchmark_qwen_prefix_cache.py")
s = P.read_text()

def rep(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{label}: expected one match, found {n}")
    s = s.replace(old, new, 1)

rep(
'''time. First-byte latency is measured by this driver at the serve protocol\nboundary. Trials alternate cache-off/cache-on process order to reduce systematic\nOS/storage warming bias.\n''',
'''time. Protocol-boundary first-token latency, first-to-second-token gap, and\ncompletion latency are all measured: moving snapshot capture after the first\nDATA frame must not be mistaken for eliminating its cost. Trials alternate\ncache-off/cache-on process order to reduce systematic OS/storage warming bias.\n''',
"doc latency scope")
rep(
'''        # CACHE would override RAM_GB inside Qwen and defeat fixed-total-memory\n        # accounting, so reject it rather than silently benchmarking another\n        # residency policy.\n        if env.get("CACHE"):\n            raise RuntimeError("unset CACHE: benchmark requires Qwen RAM_GB auto sizing")\n        env.update(\n''',
'''        env.update(\n''',
"remove CACHE rejection")
rep(
'''        # argv cap=0 deliberately selects Qwen's CACHE -> RAM_GB -> topk auto\n        # path. Passing a positive cap would make prefix memory additive.\n''',
'''        # argv cap=0 deliberately bypasses any inherited CACHE setting and\n        # selects Qwen's RAM_GB -> topk path. Passing a positive positional cap\n        # would make prefix memory additive instead of fixed-total-memory.\n''',
"cap comment")
rep(
'''        output: list[bytes] = []\n        first_byte_sec: float | None = None\n        done_fields: list[str] | None = None\n''',
'''        output: list[bytes] = []\n        data_times: list[float] = []\n        done_fields: list[str] | None = None\n''',
"response timing state")
rep(
'''            if kind == "DATA" and len(fields) == 3:\n                if first_byte_sec is None:\n                    first_byte_sec = time.perf_counter() - began\n                size = int(fields[2])\n''',
'''            if kind == "DATA" and len(fields) == 3:\n                data_times.append(time.perf_counter() - began)\n                size = int(fields[2])\n''',
"record every DATA time")
rep(
'''        done_sec = time.perf_counter() - began\n        if first_byte_sec is None:\n            # A stop token can produce DONE without DATA. Keep a comparable\n            # first-response latency instead of discarding the trial.\n            first_byte_sec = done_sec\n        return {\n            "request_id": request_id,\n            "first_response_sec": first_byte_sec,\n            "done_sec": done_sec,\n            "output_hex": b"".join(output).hex(),\n            "done_fields": done_fields,\n        }\n''',
'''        done_sec = time.perf_counter() - began\n        first_response_sec = data_times[0] if data_times else done_sec\n        second_response_sec = data_times[1] if len(data_times) > 1 else None\n        return {\n            "request_id": request_id,\n            "first_response_sec": first_response_sec,\n            "second_response_sec": second_response_sec,\n            "first_to_second_sec": (\n                second_response_sec - first_response_sec\n                if second_response_sec is not None else None\n            ),\n            "data_frames": len(data_times),\n            "done_sec": done_sec,\n            "output_hex": b"".join(output).hex(),\n            "done_fields": done_fields,\n        }\n''',
"return response gaps")
rep(
'''    parser.add_argument("--max-tokens", type=int, default=1)\n''',
'''    parser.add_argument("--max-tokens", type=int, default=2)\n''',
"default two tokens")
rep(
'''    if args.min_prefix_tokens < 1 or args.context < 2 or args.max_tokens < 1 or args.trials < 1:\n        parser.error("min-prefix/context/max-tokens/trials must be positive")\n''',
'''    if args.min_prefix_tokens < 1 or args.context < 2 or args.max_tokens < 2 or args.trials < 1:\n        parser.error("min-prefix/trials must be positive, context >=2, and max-tokens >=2")\n''',
"require two-token probe")
rep(
'''        if off["measure"]["output_hex"] != on["measure"]["output_hex"]:\n            raise RuntimeError(f"trial {trial}: cache-on output differs from cache-off")\n\n        off_ttft = float(off["measure"]["first_response_sec"])\n        on_ttft = float(on["measure"]["first_response_sec"])\n''',
'''        if off["measure"]["output_hex"] != on["measure"]["output_hex"]:\n            raise RuntimeError(f"trial {trial}: cache-on output differs from cache-off")\n        if off["measure"]["second_response_sec"] is None or on["measure"]["second_response_sec"] is None:\n            raise RuntimeError(\n                f"trial {trial}: benchmark needs at least two emitted tokens to measure "\n                "the post-first-token capture stall; choose a prompt/tail that does not stop early"\n            )\n\n        off_ttft = float(off["measure"]["first_response_sec"])\n        on_ttft = float(on["measure"]["first_response_sec"])\n        off_gap = float(off["measure"]["first_to_second_sec"])\n        on_gap = float(on["measure"]["first_to_second_sec"])\n''',
"require and compute second token")
rep(
'''            "ttft_saved_sec": off_ttft - on_ttft,\n            "ttft_speedup": off_ttft / on_ttft if on_ttft > 0 else None,\n''',
'''            "ttft_saved_sec": off_ttft - on_ttft,\n            "ttft_speedup": off_ttft / on_ttft if on_ttft > 0 else None,\n            "first_to_second_off_sec": off_gap,\n            "first_to_second_on_sec": on_gap,\n            "first_to_second_penalty_sec": on_gap - off_gap,\n            "completion_off_sec": float(off["measure"]["done_sec"]),\n            "completion_on_sec": float(on["measure"]["done_sec"]),\n''',
"pair latency fields")
rep(
'''    speedups = [float(p["ttft_speedup"]) for p in pairs if p["ttft_speedup"] is not None]\n    sample_metric = pairs[0]["cache_on"]["measure_metric"]\n''',
'''    speedups = [float(p["ttft_speedup"]) for p in pairs if p["ttft_speedup"] is not None]\n    gap_off_values = [float(p["first_to_second_off_sec"]) for p in pairs]\n    gap_on_values = [float(p["first_to_second_on_sec"]) for p in pairs]\n    completion_off_values = [float(p["completion_off_sec"]) for p in pairs]\n    completion_on_values = [float(p["completion_on_sec"]) for p in pairs]\n    sample_metric = pairs[0]["cache_on"]["measure_metric"]\n''',
"summary arrays")
rep(
'''            "ttft_speedup_median": statistics.median(speedups),\n            "restore_ms_median": statistics.median(float(p["cache_on"]["measure_metric"]["restore_ms"]) for p in pairs),\n''',
'''            "ttft_speedup_median": statistics.median(speedups),\n            "first_to_second_off_median_sec": statistics.median(gap_off_values),\n            "first_to_second_on_median_sec": statistics.median(gap_on_values),\n            "first_to_second_penalty_median_sec": (\n                statistics.median(gap_on_values) - statistics.median(gap_off_values)\n            ),\n            "completion_off_median_sec": statistics.median(completion_off_values),\n            "completion_on_median_sec": statistics.median(completion_on_values),\n            "completion_delta_median_sec": (\n                statistics.median(completion_on_values) - statistics.median(completion_off_values)\n            ),\n            "restore_ms_median": statistics.median(float(p["cache_on"]["measure_metric"]["restore_ms"]) for p in pairs),\n''',
"summary latency fields")
P.write_text(s)
print("qwen gap-aware benchmark patch applied")
