#!/usr/bin/env python3
"""Qwen decode roofline + synchronization benchmark (#156).

Runs the qwen_moe binary with QWEN_PROFILE=1 on a fixed prompt/context
fixture, parses the coli_profile scope lines, and reports per-token
subsystem attribution plus a roofline comparison against the M2's ~100
GB/s unified-memory bandwidth.

The engine already emits the spans (committed 0251337): moe_route/io/
shared/gpu/fill_ms, gdn_ms, attention_ms, head_compute_ms, expert_io_ms,
plus metal_encode/submit/wait/kernel and the moe counters. This runner
only parses and attributes — it is engine-independent and works on
older commits too.

Usage:
  benchmark_qwen_decode.py MODEL [CACHE] [--tokens N] [--prompt-ids "1 2 3"]
                            [--config /path/to/HF/model]
  benchmark_qwen_decode.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

PROFILE_RE = re.compile(
    r"coli_profile engine=qwen_moe scope=(\w+) tokens=(\d+) "
    r"wall_ms=([0-9.eE+-]+) accounted_ms=([0-9.eE+-]+) "
    r"unaccounted_ms=([0-9.eE+-]+) cpu_compute_ms=([0-9.eE+-]+) "
    r"cpu_wait_ms=([0-9.eE+-]+) io_wait_ms=([0-9.eE+-]+) "
    r"(.*)"
)
FIELD_RE = re.compile(r"(\w+)=([0-9.eE+-]+)")

# Subsystem -> bytes/token moved on Qwen3.6-35B-A3B (routed Apple8 1.17 GiB,
# shared BF16 0.5 GiB, GDN projections 0.67 GiB, lm_head 0.62 GiB). These are
# the model-geometry constants from the #150 roofline analysis; the benchmark
# reports GB/s per subsystem against the M2's ~100 GB/s ceiling.
ROOFLINE_BYTES_PER_TOKEN = {
    "moe_route_ms": 0.0,        # router: tiny f32 gate rows
    "moe_io_ms": 1.17e9,        # routed Apple8 expert payloads
    "moe_shared_ms": 0.5e9,     # shared expert BF16 matmuls
    "moe_gpu_ms": 1.17e9,       # GPU-side routed expert reads
    "moe_fill_ms": 0.0,         # host fill/publish, not bandwidth
    "gdn_ms": 0.67e9,           # GDN projections (BF16)
    "attention_ms": 0.0,        # KV reads scale with context, not fixed
    "head_compute_ms": 0.62e9,  # lm_head
    "expert_io_ms": 1.17e9,     # expert I/O wait
}
M2_BANDWIDTH_GBPS = 100.0


def parse_profile(stderr: str) -> dict:
    scopes = {}
    for line in stderr.splitlines():
        m = PROFILE_RE.match(line)
        if not m:
            continue
        scope, tokens, wall, accounted, unaccounted, cpu_compute, cpu_wait, io_wait, rest = m.groups()
        fields = {"wall_ms": float(wall), "accounted_ms": float(accounted),
                  "unaccounted_ms": float(unaccounted),
                  "cpu_compute_ms": float(cpu_compute),
                  "cpu_wait_ms": float(cpu_wait), "io_wait_ms": float(io_wait),
                  "tokens": int(tokens)}
        for name, val in FIELD_RE.findall(rest):
            try:
                fields[name] = float(val)
            except ValueError:
                pass
        scopes[scope] = fields
    return scopes


def roofline_report(scopes: dict) -> dict:
    decode = scopes.get("decode")
    if not decode:
        return {"error": "no decode scope in profile output"}
    tokens = decode.get("tokens", 0)
    if tokens <= 0:
        return {"error": "decode scope has zero tokens"}
    per_token = {}
    for name, bytes_per_token in ROOFLINE_BYTES_PER_TOKEN.items():
        ms = decode.get(name + "_ms", 0.0) if name + "_ms" in decode else decode.get(name, 0.0)
        if bytes_per_token > 0 and ms > 0:
            gbps = (bytes_per_token * tokens) / (ms * 1e-3) / 1e9
            per_token[name] = {
                "ms_per_token": ms / tokens,
                "gbps": gbps,
                "pct_of_roofline": gbps / M2_BANDWIDTH_GBPS * 100.0,
            }
        else:
            per_token[name] = {"ms_per_token": ms / tokens, "gbps": 0.0, "pct_of_roofline": 0.0}
    sync = {
        "metal_encode_ms": decode.get("metal_encode_ms", 0.0) / tokens,
        "metal_submit_ms": decode.get("metal_submit_ms", 0.0) / tokens,
        "metal_wait_ms": decode.get("metal_wait_ms", 0.0) / tokens,
        "metal_kernel_ms": decode.get("metal_kernel_ms", 0.0) / tokens,
        "metal_moe_wait_ms": decode.get("metal_moe_wait_ms", 0.0) / tokens,
        "metal_moe_kernel_ms": decode.get("metal_moe_kernel_ms", 0.0) / tokens,
        "command_buffers": decode.get("command_buffers", 0),
        "fused_layers": decode.get("fused_layers", 0),
        "fused_experts": decode.get("fused_experts", 0),
    }
    return {
        "tokens": tokens,
        "wall_ms_per_token": decode["wall_ms"] / tokens,
        "accounted_ms_per_token": decode["accounted_ms"] / tokens,
        "unaccounted_ms_per_token": decode["unaccounted_ms"] / tokens,
        "cpu_compute_ms_per_token": decode["cpu_compute_ms"] / tokens,
        "cpu_wait_ms_per_token": decode["cpu_wait_ms"] / tokens,
        "io_wait_ms_per_token": decode["io_wait_ms"] / tokens,
        "subsystems": per_token,
        "sync": sync,
        "roofline_gbps": M2_BANDWIDTH_GBPS,
    }


def run_bench(model: str, cache: str, tokens: int, prompt_ids: str,
              config: str = "") -> dict:
    env = dict(os.environ)
    env["QWEN_PROFILE"] = "1"
    env["QWENMOE_MODE"] = "greedy"
    env["QWENMOE_PROMPT_IDS"] = prompt_ids
    env["QWENMOE_MAX_NEW"] = str(tokens)
    if config:
        env["COLI_CONFIG"] = config
    cmd = ["./c/qwen_moe", model]
    if cache:
        cmd.append(cache)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"error": f"engine timed out after 600s", "exit_code": -1,
                "stderr_tail": ""}
    wall = time.time() - t0
    scopes = parse_profile(proc.stderr)
    report = roofline_report(scopes)
    report["host"] = platform.machine()
    report["wall_s"] = wall
    report["exit_code"] = proc.returncode
    if proc.returncode != 0:
        report["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-25:])
    return report


def self_test() -> int:
    sample = (
        "coli_profile engine=qwen_moe scope=decode tokens=40 wall_ms=7350.000 "
        "accounted_ms=7200.000 unaccounted_ms=+150.000 cpu_compute_ms=3100.000 "
        "cpu_wait_ms=4100.000 io_wait_ms=1900.000 "
        "moe_route_ms=120.000 moe_io_ms=1900.000 moe_shared_ms=440.000 "
        "moe_gpu_ms=1560.000 moe_fill_ms=80.000 gdn_ms=1693.000 "
        "attention_ms=673.000 head_compute_ms=729.000 expert_io_ms=1900.000 "
        "metal_encode_ms=537.000 metal_submit_ms=708.000 metal_wait_ms=2602.000 "
        "metal_kernel_ms=931.000 routed_expert_requests=3200 expert_loads=306 "
        "cache_hits=1894 cache_misses=306 command_buffers=13120 fused_layers=0 "
        "fused_experts=0\n"
    )
    scopes = parse_profile(sample)
    assert "decode" in scopes, "parse failed"
    r = roofline_report(scopes)
    assert r["tokens"] == 40
    assert abs(r["wall_ms_per_token"] - 183.75) < 0.01
    assert abs(r["subsystems"]["moe_io_ms"]["ms_per_token"] - 47.5) < 0.01
    assert r["subsystems"]["moe_io_ms"]["gbps"] > 0
    print("self-test ok: parse + roofline attribution")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", help="qwen snapshot dir or .coli package")
    ap.add_argument("cache", nargs="?", default="", help="expert cache slots (optional)")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--prompt-ids", default="1 2 3 4 5")
    ap.add_argument("--config", default="", help="HF dir with config.json (COLI_CONFIG)")
    ap.add_argument("--json", action="store_true", help="emit JSON record")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.model:
        ap.error("MODEL is required (or use --self-test)")
    report = run_bench(args.model, args.cache, args.tokens, args.prompt_ids, args.config)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    exit_code = report.get("exit_code", -1)
    print(f"host={report.get('host')} wall_s={report.get('wall_s', 0):.1f} "
          f"exit={exit_code}")
    if exit_code != 0:
        print("ENGINE FAILED (exit %d) — stderr tail:" % exit_code)
        print(report.get("stderr_tail", "(no stderr captured)"))
        return exit_code
    if "error" in report:
        print("ERROR:", report["error"])
        return 1
    print(f"decode tokens={report['tokens']} wall={report['wall_ms_per_token']:.1f} "
          f"ms/tok accounted={report['accounted_ms_per_token']:.1f} "
          f"unaccounted={report['unaccounted_ms_per_token']:+.1f}")
    print(f"  cpu_compute={report['cpu_compute_ms_per_token']:.1f} "
          f"cpu_wait={report['cpu_wait_ms_per_token']:.1f} "
          f"io_wait={report['io_wait_ms_per_token']:.1f} ms/tok")
    print("  subsystem ms/tok (GB/s, % of 100 GB/s roofline):")
    for name, s in report["subsystems"].items():
        if s["gbps"] > 0:
            print(f"    {name:16s} {s['ms_per_token']:8.1f}  {s['gbps']:6.1f} GB/s  {s['pct_of_roofline']:5.1f}%")
        else:
            print(f"    {name:16s} {s['ms_per_token']:8.1f}  (no fixed bandwidth model)")
    print("  sync ms/tok: encode=%.1f submit=%.1f wait=%.1f kernel=%.1f "
          "moe_wait=%.1f moe_kernel=%.1f" % (
        report["sync"]["metal_encode_ms"], report["sync"]["metal_submit_ms"],
        report["sync"]["metal_wait_ms"], report["sync"]["metal_kernel_ms"],
        report["sync"]["metal_moe_wait_ms"], report["sync"]["metal_moe_kernel_ms"]))
    print(f"  command_buffers={report['sync']['command_buffers']} "
          f"fused_layers={report['sync']['fused_layers']} "
          f"fused_experts={report['sync']['fused_experts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
