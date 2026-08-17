#!/usr/bin/env python3
"""Controlled Qwen prompt-prefix cache-off/cache-on TTFT A/B.

Each mode runs in a fresh process under the same RAM_GB limit. The process first
prefills a stable prefix P, then measures an identical strict P+X request. Cache
mode only changes QWEN_PREFIX_CACHE_MB; Qwen's RAM_GB path reserves that budget
before deriving its expert-residency cap, so the comparison keeps the requested
total host/UMA envelope fixed.

The engine's QWEN-PREFIX request telemetry is authoritative for token counts,
matched tokens, snapshot bytes, restore time, tail-prefill time, and capture
time. First-byte latency is measured by this driver at the serve protocol
boundary. Trials alternate cache-off/cache-on process order to reduce systematic
OS/storage warming bias.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

READY = b"\x01\x01READY\x01\x01"
BENCH_TEXT = (
    "Colibri streams mixture-of-experts weights from storage while keeping the "
    "active working set bounded. This prompt repeats stable system, tool, "
    "repository, and agent context so prompt-prefix reuse can skip old prefill "
    "work and execute only the fresh request tail. "
)
TAIL_TEXT = "\nUser request: summarize the new changes and identify the next concrete action."
METRIC_RE = re.compile(r"([A-Za-z_]+)=([^\s]+)")
RELEVANT_ENV = (
    "CACHE", "RAM_GB", "QWEN_KV_F16", "QWEN_PREFETCH", "QWEN_METAL_COMPUTE",
    "QWEN_METALIO", "EXPERT_DROP", "DENSE_DROP", "OMP_NUM_THREADS",
    "OMP_PROC_BIND", "OMP_PLACES", "COLI_USAGE", "COLI_MODEL",
)


def make_prompt(target_tokens: int, chars_per_token: float) -> bytes:
    target_chars = max(1, int(target_tokens * chars_per_token))
    copies = max(1, (target_chars + len(BENCH_TEXT) - 1) // len(BENCH_TEXT))
    text = (BENCH_TEXT * copies)[:target_chars]
    if not text.endswith((".", "!", "?", "\n")):
        text += "."
    return text.encode("utf-8")


def git_sha(repo: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=True,
        )
        return proc.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_metric_line(line: str) -> dict[str, object] | None:
    if not line.startswith("[QWEN-PREFIX] request="):
        return None
    result: dict[str, object] = {}
    for key, raw in METRIC_RE.findall(line):
        try:
            if any(ch in raw for ch in ".eE"):
                result[key] = float(raw)
            else:
                result[key] = int(raw)
        except ValueError:
            result[key] = raw
    return result


class Serve:
    def __init__(
        self,
        binary: Path,
        model: Path,
        *,
        memory_gb: float,
        cache_mb: float,
        min_prefix_tokens: int,
        context: int,
    ) -> None:
        env = os.environ.copy()
        # CACHE would override RAM_GB inside Qwen and defeat fixed-total-memory
        # accounting, so reject it rather than silently benchmarking another
        # residency policy.
        if env.get("CACHE"):
            raise RuntimeError("unset CACHE: benchmark requires Qwen RAM_GB auto sizing")
        env.update(
            SERVE="1",
            SNAP=str(model),
            CTX=str(context),
            RAM_GB=f"{memory_gb:g}",
            QWEN_PREFIX_CACHE_MB=f"{cache_mb:g}",
            QWEN_PREFIX_CACHE_MIN_TOKENS=str(min_prefix_tokens),
            QWEN_PREFIX_LOG="1",
        )
        # argv cap=0 deliberately selects Qwen's CACHE -> RAM_GB -> topk auto
        # path. Passing a positive cap would make prefix memory additive.
        self.process = subprocess.Popen(
            [str(binary), "0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        self._stderr: list[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._wait_ready()

    def _wait_ready(self) -> None:
        assert self.process.stdout is not None
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "qwen_moe exited before READY:\n" + "".join(self._stderr)[-6000:]
                )
            if READY in line:
                return
        raise RuntimeError("qwen_moe did not report READY")

    def request(self, request_id: str, prompt: bytes, max_tokens: int) -> dict[str, object]:
        assert self.process.stdin is not None and self.process.stdout is not None
        header = (
            f"SUBMIT {request_id} 0 {len(prompt)} {max_tokens} 0.0 1.0\n"
        ).encode("ascii")
        began = time.perf_counter()
        self.process.stdin.write(header + prompt + b"\n")
        self.process.stdin.flush()

        output: list[bytes] = []
        first_byte_sec: float | None = None
        done_fields: list[str] | None = None
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"qwen_moe closed during {request_id}:\n"
                    + "".join(self._stderr)[-6000:]
                )
            fields = line.decode("latin-1").split()
            if not fields:
                continue
            kind = fields[0]
            if kind == "ERROR":
                raise RuntimeError(f"engine ERROR during {request_id}: {line!r}")
            if kind == "DATA" and len(fields) == 3:
                if first_byte_sec is None:
                    first_byte_sec = time.perf_counter() - began
                size = int(fields[2])
                output.append(self.process.stdout.read(size))
                self.process.stdout.read(1)  # DATA payload trailing newline
            elif kind == "DONE":
                done_fields = fields
                break
            # ACCEPT/PROF/STAT/HWINFO/TIERS/EMAP are protocol metadata.

        done_sec = time.perf_counter() - began
        if first_byte_sec is None:
            # A stop token can produce DONE without DATA. Keep a comparable
            # first-response latency instead of discarding the trial.
            first_byte_sec = done_sec
        return {
            "request_id": request_id,
            "first_response_sec": first_byte_sec,
            "done_sec": done_sec,
            "output_hex": b"".join(output).hex(),
            "done_fields": done_fields,
        }

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in iter(self.process.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", "replace"))

    def close(self) -> str:
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=60)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=10)
        self._stderr_thread.join(timeout=10)
        return "".join(self._stderr)


def metric_for(stderr: str, request_id: str) -> dict[str, object]:
    found = [m for line in stderr.splitlines() if (m := parse_metric_line(line)) and m.get("request") == request_id]
    if len(found) != 1:
        raise RuntimeError(
            f"expected one QWEN-PREFIX metric for {request_id!r}, got {len(found)}:\n"
            + stderr[-8000:]
        )
    return found[0]


def run_mode(
    *,
    mode: str,
    binary: Path,
    model: Path,
    prefix: bytes,
    extended: bytes,
    memory_gb: float,
    cache_mb: float,
    min_prefix_tokens: int,
    context: int,
    max_tokens: int,
) -> dict[str, object]:
    enabled = mode == "cache_on"
    serve = Serve(
        binary,
        model,
        memory_gb=memory_gb,
        cache_mb=cache_mb if enabled else 0.0,
        min_prefix_tokens=min_prefix_tokens,
        context=context,
    )
    began = time.perf_counter()
    try:
        warm = serve.request("warm", prefix, max_tokens)
        measured = serve.request("measure", extended, max_tokens)
    finally:
        stderr = serve.close()
    warm_metric = metric_for(stderr, "warm")
    measured_metric = metric_for(stderr, "measure")
    record = {
        "mode": mode,
        "process_wall_sec": time.perf_counter() - began,
        "warm": warm,
        "measure": measured,
        "warm_metric": warm_metric,
        "measure_metric": measured_metric,
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./qwen_moe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-file")
    parser.add_argument("--tail-file")
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--cache-mb", type=float, default=512.0)
    parser.add_argument("--min-prefix-tokens", type=int, default=256)
    parser.add_argument("--memory-gb", type=float, required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--repo")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.memory_gb <= 0 or args.cache_mb <= 0:
        parser.error("--memory-gb and --cache-mb must be positive")
    if args.min_prefix_tokens < 1 or args.context < 2 or args.max_tokens < 1 or args.trials < 1:
        parser.error("min-prefix/context/max-tokens/trials must be positive")
    if args.prefix_tokens < 1 or args.chars_per_token <= 0:
        parser.error("--prefix-tokens and --chars-per-token must be positive")

    binary = Path(args.binary).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve() if args.repo else binary.parent.parent
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    if not model.exists():
        parser.error(f"model does not exist: {model}")

    if args.prefix_file:
        prefix = Path(args.prefix_file).expanduser().read_bytes()
    else:
        prefix = make_prompt(args.prefix_tokens, args.chars_per_token)
    if args.tail_file:
        tail = Path(args.tail_file).expanduser().read_bytes()
    else:
        tail = TAIL_TEXT.encode("utf-8")
    if not prefix or not tail:
        parser.error("prefix and tail must both be non-empty")
    extended = prefix + tail

    pairs: list[dict[str, object]] = []
    started = time.perf_counter()
    for trial_index in range(args.trials):
        trial = trial_index + 1
        order = ["cache_off", "cache_on"] if trial_index % 2 == 0 else ["cache_on", "cache_off"]
        observed: dict[str, dict[str, object]] = {}
        for mode in order:
            sys.stderr.write(f"[qwen-prefix-bench] trial={trial} mode={mode}\n")
            observed[mode] = run_mode(
                mode=mode,
                binary=binary,
                model=model,
                prefix=prefix,
                extended=extended,
                memory_gb=args.memory_gb,
                cache_mb=args.cache_mb,
                min_prefix_tokens=args.min_prefix_tokens,
                context=args.context,
                max_tokens=args.max_tokens,
            )

        off, on = observed["cache_off"], observed["cache_on"]
        off_m = off["measure_metric"]
        on_m = on["measure_metric"]
        for field in ("prompt",):
            if int(off_m[field]) != int(on_m[field]):
                raise RuntimeError(f"trial {trial}: token workload differs for {field}")
        if int(off_m.get("matched", -1)) != 0:
            raise RuntimeError(f"trial {trial}: cache-off unexpectedly reused a prefix")
        warm_tokens = int(on["warm_metric"]["prompt"])
        if int(on_m.get("matched", -1)) != warm_tokens:
            raise RuntimeError(
                f"trial {trial}: cache-on matched {on_m.get('matched')} of {warm_tokens} warm tokens"
            )
        if off["measure"]["output_hex"] != on["measure"]["output_hex"]:
            raise RuntimeError(f"trial {trial}: cache-on output differs from cache-off")

        off_ttft = float(off["measure"]["first_response_sec"])
        on_ttft = float(on["measure"]["first_response_sec"])
        pair = {
            "trial": trial,
            "order": order,
            "cache_off": off,
            "cache_on": on,
            "ttft_off_sec": off_ttft,
            "ttft_on_sec": on_ttft,
            "ttft_saved_sec": off_ttft - on_ttft,
            "ttft_speedup": off_ttft / on_ttft if on_ttft > 0 else None,
        }
        pairs.append(pair)

    off_values = [float(p["ttft_off_sec"]) for p in pairs]
    on_values = [float(p["ttft_on_sec"]) for p in pairs]
    speedups = [float(p["ttft_speedup"]) for p in pairs if p["ttft_speedup"] is not None]
    sample_metric = pairs[0]["cache_on"]["measure_metric"]
    result = {
        "schema": "colibri.qwen.prefix_cache_benchmark.v1",
        "git_sha": git_sha(repo),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "binary": str(binary),
        "model": str(model),
        "memory_gb": args.memory_gb,
        "cache_mb": args.cache_mb,
        "min_prefix_tokens": args.min_prefix_tokens,
        "context": args.context,
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "environment": {k: os.environ[k] for k in RELEVANT_ENV if k in os.environ},
        "prompt_bytes": len(prefix),
        "tail_bytes": len(tail),
        "matched_tokens": int(sample_metric["matched"]),
        "snapshot_bytes": int(sample_metric["snapshot_bytes"]),
        "summary": {
            "ttft_off_median_sec": statistics.median(off_values),
            "ttft_on_median_sec": statistics.median(on_values),
            "ttft_saved_median_sec": statistics.median(off_values) - statistics.median(on_values),
            "ttft_speedup_median": statistics.median(speedups),
            "restore_ms_median": statistics.median(float(p["cache_on"]["measure_metric"]["restore_ms"]) for p in pairs),
            "tail_prefill_ms_median": statistics.median(float(p["cache_on"]["measure_metric"]["prefill_ms"]) for p in pairs),
            "capture_ms_median": statistics.median(float(p["cache_on"]["measure_metric"]["capture_ms"]) for p in pairs),
        },
        "pairs": pairs,
        "wall_sec": time.perf_counter() - started,
    }

    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        Path(args.output).expanduser().write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
