#!/usr/bin/env python3
"""Controlled Qwen prompt-prefix cache-off/cache-on TTFT A/B.

The default mode is the rigorous benchmark: every cache-off/cache-on observation
runs in a fresh process under the same RAM_GB limit. Use --fast for local
iteration: it starts only one process per mode, warms P once, then performs all
measured P+X requests in that process. --fast is much quicker on large models,
but it measures a warm-process/steady-state workload rather than cold-process
A/B economics.

The engine's QWEN-PREFIX request telemetry is authoritative for token counts,
matched tokens, snapshot bytes, restore time, tail-prefill time, and capture
time. Protocol-boundary first-token latency, first-to-second-token gap, and
completion latency are all measured: moving snapshot capture after the first
DATA frame must not be mistaken for eliminating its cost.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import re
import statistics
import subprocess
import sys
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
    "CACHE", "RAM_GB", "QWEN_KV_F16", "QWEN_PREFETCH", "QWEN_PREFETCH_PIPE",
    "QWEN_METAL_COMPUTE", "QWEN_METAL_IO", "QWEN_METALIO", "EXPERT_DROP",
    "DENSE_KEEP_PAGES", "DENSE_DROP", "OMP_NUM_THREADS", "OMP_PROC_BIND",
    "OMP_PLACES", "COLI_USAGE", "COLI_MODEL", "COLI_CONFIG",
)
_BENCH_STARTED = time.perf_counter()
_PROGRESS = True


def progress(message: str) -> None:
    if not _PROGRESS:
        return
    elapsed = time.perf_counter() - _BENCH_STARTED
    sys.stderr.write(f"[qwen-prefix-bench +{elapsed:7.1f}s] {message}\n")
    sys.stderr.flush()


class Heartbeat:
    def __init__(self, label: str, every_sec: float) -> None:
        self.label = label
        self.every_sec = every_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.perf_counter()

    def __enter__(self) -> "Heartbeat":
        if _PROGRESS and self.every_sec > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.every_sec):
            elapsed = time.perf_counter() - self._started
            progress(f"{self.label} ... {elapsed:.1f}s elapsed")

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


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
        label: str,
        memory_gb: float,
        cache_mb: float,
        min_prefix_tokens: int,
        context: int,
        ready_timeout_sec: float,
        heartbeat_sec: float,
        show_engine_stderr: bool,
    ) -> None:
        env = os.environ.copy()
        env.update(
            SERVE="1",
            SNAP=str(model),
            CTX=str(context),
            RAM_GB=f"{memory_gb:g}",
            QWEN_PREFIX_CACHE_MB=f"{cache_mb:g}",
            QWEN_PREFIX_CACHE_MIN_TOKENS=str(min_prefix_tokens),
            QWEN_PREFIX_LOG="1",
        )
        self.label = label
        self.heartbeat_sec = heartbeat_sec
        self.show_engine_stderr = show_engine_stderr
        self._spawned = time.perf_counter()
        progress(
            f"{label}: spawning qwen_moe "
            f"(RAM_GB={memory_gb:g}, prefix_cache={cache_mb:g}MiB)"
        )
        # argv cap=0 deliberately bypasses any inherited CACHE setting and
        # selects Qwen's RAM_GB -> topk path. Passing a positive positional cap
        # would make prefix memory additive instead of fixed-total-memory.
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
        self._wait_ready(ready_timeout_sec)

    def _wait_ready(self, timeout_sec: float) -> None:
        """Wait for READY without blocking past the advertised timeout."""
        assert self.process.stdout is not None
        lines: queue.Queue[bytes | None] = queue.Queue()

        def read_until_ready() -> None:
            assert self.process.stdout is not None
            try:
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        lines.put(None)
                        return
                    lines.put(line)
                    if READY in line:
                        return
            except Exception:
                lines.put(None)

        reader = threading.Thread(target=read_until_ready, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout_sec
        with Heartbeat(f"{self.label}: loading model / waiting for READY", self.heartbeat_sec):
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.process.kill()
                    self.process.wait(timeout=10)
                    raise RuntimeError(
                        f"qwen_moe did not report READY within {timeout_sec:g}s:\n"
                        + "".join(self._stderr)[-6000:]
                    )
                try:
                    line = lines.get(timeout=min(1.0, remaining))
                except queue.Empty:
                    if self.process.poll() is not None:
                        raise RuntimeError(
                            "qwen_moe exited before READY:\n"
                            + "".join(self._stderr)[-6000:]
                        )
                    continue
                if line is None:
                    raise RuntimeError(
                        "qwen_moe exited before READY:\n"
                        + "".join(self._stderr)[-6000:]
                    )
                if READY in line:
                    reader.join(timeout=1)
                    progress(
                        f"{self.label}: READY after "
                        f"{time.perf_counter() - self._spawned:.2f}s"
                    )
                    return
                # Pre-READY stdout is metadata/noise; the protocol only becomes
                # request-capable once READY has been observed.

    def request(self, request_id: str, prompt: bytes, max_tokens: int) -> dict[str, object]:
        assert self.process.stdin is not None and self.process.stdout is not None
        header = (
            f"SUBMIT {request_id} 0 {len(prompt)} {max_tokens} 0.0 1.0\n"
        ).encode("ascii")
        began = time.perf_counter()
        progress(
            f"{self.label}: {request_id} submit "
            f"({len(prompt)} bytes, max_tokens={max_tokens})"
        )
        self.process.stdin.write(header + prompt + b"\n")
        self.process.stdin.flush()

        output: list[bytes] = []
        data_times: list[float] = []
        done_fields: list[str] | None = None
        accepted = False
        with Heartbeat(f"{self.label}: {request_id} running", self.heartbeat_sec):
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
                if kind == "ACCEPT" and not accepted:
                    accepted = True
                    progress(
                        f"{self.label}: {request_id} accepted; "
                        "prefill/decode in progress"
                    )
                elif kind == "DATA" and len(fields) == 3:
                    data_times.append(time.perf_counter() - began)
                    size = int(fields[2])
                    output.append(self.process.stdout.read(size))
                    self.process.stdout.read(1)  # DATA payload trailing newline
                    if len(data_times) == 1:
                        progress(
                            f"{self.label}: {request_id} first token/data at "
                            f"{data_times[0]:.3f}s"
                        )
                elif kind == "DONE":
                    done_fields = fields
                    break
                # PROF/STAT/HWINFO/TIERS/EMAP are protocol metadata.

        done_sec = time.perf_counter() - began
        first_response_sec = data_times[0] if data_times else done_sec
        second_response_sec = data_times[1] if len(data_times) > 1 else None
        progress(
            f"{self.label}: {request_id} DONE in {done_sec:.3f}s "
            f"({len(data_times)} DATA frames)"
        )
        return {
            "request_id": request_id,
            "first_response_sec": first_response_sec,
            "second_response_sec": second_response_sec,
            "first_to_second_sec": (
                second_response_sec - first_response_sec
                if second_response_sec is not None else None
            ),
            "data_frames": len(data_times),
            "done_sec": done_sec,
            "output_hex": b"".join(output).hex(),
            "done_fields": done_fields,
        }

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for raw in iter(self.process.stderr.readline, b""):
            line = raw.decode("utf-8", "replace")
            self._stderr.append(line)
            if self.show_engine_stderr:
                sys.stderr.write(f"[qwen-engine {self.label}] {line}")
                sys.stderr.flush()

    def close(self) -> str:
        progress(f"{self.label}: shutting down qwen_moe")
        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=60)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=10)
        self._stderr_thread.join(timeout=10)
        progress(
            f"{self.label}: process exited rc={self.process.returncode}; "
            f"lifetime={time.perf_counter() - self._spawned:.2f}s"
        )
        return "".join(self._stderr)


def metric_for(stderr: str, request_id: str) -> dict[str, object]:
    found = [
        m for line in stderr.splitlines()
        if (m := parse_metric_line(line)) and m.get("request") == request_id
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"expected one QWEN-PREFIX metric for {request_id!r}, got {len(found)}:\n"
            + stderr[-8000:]
        )
    return found[0]


def run_mode(
    *,
    mode: str,
    label: str,
    binary: Path,
    model: Path,
    prefix: bytes,
    extended: bytes,
    memory_gb: float,
    cache_mb: float,
    min_prefix_tokens: int,
    context: int,
    max_tokens: int,
    ready_timeout_sec: float,
    heartbeat_sec: float,
    show_engine_stderr: bool,
) -> dict[str, object]:
    enabled = mode == "cache_on"
    total_began = time.perf_counter()
    serve = Serve(
        binary,
        model,
        label=label,
        memory_gb=memory_gb,
        cache_mb=cache_mb if enabled else 0.0,
        min_prefix_tokens=min_prefix_tokens,
        context=context,
        ready_timeout_sec=ready_timeout_sec,
        heartbeat_sec=heartbeat_sec,
        show_engine_stderr=show_engine_stderr,
    )
    try:
        warm = serve.request("warm", prefix, max_tokens)
        measured = serve.request("measure", extended, max_tokens)
    finally:
        stderr = serve.close()
    warm_metric = metric_for(stderr, "warm")
    measured_metric = metric_for(stderr, "measure")
    return {
        "mode": mode,
        "process_wall_sec": time.perf_counter() - total_began,
        "warm": warm,
        "measure": measured,
        "warm_metric": warm_metric,
        "measure_metric": measured_metric,
    }


def run_mode_reused(
    *,
    mode: str,
    label: str,
    trials: int,
    binary: Path,
    model: Path,
    prefix: bytes,
    extended: bytes,
    memory_gb: float,
    cache_mb: float,
    min_prefix_tokens: int,
    context: int,
    max_tokens: int,
    ready_timeout_sec: float,
    heartbeat_sec: float,
    show_engine_stderr: bool,
) -> list[dict[str, object]]:
    """One model startup, one warm P, then N P+X measurements."""
    enabled = mode == "cache_on"
    total_began = time.perf_counter()
    serve = Serve(
        binary,
        model,
        label=label,
        memory_gb=memory_gb,
        cache_mb=cache_mb if enabled else 0.0,
        min_prefix_tokens=min_prefix_tokens,
        context=context,
        ready_timeout_sec=ready_timeout_sec,
        heartbeat_sec=heartbeat_sec,
        show_engine_stderr=show_engine_stderr,
    )
    measured: list[dict[str, object]] = []
    try:
        warm = serve.request("warm", prefix, max_tokens)
        for i in range(trials):
            rid = f"measure_{i + 1}"
            progress(f"{label}: steady-state measurement {i + 1}/{trials}")
            measured.append(serve.request(rid, extended, max_tokens))
    finally:
        stderr = serve.close()

    wall = time.perf_counter() - total_began
    warm_metric = metric_for(stderr, "warm")
    records: list[dict[str, object]] = []
    for i, request in enumerate(measured):
        rid = f"measure_{i + 1}"
        records.append({
            "mode": mode,
            "process_wall_sec": wall,
            "shared_process": True,
            "warm": warm,
            "measure": request,
            "warm_metric": warm_metric,
            "measure_metric": metric_for(stderr, rid),
        })
    return records


def output_preview(record: dict[str, object]) -> str:
    raw = bytes.fromhex(str(record["measure"]["output_hex"]))
    text = raw.decode("utf-8", "replace")
    return repr(text)


def save_failure(
    output_path: str | None,
    *,
    trial: int,
    reason: str,
    off: dict[str, object],
    on: dict[str, object],
    execution_mode: str,
) -> None:
    progress(f"CORRECTNESS FAILURE trial={trial}: {reason}")
    progress(f"cache_off output: {output_preview(off)}")
    progress(f"cache_on  output: {output_preview(on)}")
    if not output_path:
        return
    diagnostic = {
        "schema": "colibri.qwen.prefix_cache_benchmark_failure.v1",
        "execution_mode": execution_mode,
        "trial": trial,
        "error": reason,
        "cache_off": off,
        "cache_on": on,
    }
    path = Path(output_path).expanduser()
    path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress(f"partial diagnostic written to {path}")


def validate_pair(
    *,
    trial: int,
    off: dict[str, object],
    on: dict[str, object],
    output_path: str | None,
    execution_mode: str,
) -> dict[str, object]:
    off_m = off["measure_metric"]
    on_m = on["measure_metric"]
    for field in ("prompt",):
        if int(off_m[field]) != int(on_m[field]):
            reason = f"token workload differs for {field}"
            save_failure(
                output_path, trial=trial, reason=reason, off=off, on=on,
                execution_mode=execution_mode,
            )
            raise RuntimeError(f"trial {trial}: {reason}")

    if int(off_m.get("matched", -1)) != 0:
        reason = "cache-off unexpectedly reused a prefix"
        save_failure(
            output_path, trial=trial, reason=reason, off=off, on=on,
            execution_mode=execution_mode,
        )
        raise RuntimeError(f"trial {trial}: {reason}")

    warm_tokens = int(on["warm_metric"]["prompt"])
    if int(on_m.get("matched", -1)) != warm_tokens:
        reason = (
            f"cache-on matched {on_m.get('matched')} of {warm_tokens} warm tokens"
        )
        save_failure(
            output_path, trial=trial, reason=reason, off=off, on=on,
            execution_mode=execution_mode,
        )
        raise RuntimeError(f"trial {trial}: {reason}")

    if off["measure"]["output_hex"] != on["measure"]["output_hex"]:
        reason = "cache-on output differs from cache-off"
        save_failure(
            output_path, trial=trial, reason=reason, off=off, on=on,
            execution_mode=execution_mode,
        )
        raise RuntimeError(
            f"trial {trial}: {reason}; see output previews above"
            + (f" and {Path(output_path).expanduser()}" if output_path else "")
        )

    if (
        off["measure"]["second_response_sec"] is None
        or on["measure"]["second_response_sec"] is None
    ):
        reason = (
            "benchmark needs at least two emitted tokens to measure the "
            "post-first-token capture stall"
        )
        save_failure(
            output_path, trial=trial, reason=reason, off=off, on=on,
            execution_mode=execution_mode,
        )
        raise RuntimeError(f"trial {trial}: {reason}")

    off_ttft = float(off["measure"]["first_response_sec"])
    on_ttft = float(on["measure"]["first_response_sec"])
    off_gap = float(off["measure"]["first_to_second_sec"])
    on_gap = float(on["measure"]["first_to_second_sec"])
    return {
        "trial": trial,
        "cache_off": off,
        "cache_on": on,
        "ttft_off_sec": off_ttft,
        "ttft_on_sec": on_ttft,
        "ttft_saved_sec": off_ttft - on_ttft,
        "ttft_speedup": off_ttft / on_ttft if on_ttft > 0 else None,
        "first_to_second_off_sec": off_gap,
        "first_to_second_on_sec": on_gap,
        "first_to_second_penalty_sec": on_gap - off_gap,
        "completion_off_sec": float(off["measure"]["done_sec"]),
        "completion_on_sec": float(on["measure"]["done_sec"]),
    }


def main() -> int:
    global _PROGRESS
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
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--fast", "--reuse-processes", dest="reuse_processes", action="store_true",
        help=(
            "start one process per cache mode and reuse it for all trials; much "
            "faster on large models, but measures warm-process steady state"
        ),
    )
    parser.add_argument(
        "--heartbeat-sec", type=float, default=15.0,
        help="print a progress heartbeat while model load/prefill is quiet (0 disables)",
    )
    parser.add_argument(
        "--ready-timeout", type=float, default=300.0,
        help="hard timeout in seconds for qwen_moe READY (default: 300)",
    )
    parser.add_argument(
        "--show-engine-stderr", action="store_true",
        help="stream qwen_moe stderr live in addition to benchmark progress",
    )
    parser.add_argument("--quiet", action="store_true", help="disable progress output")
    parser.add_argument("--repo")
    parser.add_argument("--output")
    args = parser.parse_args()
    _PROGRESS = not args.quiet

    if args.memory_gb <= 0 or args.cache_mb <= 0:
        parser.error("--memory-gb and --cache-mb must be positive")
    if (
        args.min_prefix_tokens < 1 or args.context < 2
        or args.max_tokens < 2 or args.trials < 1
    ):
        parser.error("min-prefix/trials must be positive, context >=2, and max-tokens >=2")
    if args.prefix_tokens < 1 or args.chars_per_token <= 0:
        parser.error("--prefix-tokens and --chars-per-token must be positive")
    if args.heartbeat_sec < 0 or args.ready_timeout <= 0:
        parser.error("--heartbeat-sec must be >=0 and --ready-timeout must be positive")

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

    execution_mode = "reused_processes" if args.reuse_processes else "fresh_process_pairs"
    progress(
        f"starting {execution_mode}: trials={args.trials}, "
        f"prefix≈{args.prefix_tokens} tokens/{len(prefix)} bytes, "
        f"model={model.name}"
    )
    if args.reuse_processes:
        progress(
            "FAST mode: 2 model startups total; results are warm-process/steady-state "
            "and should not replace the final fresh-process A/B"
        )

    pairs: list[dict[str, object]] = []
    started = time.perf_counter()

    if args.reuse_processes:
        off_records = run_mode_reused(
            mode="cache_off", label="cache_off", trials=args.trials,
            binary=binary, model=model, prefix=prefix, extended=extended,
            memory_gb=args.memory_gb, cache_mb=args.cache_mb,
            min_prefix_tokens=args.min_prefix_tokens, context=args.context,
            max_tokens=args.max_tokens, ready_timeout_sec=args.ready_timeout,
            heartbeat_sec=args.heartbeat_sec,
            show_engine_stderr=args.show_engine_stderr,
        )
        on_records = run_mode_reused(
            mode="cache_on", label="cache_on", trials=args.trials,
            binary=binary, model=model, prefix=prefix, extended=extended,
            memory_gb=args.memory_gb, cache_mb=args.cache_mb,
            min_prefix_tokens=args.min_prefix_tokens, context=args.context,
            max_tokens=args.max_tokens, ready_timeout_sec=args.ready_timeout,
            heartbeat_sec=args.heartbeat_sec,
            show_engine_stderr=args.show_engine_stderr,
        )
        for i, (off, on) in enumerate(zip(off_records, on_records), start=1):
            pair = validate_pair(
                trial=i, off=off, on=on, output_path=args.output,
                execution_mode=execution_mode,
            )
            pair["order"] = ["cache_off", "cache_on"]
            pairs.append(pair)
    else:
        for trial_index in range(args.trials):
            trial = trial_index + 1
            order = (
                ["cache_off", "cache_on"]
                if trial_index % 2 == 0
                else ["cache_on", "cache_off"]
            )
            observed: dict[str, dict[str, object]] = {}
            for mode in order:
                progress(f"trial={trial}/{args.trials} mode={mode}")
                observed[mode] = run_mode(
                    mode=mode, label=f"trial{trial}:{mode}",
                    binary=binary, model=model, prefix=prefix, extended=extended,
                    memory_gb=args.memory_gb, cache_mb=args.cache_mb,
                    min_prefix_tokens=args.min_prefix_tokens, context=args.context,
                    max_tokens=args.max_tokens, ready_timeout_sec=args.ready_timeout,
                    heartbeat_sec=args.heartbeat_sec,
                    show_engine_stderr=args.show_engine_stderr,
                )
            pair = validate_pair(
                trial=trial, off=observed["cache_off"], on=observed["cache_on"],
                output_path=args.output, execution_mode=execution_mode,
            )
            pair["order"] = order
            pairs.append(pair)

    off_values = [float(p["ttft_off_sec"]) for p in pairs]
    on_values = [float(p["ttft_on_sec"]) for p in pairs]
    speedups = [
        float(p["ttft_speedup"]) for p in pairs
        if p["ttft_speedup"] is not None
    ]
    gap_off_values = [float(p["first_to_second_off_sec"]) for p in pairs]
    gap_on_values = [float(p["first_to_second_on_sec"]) for p in pairs]
    completion_off_values = [float(p["completion_off_sec"]) for p in pairs]
    completion_on_values = [float(p["completion_on_sec"]) for p in pairs]
    sample_metric = pairs[0]["cache_on"]["measure_metric"]
    result = {
        "schema": "colibri.qwen.prefix_cache_benchmark.v1",
        "execution_mode": execution_mode,
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
            "ttft_saved_median_sec": (
                statistics.median(off_values) - statistics.median(on_values)
            ),
            "ttft_speedup_median": statistics.median(speedups),
            "first_to_second_off_median_sec": statistics.median(gap_off_values),
            "first_to_second_on_median_sec": statistics.median(gap_on_values),
            "first_to_second_penalty_median_sec": (
                statistics.median(gap_on_values) - statistics.median(gap_off_values)
            ),
            "completion_off_median_sec": statistics.median(completion_off_values),
            "completion_on_median_sec": statistics.median(completion_on_values),
            "completion_delta_median_sec": (
                statistics.median(completion_on_values)
                - statistics.median(completion_off_values)
            ),
            "restore_ms_median": statistics.median(
                float(p["cache_on"]["measure_metric"]["restore_ms"]) for p in pairs
            ),
            "tail_prefill_ms_median": statistics.median(
                float(p["cache_on"]["measure_metric"]["prefill_ms"]) for p in pairs
            ),
            "capture_ms_median": statistics.median(
                float(p["cache_on"]["measure_metric"]["capture_ms"]) for p in pairs
            ),
        },
        "pairs": pairs,
        "wall_sec": time.perf_counter() - started,
    }

    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        path = Path(args.output).expanduser()
        path.write_text(encoded + "\n", encoding="utf-8")
        progress(f"result written to {path}")
    progress(f"benchmark complete in {result['wall_sec']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
