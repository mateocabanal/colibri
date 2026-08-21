#!/usr/bin/env python3
"""Local Qwen prefix-cache tier benchmark.

Measures four things without GitHub Actions or a long-lived server:
  1. cache-off cold P+X
  2. RAM-only seed P followed by a same-process P+X hit
  3. SSD-only cold seed P
  4. SSD-only P+X hit after qwen_moe has been fully restarted

The SSD phase uses a fresh run-specific cache directory, so a successful
ssd_hit cannot be a stale result from an earlier benchmark. Qwen's current
persistent-prefix safety contract requires CPU compute; this tool therefore
sets QWEN_METAL_COMPUTE=0 for every phase so the comparisons stay equivalent.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import benchmark_qwen_prefix_cache as base

MANAGED_ENV = (
    "COLI_PREFIX_CACHE",
    "COLI_PREFIX_CACHE_RAM_MB",
    "COLI_PREFIX_CACHE_DISK_GB",
    "COLI_PREFIX_CACHE_DIR",
    "COLI_PREFIX_CACHE_MIN_TOKENS",
    "QWEN_METAL_COMPUTE",
)


def progress(message: str) -> None:
    sys.stderr.write(f"[prefix-tier-bench] {message}\n")
    sys.stderr.flush()


@contextmanager
def cache_environment(
    *,
    mode: str,
    ram_mb: float,
    disk_gb: float,
    cache_dir: Path,
    min_prefix_tokens: int,
):
    old = {name: os.environ.get(name) for name in MANAGED_ENV}
    try:
        os.environ.update(
            COLI_PREFIX_CACHE=mode,
            COLI_PREFIX_CACHE_RAM_MB=f"{ram_mb:g}",
            COLI_PREFIX_CACHE_DISK_GB=f"{disk_gb:g}",
            COLI_PREFIX_CACHE_DIR=str(cache_dir),
            COLI_PREFIX_CACHE_MIN_TOKENS=str(min_prefix_tokens),
            # Current Qwen persistent snapshots intentionally reject explicit
            # Metal compute. Keep every tier on the same safe CPU path.
            QWEN_METAL_COMPUTE="0",
        )
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_requests(
    *,
    mode: str,
    label: str,
    requests: list[tuple[str, bytes]],
    binary: Path,
    model: Path,
    memory_gb: float,
    ram_mb: float,
    disk_gb: float,
    cache_dir: Path,
    min_prefix_tokens: int,
    context: int,
    max_tokens: int,
    ready_timeout_sec: float,
    heartbeat_sec: float,
    show_engine_stderr: bool,
) -> dict[str, object]:
    with cache_environment(
        mode=mode,
        ram_mb=ram_mb,
        disk_gb=disk_gb,
        cache_dir=cache_dir,
        min_prefix_tokens=min_prefix_tokens,
    ):
        serve = base.Serve(
            binary,
            model,
            label=label,
            memory_gb=memory_gb,
            # Must stay non-zero for the Qwen compatibility gate. In SSD mode
            # the global mode still forces the RAM budget to zero.
            cache_mb=ram_mb,
            min_prefix_tokens=min_prefix_tokens,
            context=context,
            ready_timeout_sec=ready_timeout_sec,
            heartbeat_sec=heartbeat_sec,
            show_engine_stderr=show_engine_stderr,
        )
        began = time.perf_counter()
        records: dict[str, object] = {}
        try:
            for request_id, prompt in requests:
                records[request_id] = serve.request(request_id, prompt, max_tokens)
        finally:
            stderr = serve.close()

    metrics = {
        request_id: base.metric_for(stderr, request_id)
        for request_id, _ in requests
    }
    return {
        "mode": mode,
        "wall_sec": time.perf_counter() - began,
        "requests": records,
        "metrics": metrics,
    }


def cache_files(cache_dir: Path) -> tuple[int, int]:
    files = [path for path in cache_dir.glob("cpfx-*.bin") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def require_match(
    label: str,
    metric: dict[str, object],
    expected: int,
) -> None:
    got = int(metric.get("matched", -1))
    if got != expected:
        raise RuntimeError(f"{label}: expected matched={expected}, got {got}")


def ttft(phase: dict[str, object], request_id: str) -> float:
    requests = phase["requests"]
    assert isinstance(requests, dict)
    request = requests[request_id]
    assert isinstance(request, dict)
    return float(request["first_response_sec"])


def metric(phase: dict[str, object], request_id: str) -> dict[str, object]:
    metrics = phase["metrics"]
    assert isinstance(metrics, dict)
    result = metrics[request_id]
    assert isinstance(result, dict)
    return result


def request(phase: dict[str, object], request_id: str) -> dict[str, object]:
    requests = phase["requests"]
    assert isinstance(requests, dict)
    result = requests[request_id]
    assert isinstance(result, dict)
    return result


def default_cache_root() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "colibri" / "prefix-tier-bench"
    if os.name == "nt":
        base_dir = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP")
        if base_dir:
            return Path(base_dir) / "Colibri" / "prefix-tier-bench"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "colibri" / "prefix-tier-bench"
    return home / ".cache" / "colibri" / "prefix-tier-bench"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./qwen_moe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-gb", type=float, required=True)
    parser.add_argument("--prefix-file")
    parser.add_argument("--tail-file")
    parser.add_argument(
        "--prefix-tokens",
        type=int,
        default=256,
        help="approximate generated prefix size; 256 is the local smoke default",
    )
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--ram-cache-mb", type=float, default=256.0)
    parser.add_argument("--disk-cache-gb", type=float, default=1.0)
    parser.add_argument("--min-prefix-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--heartbeat-sec", type=float, default=15.0)
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--show-engine-stderr", action="store_true")
    parser.add_argument("--cache-root")
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="delete this benchmark's run-specific SSD cache after recording results",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print the local test plan and exit before starting qwen_moe",
    )
    parser.add_argument("--repo")
    parser.add_argument("--output", default="prefix-tier-benchmark.json")
    args = parser.parse_args()

    if args.memory_gb <= 0:
        parser.error("--memory-gb must be positive")
    if args.ram_cache_mb <= 0:
        parser.error("--ram-cache-mb must be positive")
    if args.disk_cache_gb <= 0:
        parser.error("--disk-cache-gb must be positive")
    if args.min_prefix_tokens < 1 or args.context < 2 or args.max_tokens < 2:
        parser.error("min-prefix must be positive, context >=2, and max-tokens >=2")
    if args.prefix_tokens < 1 or args.chars_per_token <= 0:
        parser.error("--prefix-tokens and --chars-per-token must be positive")
    if args.heartbeat_sec < 0 or args.ready_timeout <= 0:
        parser.error("--heartbeat-sec must be >=0 and --ready-timeout must be positive")

    binary = Path(args.binary).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve() if args.repo else binary.parent.parent
    cache_root = (
        Path(args.cache_root).expanduser().resolve()
        if args.cache_root
        else default_cache_root()
    )
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    cache_dir = cache_root / run_id

    if args.prefix_file:
        prefix = Path(args.prefix_file).expanduser().read_bytes()
    else:
        prefix = base.make_prompt(args.prefix_tokens, args.chars_per_token)
    if args.tail_file:
        tail = Path(args.tail_file).expanduser().read_bytes()
    else:
        tail = base.TAIL_TEXT.encode("utf-8")
    if not prefix or not tail:
        parser.error("prefix and tail must both be non-empty")
    extended = prefix + tail

    plan = {
        "binary": str(binary),
        "model": str(model),
        "memory_gb": args.memory_gb,
        "prefix_bytes": len(prefix),
        "tail_bytes": len(tail),
        "ram_cache_mb": args.ram_cache_mb,
        "disk_cache_gb": args.disk_cache_gb,
        "min_prefix_tokens": args.min_prefix_tokens,
        "cache_dir": str(cache_dir),
        "phases": [
            "cache off: cold P+X",
            "RAM only: cold P then same-process P+X hit",
            "SSD only: cold P seed",
            "SSD only: restart qwen_moe then P+X persistent hit",
        ],
        "forced_environment": {"QWEN_METAL_COMPUTE": "0"},
    }
    if args.plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    if not model.exists():
        parser.error(f"model does not exist: {model}")

    cache_dir.mkdir(parents=True, exist_ok=False)
    progress(f"SSD benchmark directory: {cache_dir}")
    started = time.perf_counter()

    try:
        progress("1/4 cache-off cold P+X")
        cold = run_requests(
            mode="off",
            label="tier:cold",
            requests=[("cold", extended)],
            binary=binary,
            model=model,
            memory_gb=args.memory_gb,
            ram_mb=args.ram_cache_mb,
            disk_gb=args.disk_cache_gb,
            cache_dir=cache_dir,
            min_prefix_tokens=args.min_prefix_tokens,
            context=args.context,
            max_tokens=args.max_tokens,
            ready_timeout_sec=args.ready_timeout,
            heartbeat_sec=args.heartbeat_sec,
            show_engine_stderr=args.show_engine_stderr,
        )
        require_match("cold", metric(cold, "cold"), 0)

        progress("2/4 RAM-only seed P -> same-process P+X")
        ram = run_requests(
            mode="ram",
            label="tier:ram",
            requests=[("ram_seed", prefix), ("ram_hit", extended)],
            binary=binary,
            model=model,
            memory_gb=args.memory_gb,
            ram_mb=args.ram_cache_mb,
            disk_gb=args.disk_cache_gb,
            cache_dir=cache_dir,
            min_prefix_tokens=args.min_prefix_tokens,
            context=args.context,
            max_tokens=args.max_tokens,
            ready_timeout_sec=args.ready_timeout,
            heartbeat_sec=args.heartbeat_sec,
            show_engine_stderr=args.show_engine_stderr,
        )
        ram_seed_tokens = int(metric(ram, "ram_seed")["prompt"])
        require_match("ram_seed", metric(ram, "ram_seed"), 0)
        require_match("ram_hit", metric(ram, "ram_hit"), ram_seed_tokens)

        progress("3/4 SSD-only cold seed P")
        ssd_seed = run_requests(
            mode="ssd",
            label="tier:ssd-seed",
            requests=[("ssd_seed", prefix)],
            binary=binary,
            model=model,
            memory_gb=args.memory_gb,
            ram_mb=args.ram_cache_mb,
            disk_gb=args.disk_cache_gb,
            cache_dir=cache_dir,
            min_prefix_tokens=args.min_prefix_tokens,
            context=args.context,
            max_tokens=args.max_tokens,
            ready_timeout_sec=args.ready_timeout,
            heartbeat_sec=args.heartbeat_sec,
            show_engine_stderr=args.show_engine_stderr,
        )
        ssd_seed_tokens = int(metric(ssd_seed, "ssd_seed")["prompt"])
        require_match("ssd_seed", metric(ssd_seed, "ssd_seed"), 0)
        files_after_seed, bytes_after_seed = cache_files(cache_dir)
        if files_after_seed < 1 or bytes_after_seed < 1:
            raise RuntimeError(
                "SSD seed completed but no cpfx-*.bin object was published; "
                "run with --show-engine-stderr and confirm persistent caching is enabled"
            )

        progress(
            f"SSD seed published {files_after_seed} object(s), "
            f"{bytes_after_seed / (1024 * 1024):.1f} MiB"
        )
        progress("4/4 restart qwen_moe -> SSD-only P+X persistent hit")
        ssd_hit = run_requests(
            mode="ssd",
            label="tier:ssd-hit",
            requests=[("ssd_hit", extended)],
            binary=binary,
            model=model,
            memory_gb=args.memory_gb,
            ram_mb=args.ram_cache_mb,
            disk_gb=args.disk_cache_gb,
            cache_dir=cache_dir,
            min_prefix_tokens=args.min_prefix_tokens,
            context=args.context,
            max_tokens=args.max_tokens,
            ready_timeout_sec=args.ready_timeout,
            heartbeat_sec=args.heartbeat_sec,
            show_engine_stderr=args.show_engine_stderr,
        )
        require_match("ssd_hit", metric(ssd_hit, "ssd_hit"), ssd_seed_tokens)

        cold_output = request(cold, "cold")["output_hex"]
        ram_output = request(ram, "ram_hit")["output_hex"]
        ssd_output = request(ssd_hit, "ssd_hit")["output_hex"]
        if cold_output != ram_output:
            raise RuntimeError("RAM-hit output differs from cache-off output")
        if cold_output != ssd_output:
            raise RuntimeError("SSD-hit output differs from cache-off output")

        cold_ttft = ttft(cold, "cold")
        ram_ttft = ttft(ram, "ram_hit")
        ssd_ttft = ttft(ssd_hit, "ssd_hit")
        result = {
            "schema": "colibri.qwen.prefix_cache_tiers.v1",
            "git_sha": base.git_sha(repo),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "binary": str(binary),
            "model": str(model),
            "memory_gb": args.memory_gb,
            "context": args.context,
            "max_tokens": args.max_tokens,
            "prefix_bytes": len(prefix),
            "tail_bytes": len(tail),
            "cache": {
                "ram_mb": args.ram_cache_mb,
                "disk_gb": args.disk_cache_gb,
                "min_prefix_tokens": args.min_prefix_tokens,
                "directory": str(cache_dir),
                "objects_after_seed": files_after_seed,
                "bytes_after_seed": bytes_after_seed,
            },
            "environment": {
                key: os.environ[key]
                for key in ("COLI_CONFIG", "SNAP", "QWEN_KV_F16")
                if key in os.environ
            },
            "forced_environment": {"QWEN_METAL_COMPUTE": "0"},
            "phases": {
                "cold": cold,
                "ram": ram,
                "ssd_seed": ssd_seed,
                "ssd_hit": ssd_hit,
            },
            "summary": {
                "matched_tokens": ssd_seed_tokens,
                "cold_ttft_sec": cold_ttft,
                "ram_hit_ttft_sec": ram_ttft,
                "ssd_hit_ttft_sec": ssd_ttft,
                "ram_speedup_vs_cold": cold_ttft / ram_ttft if ram_ttft > 0 else None,
                "ssd_speedup_vs_cold": cold_ttft / ssd_ttft if ssd_ttft > 0 else None,
                "ssd_penalty_vs_ram_sec": ssd_ttft - ram_ttft,
                "ram_restore_ms": float(metric(ram, "ram_hit")["restore_ms"]),
                "ssd_restore_ms": float(metric(ssd_hit, "ssd_hit")["restore_ms"]),
                "ram_tail_prefill_ms": float(metric(ram, "ram_hit")["prefill_ms"]),
                "ssd_tail_prefill_ms": float(metric(ssd_hit, "ssd_hit")["prefill_ms"]),
            },
            "wall_sec": time.perf_counter() - started,
        }

        output_path = Path(args.output).expanduser()
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        summary = result["summary"]
        print("\nPrefix cache tier results")
        print("=" * 72)
        print(f"{'case':<18} {'matched':>10} {'TTFT (s)':>12} {'restore (ms)':>15}")
        print("-" * 72)
        print(
            f"{'cold/off':<18} {0:>10} {cold_ttft:>12.3f} "
            f"{float(metric(cold, 'cold')['restore_ms']):>15.3f}"
        )
        print(
            f"{'RAM hit':<18} {ram_seed_tokens:>10} {ram_ttft:>12.3f} "
            f"{float(metric(ram, 'ram_hit')['restore_ms']):>15.3f}"
        )
        print(
            f"{'SSD hit/restart':<18} {ssd_seed_tokens:>10} {ssd_ttft:>12.3f} "
            f"{float(metric(ssd_hit, 'ssd_hit')['restore_ms']):>15.3f}"
        )
        print("-" * 72)
        print(f"RAM speedup vs cold: {summary['ram_speedup_vs_cold']:.2f}x")
        print(f"SSD speedup vs cold: {summary['ssd_speedup_vs_cold']:.2f}x")
        print(f"SSD penalty vs RAM:   {summary['ssd_penalty_vs_ram_sec']:.3f}s")
        print(f"JSON: {output_path}")
        print(f"SSD cache: {cache_dir}")
        return 0
    finally:
        if args.cleanup_cache and cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
