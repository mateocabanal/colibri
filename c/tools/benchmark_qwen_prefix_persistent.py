#!/usr/bin/env python3
"""Local restart-persistence benchmark for Qwen prompt-prefix snapshots.

This is the cheap follow-up to benchmark_qwen_prefix_tiers.py. It focuses only
on the persistent tier:

  1. optional cache-off P+X correctness baseline
  2. SSD-only seed of P
  3. full qwen_moe restart
  4. SSD-only P+X restore after restart

The cache directory is fresh for every invocation, so the final hit proves that
state written by the seed process survived process teardown. On a backend-aware
runtime, --compute metal also proves Metal -> restart -> Metal persistence.
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
    "COLI_PREFIX_CACHE_DISK_GB",
    "COLI_PREFIX_CACHE_DIR",
    "COLI_PREFIX_CACHE_MIN_TOKENS",
    "QWEN_METAL_COMPUTE",
    "QWEN_CUDA_COMPUTE",
)


def progress(message: str) -> None:
    sys.stderr.write(f"[prefix-persist-bench] {message}\n")
    sys.stderr.flush()


@contextmanager
def runtime_environment(
    *,
    mode: str,
    compute: str,
    disk_gb: float,
    cache_dir: Path,
    min_prefix_tokens: int,
):
    old = {name: os.environ.get(name) for name in MANAGED_ENV}
    try:
        os.environ.update(
            COLI_PREFIX_CACHE=mode,
            COLI_PREFIX_CACHE_DISK_GB=f"{disk_gb:g}",
            COLI_PREFIX_CACHE_DIR=str(cache_dir),
            COLI_PREFIX_CACHE_MIN_TOKENS=str(min_prefix_tokens),
        )
        if compute == "metal":
            os.environ["QWEN_METAL_COMPUTE"] = "1"
            os.environ["QWEN_CUDA_COMPUTE"] = "0"
        elif compute == "cuda":
            os.environ["QWEN_METAL_COMPUTE"] = "0"
            os.environ["QWEN_CUDA_COMPUTE"] = "1"
        else:
            os.environ["QWEN_METAL_COMPUTE"] = "0"
            os.environ["QWEN_CUDA_COMPUTE"] = "0"
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_one(
    *,
    mode: str,
    compute: str,
    label: str,
    request_id: str,
    prompt: bytes,
    binary: Path,
    model: Path,
    memory_gb: float,
    compatibility_cache_mb: float,
    disk_gb: float,
    cache_dir: Path,
    min_prefix_tokens: int,
    context: int,
    max_tokens: int,
    ready_timeout_sec: float,
    heartbeat_sec: float,
    show_engine_stderr: bool,
) -> dict[str, object]:
    with runtime_environment(
        mode=mode,
        compute=compute,
        disk_gb=disk_gb,
        cache_dir=cache_dir,
        min_prefix_tokens=min_prefix_tokens,
    ):
        serve = base.Serve(
            binary,
            model,
            label=label,
            memory_gb=memory_gb,
            # The common COLI mode owns tier selection. Keep the legacy Qwen
            # cap non-zero so it cannot explicitly veto SSD persistence.
            cache_mb=compatibility_cache_mb,
            min_prefix_tokens=min_prefix_tokens,
            context=context,
            ready_timeout_sec=ready_timeout_sec,
            heartbeat_sec=heartbeat_sec,
            show_engine_stderr=show_engine_stderr,
        )
        began = time.perf_counter()
        try:
            record = serve.request(request_id, prompt, max_tokens)
        finally:
            stderr = serve.close()

    if compute == "metal" and "Metal unavailable" in stderr:
        raise RuntimeError(
            "QWEN_METAL_COMPUTE=1 fell back to CPU; this run does not validate "
            "Metal persistence"
        )
    if compute == "cuda" and "CUDA/HIP unavailable" in stderr:
        raise RuntimeError(
            "QWEN_CUDA_COMPUTE=1 fell back to CPU; this run does not validate "
            "GPU persistence"
        )

    return {
        "mode": mode,
        "compute": compute,
        "wall_sec": time.perf_counter() - began,
        "request": record,
        "metric": base.metric_for(stderr, request_id),
    }


def default_cache_root() -> Path:
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "colibri" / "prefix-persist-bench"
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP")
        if root:
            return Path(root) / "Colibri" / "prefix-persist-bench"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "colibri" / "prefix-persist-bench"
    return home / ".cache" / "colibri" / "prefix-persist-bench"


def cache_files(cache_dir: Path) -> tuple[int, int]:
    objects = [p for p in cache_dir.glob("cpfx-*.bin") if p.is_file()]
    return len(objects), sum(p.stat().st_size for p in objects)


def main() -> int:
    default_compute = "metal" if sys.platform == "darwin" else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./qwen_moe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-gb", type=float, required=True)
    parser.add_argument("--compute", choices=("cpu", "metal", "cuda"), default=default_compute)
    parser.add_argument("--prefix-file")
    parser.add_argument("--tail-file")
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--compatibility-cache-mb", type=float, default=256.0)
    parser.add_argument("--disk-cache-gb", type=float, default=1.0)
    parser.add_argument("--min-prefix-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--heartbeat-sec", type=float, default=15.0)
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--cache-root")
    parser.add_argument("--show-engine-stderr", action="store_true")
    parser.add_argument(
        "--skip-cold",
        action="store_true",
        help="run only SSD seed + restart hit; faster, but skips output comparison",
    )
    parser.add_argument("--cleanup-cache", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--output", default="prefix-persistent-benchmark.json")
    args = parser.parse_args()

    if args.memory_gb <= 0 or args.compatibility_cache_mb <= 0 or args.disk_cache_gb <= 0:
        parser.error("memory/cache/disk budgets must be positive")
    if args.min_prefix_tokens < 1 or args.context < 2 or args.max_tokens < 2:
        parser.error("min-prefix must be positive, context >=2, and max-tokens >=2")
    if args.prefix_tokens < 1 or args.chars_per_token <= 0:
        parser.error("prefix-tokens and chars-per-token must be positive")

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
        prefix = base.make_prompt(args.prefix_tokens, args.chars_per_token)
    if args.tail_file:
        tail = Path(args.tail_file).expanduser().read_bytes()
    else:
        tail = base.TAIL_TEXT.encode("utf-8")
    if not prefix or not tail:
        parser.error("prefix and tail must be non-empty")
    extended = prefix + tail

    cache_root = (
        Path(args.cache_root).expanduser().resolve()
        if args.cache_root
        else default_cache_root()
    )
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    cache_dir = cache_root / run_id
    cache_dir.mkdir(parents=True, exist_ok=False)
    progress(f"fresh SSD directory: {cache_dir}")
    started = time.perf_counter()

    common = dict(
        compute=args.compute,
        binary=binary,
        model=model,
        memory_gb=args.memory_gb,
        compatibility_cache_mb=args.compatibility_cache_mb,
        disk_gb=args.disk_cache_gb,
        cache_dir=cache_dir,
        min_prefix_tokens=args.min_prefix_tokens,
        context=args.context,
        max_tokens=args.max_tokens,
        ready_timeout_sec=args.ready_timeout,
        heartbeat_sec=args.heartbeat_sec,
        show_engine_stderr=args.show_engine_stderr,
    )

    try:
        cold = None
        if not args.skip_cold:
            progress("1/3 cache-off P+X correctness baseline")
            cold = run_one(
                mode="off", label="persist:cold", request_id="cold",
                prompt=extended, **common,
            )
            if int(cold["metric"].get("matched", -1)) != 0:
                raise RuntimeError("cache-off baseline unexpectedly reused a prefix")

        progress("2/3 SSD-only seed P")
        seed = run_one(
            mode="ssd", label="persist:seed", request_id="seed",
            prompt=prefix, **common,
        )
        if int(seed["metric"].get("matched", -1)) != 0:
            raise RuntimeError("fresh SSD seed unexpectedly reused a prefix")
        seed_tokens = int(seed["metric"]["prompt"])
        object_count, object_bytes = cache_files(cache_dir)
        if object_count < 1 or object_bytes < 1:
            raise RuntimeError("SSD seed finished but published no cpfx-*.bin object")
        progress(
            f"seed published {object_count} object(s), "
            f"{object_bytes / (1024 * 1024):.1f} MiB; restarting engine"
        )

        progress("3/3 restarted SSD-only P+X restore")
        hit = run_one(
            mode="ssd", label="persist:hit", request_id="hit",
            prompt=extended, **common,
        )
        matched = int(hit["metric"].get("matched", -1))
        if matched != seed_tokens:
            raise RuntimeError(
                f"persistent restore matched {matched}; expected seeded {seed_tokens} tokens"
            )

        if cold is not None and cold["request"]["output_hex"] != hit["request"]["output_hex"]:
            raise RuntimeError("persistent-hit output differs from cache-off output")

        hit_ttft = float(hit["request"]["first_response_sec"])
        restore_ms = float(hit["metric"]["restore_ms"])
        result = {
            "schema": "colibri.qwen.prefix_cache_persistent.v1",
            "git_sha": base.git_sha(repo),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "binary": str(binary),
            "model": str(model),
            "compute": args.compute,
            "memory_gb": args.memory_gb,
            "prefix_bytes": len(prefix),
            "tail_bytes": len(tail),
            "cache": {
                "directory": str(cache_dir),
                "disk_gb": args.disk_cache_gb,
                "objects_after_seed": object_count,
                "bytes_after_seed": object_bytes,
            },
            "environment": {
                key: os.environ[key]
                for key in ("COLI_CONFIG", "QWEN_KV_F16")
                if key in os.environ
            },
            "cold": cold,
            "seed": seed,
            "hit": hit,
            "summary": {
                "matched_tokens": matched,
                "hit_ttft_sec": hit_ttft,
                "restore_ms": restore_ms,
                "restore_bytes": int(hit["metric"].get("restore_bytes", 0)),
                "tail_prefill_ms": float(hit["metric"]["prefill_ms"]),
                "cold_ttft_sec": (
                    float(cold["request"]["first_response_sec"])
                    if cold is not None else None
                ),
            },
            "wall_sec": time.perf_counter() - started,
        }

        output = Path(args.output).expanduser()
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("\nPersistent prefix-cache result")
        print("=" * 72)
        print(f"compute:              {args.compute}")
        print(f"matched after restart: {matched} tokens")
        print(f"SSD restore:           {restore_ms:.3f} ms")
        print(f"restore bytes:         {result['summary']['restore_bytes']}")
        print(f"hit TTFT:              {hit_ttft:.3f} s")
        if cold is not None:
            cold_ttft = float(cold["request"]["first_response_sec"])
            speedup = cold_ttft / hit_ttft if hit_ttft > 0 else None
            print(f"cold TTFT:             {cold_ttft:.3f} s")
            if speedup is not None:
                print(f"speedup:               {speedup:.2f}x")
            print("output equality:       PASS")
        print(f"JSON:                  {output}")
        print(f"SSD cache:             {cache_dir}")
        return 0
    finally:
        if args.cleanup_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
