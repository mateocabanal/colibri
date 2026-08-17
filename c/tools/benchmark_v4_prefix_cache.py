#!/usr/bin/env python3
"""Controlled cache-off/cache-on V4 process-local prefix-cache TTFT A/B."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


BENCH_TEXT = (
    "Colibri streams mixture-of-experts weights from storage while the active "
    "working set remains bounded by a memory budget. The prompt-prefix benchmark "
    "repeats stable system, tool, repository, and agent context so only the new "
    "request tail should require target prefill work. "
)

RELEVANT_ENV = (
    "V4_RAM_GB", "V4_COLI_CACHE_SLOTS", "V4_PIN_SLOTS_PER_LAYER",
    "V4_COLI_DENSE_CACHE", "V4_COLI_DENSE_CACHE_MIN_BENEFIT",
    "V4_COLI_DENSE_CACHE_MAX_GB", "V4_COLI_PERSISTENT_SLOTS_PER_LAYER",
    "V4_COLI_RESIDENCY_POLICY", "V4_EXPERT_ROWS16",
    "V4_EXPERT_HOT_HYSTERESIS", "V4_TUNE_CACHED_HEAD",
    "V4_COLI_HEAD_MODE", "V4_FULL_DSPARK", "V4_FORCE_DENSE_STREAM",
    "V4_KV_I8", "V4_KV_BF16", "V4_PROFILE", "V4_PROFILE_OVERHEAD",
    "V4_PINNED_AVAILABLE", "V4_DISABLE_AUTOPIN", "V4_PIN_MEMORY_FRACTION",
    "V4_PIN_MLOCK", "COLI_MODEL", "COLI_V4_NATIVE_QUANT",
    "COLI_V4_NATIVE_QUANT_CACHE", "OMP_NUM_THREADS",
    "OMP_PROC_BIND", "OMP_PLACES",
)


def make_prompt(target_tokens: int, chars_per_token: float) -> str:
    target_chars = max(1, int(target_tokens * chars_per_token))
    copies = max(1, (target_chars + len(BENCH_TEXT) - 1) // len(BENCH_TEXT))
    text = (BENCH_TEXT * copies)[:target_chars]
    if not text.endswith((".", "!", "?", "\n")):
        text += "."
    return text


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


def build_config(binary: Path) -> dict[str, str]:
    path = binary.parent / ".deepseek-v4-build-config"
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text("utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            result[key] = value
    return result


def relevant_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in RELEVANT_ENV if key in os.environ}


def parse_probe(stdout: str) -> dict[str, object] | None:
    record = None
    for line in stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("schema") == "colibri.v4.prefix_cache_probe.v1":
            record = candidate
    return record


def run_probe(
    *,
    binary: Path,
    model: Path,
    coli_model: Path | None,
    prefix_path: Path,
    memory_gb: float,
    context: int,
    max_new: int,
    cache_mb: float,
    min_prefix_tokens: int,
    timeout_sec: float,
    mode: str,
    trial: int,
) -> dict[str, object]:
    enabled = mode == "cache_on"
    env = os.environ.copy()
    env["V4_PREFIX_CACHE_MB"] = f"{cache_mb:g}" if enabled else "0"
    env["V4_PREFIX_CACHE_MIN_TOKENS"] = str(min_prefix_tokens)

    command = [
        str(binary), str(model), str(prefix_path),
        "--memory-gb", f"{memory_gb:g}",
        "--context", str(context),
        "--max-new", str(max_new),
    ]
    if coli_model is not None:
        command.extend(["--coli-model", str(coli_model)])

    began = time.perf_counter()
    try:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout_sec if timeout_sec > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        if exc.stderr:
            sys.stderr.write(str(exc.stderr))
        raise RuntimeError(
            f"{mode} trial {trial} timed out after {timeout_sec:g}s"
        ) from exc

    if process.stderr:
        sys.stderr.write(f"[{mode} trial={trial}]\n{process.stderr}")
    if process.returncode:
        raise RuntimeError(
            f"{mode} trial {trial} failed with exit {process.returncode}:\n"
            f"{process.stdout}"
        )
    record = parse_probe(process.stdout)
    if record is None:
        raise RuntimeError(
            f"{mode} trial {trial} produced no probe JSON:\n{process.stdout}"
        )
    if record.get("mode") != mode:
        raise RuntimeError(
            f"{mode} trial {trial} reported mode={record.get('mode')!r}"
        )
    record["process_wall_sec"] = time.perf_counter() - began
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./benchmark_v4_prefix_cache_sessions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--coli-model")
    parser.add_argument("--prefix-file")
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument("--cache-mb", type=float, default=256.0)
    parser.add_argument("--min-prefix-tokens", type=int, default=256)
    parser.add_argument("--memory-gb", type=float, required=True)
    parser.add_argument("--context", type=int)
    parser.add_argument("--max-new", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--repo")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.prefix_tokens < 1 or args.chars_per_token <= 0:
        parser.error("--prefix-tokens and --chars-per-token must be positive")
    if args.cache_mb <= 0 or args.min_prefix_tokens < 1:
        parser.error("--cache-mb and --min-prefix-tokens must be positive")
    if args.memory_gb <= 0:
        parser.error("--memory-gb must be positive")
    if args.context is not None and args.context < 2:
        parser.error("--context must be at least 2")
    if args.max_new < 1 or args.trials < 1 or args.timeout_sec < 0:
        parser.error("--max-new/--trials must be positive and timeout non-negative")

    binary = Path(args.binary).expanduser().resolve()
    model = Path(args.model).expanduser().resolve()
    coli_model = Path(args.coli_model).expanduser().resolve() if args.coli_model else None
    repo = Path(args.repo).expanduser().resolve() if args.repo else binary.parent.parent
    if not binary.is_file():
        parser.error(f"benchmark binary does not exist: {binary}")

    generated_prompt = args.prefix_file is None
    if generated_prompt:
        prompt = make_prompt(args.prefix_tokens, args.chars_per_token)
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        )
        handle.write(prompt)
        handle.close()
        prefix_path = Path(handle.name)
    else:
        prefix_path = Path(args.prefix_file).expanduser().resolve()
        if not prefix_path.is_file():
            parser.error(f"prefix file does not exist: {prefix_path}")

    context = args.context
    if context is None:
        # This is only a sizing hint; each C probe reports authoritative counts.
        context = max(4096, args.prefix_tokens * 2 if generated_prompt else 4096)

    started = time.perf_counter()
    pairs: list[dict[str, object]] = []
    try:
        for trial_index in range(args.trials):
            trial_number = trial_index + 1
            # Alternate which process runs first when there is more than one pair
            # so global OS/storage warming cannot systematically favor one mode.
            order = (
                ["cache_off", "cache_on"]
                if trial_index % 2 == 0 else ["cache_on", "cache_off"]
            )
            observed: dict[str, dict[str, object]] = {}
            for mode in order:
                observed[mode] = run_probe(
                    binary=binary,
                    model=model,
                    coli_model=coli_model,
                    prefix_path=prefix_path,
                    memory_gb=args.memory_gb,
                    context=context,
                    max_new=args.max_new,
                    cache_mb=args.cache_mb,
                    min_prefix_tokens=args.min_prefix_tokens,
                    timeout_sec=args.timeout_sec,
                    mode=mode,
                    trial=trial_number,
                )

            baseline = observed["cache_off"]
            treatment = observed["cache_on"]
            if baseline["prefix_tokens"] != treatment["prefix_tokens"] or \
               baseline["second_prompt_tokens"] != treatment["second_prompt_tokens"]:
                raise RuntimeError(
                    f"trial {trial_number} did not run identical token workloads: "
                    f"off={baseline['prefix_tokens']}/{baseline['second_prompt_tokens']} "
                    f"on={treatment['prefix_tokens']}/{treatment['second_prompt_tokens']}"
                )
            if int(baseline["second_prefix_reused"]) != 0:
                raise RuntimeError(f"trial {trial_number} cache-off unexpectedly reused prefix")
            if int(treatment["second_prefix_reused"]) != int(treatment["prefix_tokens"]):
                raise RuntimeError(f"trial {trial_number} cache-on did not restore full prefix")

            baseline_ttft = float(baseline["second_ttft_sec"])
            treatment_ttft = float(treatment["second_ttft_sec"])
            pairs.append({
                "trial": trial_number,
                "order": order,
                "cache_off": baseline,
                "cache_on": treatment,
                "ttft_saved_sec": baseline_ttft - treatment_ttft,
                "ttft_speedup": (
                    baseline_ttft / treatment_ttft if treatment_ttft > 0.0 else 0.0
                ),
            })
    except RuntimeError as error:
        print(f"prefix-cache A/B failed: {error}", file=sys.stderr)
        return 2
    finally:
        if generated_prompt:
            try:
                prefix_path.unlink()
            except OSError:
                pass

    baseline_ttfts = [float(pair["cache_off"]["second_ttft_sec"]) for pair in pairs]  # type: ignore[index]
    treatment_ttfts = [float(pair["cache_on"]["second_ttft_sec"]) for pair in pairs]  # type: ignore[index]
    paired_saved = [float(pair["ttft_saved_sec"]) for pair in pairs]
    paired_speedup = [float(pair["ttft_speedup"]) for pair in pairs]
    first_baseline = pairs[0]["cache_off"]  # type: ignore[assignment]

    result = {
        "schema": "colibri.v4.prefix_cache_ab.v1",
        "method": "paired second-request cache-off/cache-on; same total memory limit",
        "git_sha": git_sha(repo),
        "repo": str(repo),
        "platform": platform.platform(),
        "binary": str(binary),
        "model": str(model),
        "coli_model": str(coli_model) if coli_model else None,
        "target_prefix_tokens": args.prefix_tokens if generated_prompt else None,
        "prefix_tokens": int(first_baseline["prefix_tokens"]),
        "second_prompt_tokens": int(first_baseline["second_prompt_tokens"]),
        "prefix_file": None if generated_prompt else str(prefix_path),
        "memory_gb": args.memory_gb,
        "cache_mb": args.cache_mb,
        "min_prefix_tokens": args.min_prefix_tokens,
        "context": context,
        "max_new": args.max_new,
        "trials": args.trials,
        "baseline_second_ttft_median_sec": statistics.median(baseline_ttfts),
        "cache_second_ttft_median_sec": statistics.median(treatment_ttfts),
        "paired_ttft_saved_median_sec": statistics.median(paired_saved),
        "paired_ttft_speedup_median": statistics.median(paired_speedup),
        "pairs": pairs,
        "environment": relevant_environment(),
        "mode_environment": {
            "cache_off": {
                "V4_PREFIX_CACHE_MB": "0",
                "V4_PREFIX_CACHE_MIN_TOKENS": str(args.min_prefix_tokens),
            },
            "cache_on": {
                "V4_PREFIX_CACHE_MB": f"{args.cache_mb:g}",
                "V4_PREFIX_CACHE_MIN_TOKENS": str(args.min_prefix_tokens),
            },
        },
        "build_config": build_config(binary),
        "wall_sec": time.perf_counter() - started,
    }
    encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
