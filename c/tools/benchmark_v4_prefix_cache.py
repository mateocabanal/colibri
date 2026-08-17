#!/usr/bin/env python3
"""Run the real two-session V4 process-local prefix-cache TTFT benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
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


def make_prompt(target_tokens: int, chars_per_token: float) -> str:
    target_chars = max(1, int(target_tokens * chars_per_token))
    copies = max(1, (target_chars + len(BENCH_TEXT) - 1) // len(BENCH_TEXT))
    text = (BENCH_TEXT * copies)[:target_chars]
    if not text.endswith((".", "!", "?", "\n")):
        text += "."
    return text


def git_sha(repo: str | None) -> str | None:
    if not repo:
        return None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=True,
        )
        return proc.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


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
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--context", type=int)
    parser.add_argument("--max-new", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=900.0)
    parser.add_argument("--repo", default="..")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.prefix_tokens < 1 or args.chars_per_token <= 0:
        parser.error("--prefix-tokens and --chars-per-token must be positive")
    if args.cache_mb <= 0 or args.min_prefix_tokens < 1:
        parser.error("--cache-mb and --min-prefix-tokens must be positive")
    if args.memory_gb is not None and args.memory_gb <= 0:
        parser.error("--memory-gb must be positive")
    if args.context is not None and args.context < 2:
        parser.error("--context must be at least 2")
    if args.max_new < 1 or args.timeout_sec < 0:
        parser.error("--max-new must be positive and --timeout-sec non-negative")

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
        # Sizing hint only; the C probe reports authoritative tokenizer counts.
        context = max(4096, args.prefix_tokens * 2 if generated_prompt else 4096)

    env = os.environ.copy()
    env["V4_PREFIX_CACHE_MB"] = f"{args.cache_mb:g}"
    env["V4_PREFIX_CACHE_MIN_TOKENS"] = str(args.min_prefix_tokens)

    command = [
        args.binary,
        str(Path(args.model).expanduser()),
        str(prefix_path),
        "--context", str(context),
        "--max-new", str(args.max_new),
    ]
    if args.memory_gb is not None:
        command.extend(["--memory-gb", f"{args.memory_gb:g}"])
    if args.coli_model:
        command.extend(["--coli-model", str(Path(args.coli_model).expanduser())])

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=args.timeout_sec if args.timeout_sec > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        sys.stderr.write(exc.stderr or "")
        print(f"prefix-cache benchmark timed out after {args.timeout_sec:g}s", file=sys.stderr)
        return 124
    finally:
        if generated_prompt:
            try:
                prefix_path.unlink()
            except OSError:
                pass

    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        return proc.returncode

    record = None
    for line in proc.stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("schema") == "colibri.v4.prefix_cache_bench.v1":
            record = candidate
    if record is None:
        sys.stderr.write(proc.stdout)
        print("benchmark produced no prefix-cache JSON record", file=sys.stderr)
        return 2

    record.update({
        "git_sha": git_sha(args.repo),
        "platform": platform.platform(),
        "model": str(Path(args.model).expanduser().resolve()),
        "coli_model": (
            str(Path(args.coli_model).expanduser().resolve())
            if args.coli_model else None
        ),
        "target_prefix_tokens": args.prefix_tokens if generated_prompt else None,
        "prefix_file": None if generated_prompt else str(prefix_path),
        "cache_mb": args.cache_mb,
        "min_prefix_tokens": args.min_prefix_tokens,
        "context": context,
        "wall_sec": time.perf_counter() - started,
    })
    encoded = json.dumps(record, sort_keys=True)
    print(encoded)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
