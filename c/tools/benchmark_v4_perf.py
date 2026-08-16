#!/usr/bin/env python3
"""Canonical DeepSeek-V4 end-to-end benchmark harness.

This intentionally uses only the Python standard library. It parses the stable
diagnostics already emitted by the V4 CLI and emits one machine-readable record
per run. Deeper phase timers are added in the engine separately; keeping the
runner independent makes it useful for comparing old commits too.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Optional


PROMPT_RE = re.compile(r"prompt_tokens=(\d+)\s+max_new_tokens=(\d+)")
TIMING_RE = re.compile(
    r"timing time_to_first_token=([0-9.]+)s after_first=([0-9.]+)s"
)
TUNE_RE = re.compile(r"TUNE decode:\s+(\d+)\s+tokens in\s+([0-9.]+)s")
TOKENS_RE = re.compile(
    r"v4_tokens prompt=(\d+) generated=(\d+) total=(\d+) "
    r"expert_requests=(\d+) hits=(\d+) misses=(\d+) "
    r"hit_rate=([0-9.]+) bytes=(\d+) target_only=(\d+)"
)

# Repeated to make long prompts deterministic and compressibility-neutral enough
# for an end-to-end engine benchmark. The runner records the ACTUAL tokenizer
# count emitted by the engine; target token counts below are sizing hints.
BENCH_TEXT = (
    "Colibri streams mixture-of-experts weights from storage while the active "
    "working set remains bounded by a memory budget. The benchmark measures "
    "prompt processing, generated-token latency, expert cache traffic, and "
    "end-to-end wall time using deterministic prose. "
)


@dataclass(frozen=True)
class Case:
    name: str
    target_prompt_tokens: int
    max_new_tokens: int


DEFAULT_CASES = {
    "decode1": Case("decode1", 64, 1),
    "decode32": Case("decode32", 64, 32),
    "decode128": Case("decode128", 64, 128),
    "prompt512": Case("prompt512", 512, 32),
    "prompt2k": Case("prompt2k", 2048, 32),
    "prompt8k": Case("prompt8k", 8192, 32),
}


@dataclass
class Result:
    case: str
    trial: int
    memory_gb: Optional[float]
    target_prompt_tokens: int
    prompt_bytes: int
    requested_new_tokens: int
    exit_code: int
    wall_sec: float
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    ttft_sec: Optional[float] = None
    after_first_sec: Optional[float] = None
    after_first_tok_s: Optional[float] = None
    tune_generated_tokens: Optional[int] = None
    tune_total_sec: Optional[float] = None
    tune_tok_s: Optional[float] = None
    expert_requests: Optional[int] = None
    expert_hits: Optional[int] = None
    expert_misses: Optional[int] = None
    expert_hit_rate_pct: Optional[float] = None
    expert_bytes: Optional[int] = None
    expert_bytes_per_total_token: Optional[float] = None
    target_only: Optional[int] = None
    git_sha: Optional[str] = None
    engine: Optional[str] = None
    model: Optional[str] = None
    platform: Optional[str] = None
    environment: Optional[dict] = None
    error: Optional[str] = None


def make_prompt(target_tokens: int, chars_per_token: float) -> str:
    """Create stable text near the requested tokenizer size.

    Tokenizers differ, so this is not claimed to be exact. The authoritative
    prompt_tokens value comes back from the engine and is stored in the result.
    """
    target_chars = max(1, int(target_tokens * chars_per_token))
    copies = max(1, (target_chars + len(BENCH_TEXT) - 1) // len(BENCH_TEXT))
    text = (BENCH_TEXT * copies)[:target_chars]
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def parse_output(stderr: str, stdout: str, result: Result) -> None:
    match = PROMPT_RE.search(stderr)
    if match:
        result.prompt_tokens = int(match.group(1))

    match = TIMING_RE.search(stderr)
    if match:
        result.ttft_sec = float(match.group(1))
        result.after_first_sec = float(match.group(2))

    match = TUNE_RE.search(stdout)
    if match:
        result.tune_generated_tokens = int(match.group(1))
        result.tune_total_sec = float(match.group(2))
        if result.tune_total_sec > 0:
            result.tune_tok_s = result.tune_generated_tokens / result.tune_total_sec

    match = TOKENS_RE.search(stderr)
    if match:
        result.prompt_tokens = int(match.group(1))
        result.generated_tokens = int(match.group(2))
        result.total_tokens = int(match.group(3))
        result.expert_requests = int(match.group(4))
        result.expert_hits = int(match.group(5))
        result.expert_misses = int(match.group(6))
        result.expert_hit_rate_pct = float(match.group(7))
        result.expert_bytes = int(match.group(8))
        result.target_only = int(match.group(9))

    if result.generated_tokens is None and result.tune_generated_tokens is not None:
        result.generated_tokens = result.tune_generated_tokens

    if (
        result.generated_tokens is not None
        and result.generated_tokens > 1
        and result.after_first_sec is not None
        and result.after_first_sec > 0
    ):
        result.after_first_tok_s = (result.generated_tokens - 1) / result.after_first_sec

    if (
        result.expert_bytes is not None
        and result.total_tokens is not None
        and result.total_tokens > 0
    ):
        result.expert_bytes_per_total_token = result.expert_bytes / result.total_tokens


def git_sha(repo: Optional[str]) -> Optional[str]:
    if not repo:
        return None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return proc.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def relevant_environment(env: dict[str, str]) -> dict[str, str]:
    prefixes = ("COLI_", "V4_", "OMP_")
    exact = {"METAL", "DRAFT", "SERVE", "SERVE_BATCH", "KV_SLOTS"}
    return {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(prefixes) or key in exact
    }


def case_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in DEFAULT_CASES]
    if unknown:
        raise argparse.ArgumentTypeError("unknown case(s): " + ", ".join(unknown))
    return names


def run_case(
    *,
    engine: str,
    model: str,
    case: Case,
    trial: int,
    memory_gb: Optional[float],
    raw_prompt: bool,
    chars_per_token: float,
    env: dict[str, str],
    sha: Optional[str],
    keep_logs: Optional[Path],
) -> Result:
    prompt = make_prompt(case.target_prompt_tokens, chars_per_token)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    ) as handle:
        handle.write(prompt)
        prompt_path = handle.name

    command = [
        engine,
        model,
        "--prompt-file",
        prompt_path,
        "--max-tokens",
        str(case.max_new_tokens),
        "--no-dspark",
    ]
    if memory_gb is not None:
        command.extend(["--memory-gb", f"{memory_gb:g}"])
    if raw_prompt:
        command.append("--raw-prompt")

    result = Result(
        case=case.name,
        trial=trial,
        memory_gb=memory_gb,
        target_prompt_tokens=case.target_prompt_tokens,
        prompt_bytes=len(prompt.encode("utf-8")),
        requested_new_tokens=case.max_new_tokens,
        exit_code=-1,
        wall_sec=0.0,
        git_sha=sha,
        engine=str(Path(engine).resolve()),
        model=str(Path(model).resolve()),
        platform=platform.platform(),
        environment=relevant_environment(env),
    )

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
            check=False,
        )
        result.wall_sec = time.perf_counter() - started
        result.exit_code = proc.returncode
        parse_output(proc.stderr, proc.stdout, result)
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-6:]
            result.error = "\n".join(tail) or f"engine exited {proc.returncode}"
        if keep_logs is not None:
            keep_logs.mkdir(parents=True, exist_ok=True)
            stem = (
                f"{case.name}-mem"
                f"{memory_gb if memory_gb is not None else 'auto'}-trial{trial}"
            )
            (keep_logs / f"{stem}.stdout").write_text(proc.stdout, encoding="utf-8")
            (keep_logs / f"{stem}.stderr").write_text(proc.stderr, encoding="utf-8")
            (keep_logs / f"{stem}.cmd").write_text(
                shlex.join(command) + "\n", encoding="utf-8"
            )
    except OSError as exc:
        result.wall_sec = time.perf_counter() - started
        result.error = str(exc)
    finally:
        try:
            os.unlink(prompt_path)
        except OSError:
            pass
    return result


def emit_jsonl(results: Iterable[Result], stream) -> None:
    for result in results:
        print(json.dumps(asdict(result), sort_keys=True), file=stream, flush=True)


def emit_csv(results: list[Result], stream) -> None:
    names = [field.name for field in fields(Result)]
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    for result in results:
        row = asdict(result)
        row["environment"] = json.dumps(row["environment"], sort_keys=True)
        writer.writerow(row)


def selftest() -> None:
    stderr = (
        "prompt_tokens=512 max_new_tokens=32 eos_token=1\n"
        "v4_tokens prompt=512 generated=32 total=544 expert_requests=99 "
        "hits=80 misses=19 hit_rate=80.808 bytes=123456 target_only=1\n"
        "timing time_to_first_token=2.500s after_first=6.200s\n"
    )
    stdout = "hello\nTUNE decode: 32 tokens in 8.700s\n"
    result = Result(
        case="selftest",
        trial=0,
        memory_gb=10.0,
        target_prompt_tokens=512,
        prompt_bytes=2048,
        requested_new_tokens=32,
        exit_code=0,
        wall_sec=9.0,
    )
    parse_output(stderr, stdout, result)
    assert result.prompt_tokens == 512
    assert result.generated_tokens == 32
    assert result.expert_hits == 80
    assert result.expert_misses == 19
    assert result.expert_bytes == 123456
    assert abs(result.ttft_sec - 2.5) < 1e-9
    assert abs(result.after_first_tok_s - 5.0) < 1e-9
    assert abs(result.tune_tok_s - (32 / 8.7)) < 1e-9
    assert make_prompt(512, 4.0) == make_prompt(512, 4.0)
    print("benchmark_v4_perf selftest: ok")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical DeepSeek-V4 performance matrix."
    )
    parser.add_argument("--engine", default="./deepseek_v4")
    parser.add_argument("--model", required=False)
    parser.add_argument(
        "--cases",
        type=case_names,
        default=list(DEFAULT_CASES),
        help="comma-separated: " + ",".join(DEFAULT_CASES),
    )
    parser.add_argument(
        "--memory-gb",
        type=float,
        action="append",
        dest="memory_gb",
        help="repeat to benchmark multiple budgets; omitted means engine auto",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="prompt sizing hint only; actual prompt_tokens is parsed from engine",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="bypass chat template; default exercises normal chat mode",
    )
    parser.add_argument("--repo", default="..", help="repo path used only for git SHA")
    parser.add_argument("--keep-logs", type=Path)
    parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    parser.add_argument("--output", default="-")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0
    if not args.model:
        parser.error("--model is required unless --selftest is used")
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    if args.chars_per_token <= 0:
        parser.error("--chars-per-token must be > 0")
    if args.memory_gb and any(value <= 0 for value in args.memory_gb):
        parser.error("--memory-gb values must be > 0")

    budgets: list[Optional[float]] = args.memory_gb or [None]
    env = os.environ.copy()
    sha = git_sha(args.repo)

    results: list[Result] = []
    for memory_gb in budgets:
        for case_name in args.cases:
            case = DEFAULT_CASES[case_name]
            for trial in range(1, args.trials + 1):
                print(
                    f"[v4-bench] case={case.name} trial={trial} "
                    f"memory_gb={memory_gb if memory_gb is not None else 'auto'}",
                    file=sys.stderr,
                    flush=True,
                )
                result = run_case(
                    engine=args.engine,
                    model=args.model,
                    case=case,
                    trial=trial,
                    memory_gb=memory_gb,
                    raw_prompt=args.raw_prompt,
                    chars_per_token=args.chars_per_token,
                    env=env,
                    sha=sha,
                    keep_logs=args.keep_logs,
                )
                results.append(result)
                if result.exit_code != 0:
                    print(
                        f"[v4-bench] FAILED case={case.name}: {result.error}",
                        file=sys.stderr,
                        flush=True,
                    )

    if args.output == "-":
        stream = sys.stdout
        close = False
    else:
        stream = open(args.output, "w", encoding="utf-8", newline="")
        close = True
    try:
        if args.format == "jsonl":
            emit_jsonl(results, stream)
        else:
            emit_csv(results, stream)
    finally:
        if close:
            stream.close()

    return 0 if all(result.exit_code == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
