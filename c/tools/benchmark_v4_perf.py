#!/usr/bin/env python3
"""DeepSeek-V4 end-to-end benchmark harness.

The default `quick` profile is intentionally suitable for development loops: it
runs one representative 8-token decode and caps that subprocess at 120 seconds.
Long-context and sustained-decode measurements remain available explicitly via
`--profile standard` / `--profile full`.

Only the Python standard library is used. The runner parses stable diagnostics
already emitted by the V4 CLI and emits one machine-readable record per run.
Deeper phase timers are added in the engine separately; keeping this runner
independent makes it useful for comparing older commits too.
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
# v4_phases scope=startup tokens=0 wall_ms=.. accounted_ms=.. unaccounted_ms=..
# cpu_compute_ms=.. io_wait_ms=.. dense_read_ms=.. ... emitted by the engine
# when V4_PROFILE=1.
PHASES_RE = re.compile(r"v4_phases scope=(\w+) tokens=(\d+) wall_ms=([0-9.eE+-]+) "
                       r"accounted_ms=([0-9.eE+-]+) unaccounted_ms=([0-9.eE+-]+) "
                       r"cpu_compute_ms=([0-9.eE+-]+) io_wait_ms=([0-9.eE+-]+) "
                       r"(.*)")
PHASE_FIELD_RE = re.compile(r"(\w+)_ms=([0-9.eE+-]+)|(\w+)=(\d+)")

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


# Split prefill from sustained decode instead of always doing +32 generation
# after long prompts. That makes targeted runs far cheaper and gives cleaner
# attribution: prefill cases primarily measure TTFT, decode cases measure token
# latency after a short prompt.
CASES = {
    "decode1": Case("decode1", 32, 1),
    "decode8": Case("decode8", 32, 8),
    "decode32": Case("decode32", 32, 32),
    "decode128": Case("decode128", 32, 128),
    "prefill512": Case("prefill512", 512, 1),
    "prefill2k": Case("prefill2k", 2048, 1),
    "prefill8k": Case("prefill8k", 8192, 1),
}

PROFILES = {
    # Normal edit -> build -> benchmark loop. One subprocess only. At the
    # measured ~7 s/token, decode8 is roughly a one-minute decode.
    "quick": ("decode8",),
    # Adds representative prefill cases without long generation tails.
    "standard": ("decode8", "prefill512", "prefill2k"),
    # Explicit regression/sustained suite. This is allowed to take a long time.
    "full": (
        "decode1",
        "decode32",
        "decode128",
        "prefill512",
        "prefill2k",
        "prefill8k",
    ),
}

PROFILE_TIMEOUT_SEC = {
    "quick": 120.0,
    "standard": 300.0,
    "full": 0.0,
}


@dataclass
class Result:
    profile: str
    case: str
    trial: int
    warm: bool
    memory_gb: Optional[float]
    target_prompt_tokens: int
    prompt_bytes: int
    requested_new_tokens: int
    timeout_sec: float
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
    # v4_phases rows (V4_PROFILE=1): scope name -> dict of phase ms/bytes.
    phases: Optional[dict] = None
    # Reconciliation flags computed after parsing.
    phases_reconcile: Optional[bool] = None
    phases_unaccounted_ratio: Optional[float] = None
    git_sha: Optional[str] = None
    engine: Optional[str] = None
    model: Optional[str] = None
    platform: Optional[str] = None
    environment: Optional[dict] = None
    error: Optional[str] = None


def make_prompt(target_tokens: int, chars_per_token: float) -> str:
    """Create stable text near the requested tokenizer size.

    Tokenizers differ, so this is only a sizing hint. The authoritative
    `prompt_tokens` value emitted by the engine is stored in every result.
    """
    target_chars = max(1, int(target_tokens * chars_per_token))
    copies = max(1, (target_chars + len(BENCH_TEXT) - 1) // len(BENCH_TEXT))
    text = (BENCH_TEXT * copies)[:target_chars]
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def parse_phases(stderr: str) -> Optional[dict]:
    """Parse engine v4_phases lines (V4_PROFILE=1) into {scope: {field: val}}.

    Float fields keep ms units; integer fields (bytes, tokens) parsed as ints.
    The runner additionally derives the issue's per-token bytes later.
    """
    scopes = {}
    for line in stderr.splitlines():
        match = PHASES_RE.search(line)
        if not match:
            continue
        scope = match.group(1)
        row = {
            "tokens": int(match.group(2)),
            "wall_ms": float(match.group(3)),
            "accounted_ms": float(match.group(4)),
            "unaccounted_ms": float(match.group(5)),
            "cpu_compute_ms": float(match.group(6)),
            "io_wait_ms": float(match.group(7)),
        }
        for field_match in PHASE_FIELD_RE.finditer(match.group(8)):
            if field_match.group(1):  # *_ms=
                row[field_match.group(1) + "_ms"] = float(field_match.group(2))
            else:  # bytes/counters
                row[field_match.group(3)] = int(field_match.group(4))
        scopes[scope] = row
    return scopes or None


def reconcile_phases(result: Result) -> None:
    """Check the issue's reconciliation gate on the run scope:

    |unaccounted_ms| <= max(5 ms, 2% of run wall) -- the 2% ceiling the
    PROFILE1 design set for the quick benchmark.  Marks were taken around
    session_generate, so run.wall_ms should be close to TUNE decode seconds.
    unaccounted_ms is SIGNED (wall - accounted) so over-counting shows up as a
    negative number and still fails the gate. If V4_PROFILE was off or the
    engine is older, leave the flags None.
    """
    if not result.phases or "run" not in result.phases:
        return
    run = result.phases["run"]
    wall = run.get("wall_ms", 0.0)
    unaccounted = run.get("unaccounted_ms", 0.0)
    budget = max(5.0, 0.02 * wall)
    result.phases_reconcile = abs(unaccounted) <= budget
    result.phases_unaccounted_ratio = (
        abs(unaccounted) / wall if wall > 0 else None
    )


def parse_output(stderr: str, stdout: str, result: Result) -> None:    # noqa: C901 (branchy by design; fields mirror the engine's lines)
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

    result.phases = parse_phases(stderr)

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


def parse_case_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise argparse.ArgumentTypeError("unknown case(s): " + ", ".join(unknown))
    if not names:
        raise argparse.ArgumentTypeError("at least one case is required")
    return names


def text_from_timeout(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def run_case(
    *,
    engine: str,
    model: str,
    profile: str,
    case: Case,
    trial: int,
    warm: bool,
    memory_gb: Optional[float],
    raw_prompt: bool,
    chars_per_token: float,
    timeout_sec: float,
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
        profile=profile,
        case=case.name,
        trial=trial,
        warm=warm,
        memory_gb=memory_gb,
        target_prompt_tokens=case.target_prompt_tokens,
        prompt_bytes=len(prompt.encode("utf-8")),
        requested_new_tokens=case.max_new_tokens,
        timeout_sec=timeout_sec,
        exit_code=-1,
        wall_sec=0.0,
        git_sha=sha,
        engine=str(Path(engine).resolve()),
        model=str(Path(model).resolve()),
        platform=platform.platform(),
        environment=relevant_environment(env),
    )

    stdout = ""
    stderr = ""
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
            timeout=timeout_sec if timeout_sec > 0 else None,
        )
        result.wall_sec = time.perf_counter() - started
        result.exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
        parse_output(stderr, stdout, result)
        reconcile_phases(result)
        if proc.returncode != 0:
            tail = stderr.strip().splitlines()[-6:]
            result.error = "\n".join(tail) or f"engine exited {proc.returncode}"
    except subprocess.TimeoutExpired as exc:
        result.wall_sec = time.perf_counter() - started
        result.exit_code = 124
        stdout = text_from_timeout(exc.stdout)
        stderr = text_from_timeout(exc.stderr)
        parse_output(stderr, stdout, result)
        result.error = f"benchmark case exceeded {timeout_sec:g}s timeout"
    except OSError as exc:
        result.wall_sec = time.perf_counter() - started
        result.error = str(exc)
    finally:
        if keep_logs is not None:
            keep_logs.mkdir(parents=True, exist_ok=True)
            stem = (
                f"{profile}-{case.name}-mem"
                f"{memory_gb if memory_gb is not None else 'auto'}-trial{trial}"
            )
            (keep_logs / f"{stem}.stdout").write_text(stdout, encoding="utf-8")
            (keep_logs / f"{stem}.stderr").write_text(stderr, encoding="utf-8")
            (keep_logs / f"{stem}.cmd").write_text(
                shlex.join(command) + "\n", encoding="utf-8"
            )
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
        "prompt_tokens=512 max_new_tokens=8 eos_token=1\n"
        "v4_tokens prompt=512 generated=8 total=520 expert_requests=99 "
        "hits=80 misses=19 hit_rate=80.808 bytes=123456 target_only=1\n"
        "timing time_to_first_token=2.500s after_first=1.400s\n"
        "v4_phases scope=startup tokens=0 wall_ms=123.000 accounted_ms=100.000 "
        "unaccounted_ms=23.000 cpu_compute_ms=90.000 io_wait_ms=10.000 "
        "dense_read_ms=20.000 dense_read_bytes=6710886400 "
        "head_read_bytes=1073741824\n"
        "v4_phases scope=prompt tokens=512 wall_ms=500.000 accounted_ms=490.000 "
        "unaccounted_ms=10.000 cpu_compute_ms=400.000 io_wait_ms=90.000 "
        "dense_read_ms=0.000 attn_proj_ms=60.000\n"
        "v4_phases scope=run tokens=8 wall_ms=1000.000 accounted_ms=990.000 "
        "unaccounted_ms=10.000 cpu_compute_ms=800.000 io_wait_ms=190.000 "
        "dense_read_ms=0.000 expert_wait_ms=150.000 expert_compute_ms=400.000 "
        "hc_pre_ms=50.000 hc_post_ms=40.000 attn_proj_ms=120.000 "
        "expert_read_work_ms=300.000 expert_read_bytes=7340032000\n"
    )
    stdout = "hello\nTUNE decode: 8 tokens in 3.900s\n"
    result = Result(
        profile="quick",
        case="selftest",
        trial=0,
        warm=False,
        memory_gb=10.0,
        target_prompt_tokens=512,
        prompt_bytes=2048,
        requested_new_tokens=8,
        timeout_sec=120.0,
        exit_code=0,
        wall_sec=4.0,
    )
    parse_output(stderr, stdout, result)
    reconcile_phases(result)
    assert result.prompt_tokens == 512
    assert result.generated_tokens == 8
    assert result.expert_hits == 80
    assert result.expert_misses == 19
    assert result.expert_bytes == 123456
    assert abs(result.ttft_sec - 2.5) < 1e-9
    assert abs(result.after_first_tok_s - 5.0) < 1e-9
    assert abs(result.tune_tok_s - (8 / 3.9)) < 1e-9
    assert make_prompt(512, 4.0) == make_prompt(512, 4.0)
    assert PROFILES["quick"] == ("decode8",)
    assert PROFILE_TIMEOUT_SEC["quick"] == 120.0
    assert text_from_timeout(b"partial") == "partial"
    assert result.phases is not None
    assert "run" in result.phases
    run = result.phases["run"]
    assert run["tokens"] == 8
    assert abs(run["wall_ms"] - 1000.0) < 1e-9
    assert abs(run["attn_proj_ms"] - 120.0) < 1e-9
    assert run["expert_read_bytes"] == 7340032000
    assert result.phases_reconcile is True  # 10ms unaccounted on 1000ms wall
    assert result.phases_unaccounted_ratio == 0.01
    print("benchmark_v4_perf selftest: ok")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-V4 performance benchmarks. Default: one <=120s quick run."
    )
    parser.add_argument("--engine", default="./deepseek_v4")
    parser.add_argument("--model", required=False)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="quick",
        help="quick is the default edit/build loop; standard/full are explicit",
    )
    parser.add_argument(
        "--cases",
        type=parse_case_names,
        default=None,
        help="override profile cases; available: " + ",".join(CASES),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=None,
        help="per-case timeout; profile default if omitted, 0 disables timeout",
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
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="do not set V4_PROFILE=1; disables engine phase telemetry",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        help="run each case twice and label the second trial warm (page-cache "
        "warm); requires --trials 2 for a full cold/warm pair, else adds one",
    )
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
    if args.timeout_sec is not None and args.timeout_sec < 0:
        parser.error("--timeout-sec must be >= 0")

    case_names = args.cases or list(PROFILES[args.profile])
    timeout_sec = (
        args.timeout_sec
        if args.timeout_sec is not None
        else PROFILE_TIMEOUT_SEC[args.profile]
    )
    budgets: list[Optional[float]] = args.memory_gb or [None]
    env = os.environ.copy()
    if not args.no_profile:
        env["V4_PROFILE"] = "1"
    sha = git_sha(args.repo)

    cases_or_warm: list[tuple[int, bool]] = []
    if args.warm_cache:
        # Warm-cache pair: trial 1 cold (fresh page cache), trial 2 warm.
        if args.trials == 1:
            args.trials = 2
        for trial in range(1, args.trials + 1):
            cases_or_warm.append((trial, trial > 1))
    results: list[Result] = []
    for memory_gb in budgets:
        for case_name in case_names:
            case = CASES[case_name]
            if cases_or_warm:
                trials: Iterable[tuple[int, bool]] = cases_or_warm
            else:
                trials = ((trial, False)
                          for trial in range(1, args.trials + 1))
            for trial, is_warm in trials:
                print(
                    f"[v4-bench] profile={args.profile} case={case.name} "
                    f"trial={trial}{' warm' if is_warm else ''} memory_gb="
                    f"{memory_gb if memory_gb is not None else 'auto'} "
                    f"timeout_sec={timeout_sec:g}",
                    file=sys.stderr,
                    flush=True,
                )
                result = run_case(
                    engine=args.engine,
                    model=args.model,
                    profile=args.profile,
                    case=case,
                    trial=trial,
                    warm=is_warm,
                    memory_gb=memory_gb,
                    raw_prompt=args.raw_prompt,
                    chars_per_token=args.chars_per_token,
                    timeout_sec=timeout_sec,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
