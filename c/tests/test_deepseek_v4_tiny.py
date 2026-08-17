#!/usr/bin/env python3
"""Dependency-free token-exact checks for the generated tiny V4 target fixture."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import openai_server


def run(
    label: str,
    command: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
        env=env,
    )
    if result.returncode:
        raise AssertionError(
            f"{label} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def flat_oracle(case: dict[str, object]) -> dict[str, object]:
    return {
        "prompt_ids": case["prompt_ids"],
        "full_ids": case["greedy_full_ids"],
        "tf_pred": case["teacher_forcing_ids"],
    }


def check_target(
    binary: Path,
    model: Path,
    name: str,
    case: dict[str, object],
    temporary: Path,
) -> None:
    oracle = temporary / f"{name}.json"
    oracle.write_text(json.dumps(flat_oracle(case)), encoding="utf-8")
    full = case["greedy_full_ids"]
    generated = case["greedy_new_ids"]
    result = run(
        f"target {name}",
        [
            binary.as_posix(),
            model.as_posix(),
            "--oracle",
            oracle.as_posix(),
            "--teacher-forcing",
            str(len(full)),
            "--greedy",
            str(len(generated)),
        ],
    )
    if f"{len(full)}/{len(full)} positions" not in result.stdout:
        raise AssertionError(f"target {name}: missing exact teacher-forcing result")
    if f"{len(generated)}/{len(generated)} tokens" not in result.stdout:
        raise AssertionError(f"target {name}: missing exact greedy result")
    print(f"PASS target {name}: teacher forcing and greedy token-exact")


def check_prefix_is_rejected(
    binary: Path, model: Path, case: dict[str, object], temporary: Path
) -> None:
    truncated = flat_oracle(case)
    truncated["full_ids"] = truncated["full_ids"][:-1]
    truncated["tf_pred"] = truncated["tf_pred"][:-1]
    oracle = temporary / "truncated-prefix.json"
    oracle.write_text(json.dumps(truncated), encoding="utf-8")
    result = subprocess.run(
        [
            binary.as_posix(),
            model.as_posix(),
            "--oracle",
            oracle.as_posix(),
            "--teacher-forcing",
            str(len(truncated["full_ids"])),
            "--greedy",
            str(len(case["greedy_new_ids"])),
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    if result.returncode == 0 or "greedy length mismatch" not in result.stderr:
        raise AssertionError("truncated greedy prefix was not rejected")
    print("PASS target greedy: truncated prefix rejected")


def token_prompt(ids: list[int]) -> str:
    return "".join(f"<t{token:03d}>" for token in ids)


def check_session(
    binary: Path,
    model: Path,
    name: str,
    case: dict[str, object],
    temporary: Path,
    ordinal: int = 0,
    compatibility_flag: bool = False,
) -> list[int]:
    record = temporary / f"{name}-target-{ordinal}.json"
    command = [
        binary.as_posix(),
        model.as_posix(),
        token_prompt(case["prompt_ids"]),
        "--raw-prompt",
        "--max-tokens",
        str(case["max_new_tokens"]),
        "--record-oracle",
        record.as_posix(),
    ]
    if compatibility_flag:
        command.append("--no-dspark")
    result = run(f"target session {name}", command)
    actual = json.loads(record.read_text(encoding="utf-8"))
    expected_prompt = case["prompt_ids"]
    expected_full = case["greedy_full_ids"]
    if actual.get("prompt_ids") != expected_prompt:
        raise AssertionError(
            f"session {name}: tokenized prompt mismatch: "
            f"expected {expected_prompt}, got {actual.get('prompt_ids')}"
        )
    if actual.get("full_ids") != expected_full:
        raise AssertionError(
            f"session {name}: exact output mismatch: "
            f"expected {expected_full}, got {actual.get('full_ids')}"
        )
    stats = re.search(
        r"v4_tokens prompt=(\d+) generated=(\d+).*?target_only=1",
        result.stderr,
    )
    if not stats:
        raise AssertionError(f"session {name}: missing target-only statistics")
    prompt_count, generated_count = (int(value) for value in stats.groups())
    if prompt_count != len(expected_prompt) or generated_count != case["max_new_tokens"]:
        raise AssertionError(
            f"session {name}: length mismatch: "
            f"prompt={prompt_count} generated={generated_count}"
        )
    if compatibility_flag:
        attempts = re.search(r"v4_dspark attempts=(\d+)", result.stderr)
        if attempts and int(attempts.group(1)) != 0:
            raise AssertionError(
                f"--no-dspark still attempted {attempts.group(1)} speculations")
    print(f"PASS target session {name}: exact IDs and exact length")
    return actual["full_ids"]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path.name}:{number}: malformed JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AssertionError(f"{path.name}:{number}: expected JSON object")
        rows.append(row)
    if not rows:
        raise AssertionError(f"{path.name}: empty trace")
    return rows


def check_route_trace(
    binary: Path, model: Path, case: dict[str, object], temporary: Path
) -> None:
    """Gate #56 when this binary was compiled with detailed route tracing."""
    routes = temporary / "routes.jsonl"
    max_new = min(2, int(case["max_new_tokens"]))
    env = dict(
        os.environ,
        V4_ROUTE_TRACE=routes.as_posix(),
        V4_ROUTE_TRACE_CAP="65536",
        V4_ROUTE_REQUEST_ID="77",
    )
    result = run(
        "target explicit route trace",
        [
            binary.as_posix(),
            model.as_posix(),
            token_prompt(case["prompt_ids"]),
            "--raw-prompt",
            "--max-tokens",
            str(max_new),
            "--no-dspark",
        ],
        env=env,
    )
    if "v4_route_trace status=written" not in result.stderr:
        # Detailed route instrumentation is deliberately compile-gated. The
        # ordinary all-engines/default-hot-path CI build must remain valid with
        # V4_TRACE_ROUTE=0; the dedicated V4_TRACE_EXEC=1 jobs below/alongside
        # this oracle exercise the strict route lifecycle assertions.
        print("SKIP target trace: detailed route tracing not compiled into this binary")
        return

    route_rows = read_jsonl(routes)
    header = route_rows[0]
    if header.get("schema") != "colibri.v4.expert_trace.v2":
        raise AssertionError(f"unexpected logical trace header: {header}")
    if header.get("source") != "route_selected" or header.get("dropped") != 0:
        raise AssertionError(f"bad logical trace metadata: {header}")
    if header.get("requests_started") != 1:
        raise AssertionError(f"expected one traced request: {header}")
    if not isinstance(header.get("physical_lookups"), int) or header["physical_lookups"] <= 0:
        raise AssertionError(f"trace did not observe physical expert lookups: {header}")
    if header.get("correlation_misses") != 0 or header.get("uncorrelated_routes") != 0:
        raise AssertionError(f"route/lookup correlation was incomplete: {header}")

    selections = [row for row in route_rows[1:] if row.get("event") == "request"]
    if not selections:
        raise AssertionError("logical route trace contains no selections")
    prompt_count = len(case["prompt_ids"])
    positions: set[int] = set()
    route_groups: dict[tuple[int, int], list[int]] = {}
    lookup_groups: dict[int, list[dict[str, object]]] = {}
    for row in selections:
        if row.get("request_id") != 77:
            raise AssertionError(f"route request id was not preserved: {row}")
        position = row.get("token_position")
        layer = row.get("layer")
        expert = row.get("expert")
        rank = row.get("route_rank")
        weight = row.get("route_weight")
        lookup_id = row.get("lookup_id")
        lookup_ns = row.get("lookup_ns")
        lookup_routes = row.get("lookup_routes")
        lookup_result = row.get("lookup_result")
        generation = row.get("lease_generation")
        if not isinstance(position, int) or position < 0:
            raise AssertionError(f"route is missing token position: {row}")
        if not isinstance(layer, int) or layer < 0:
            raise AssertionError(f"route is missing layer identity: {row}")
        if not isinstance(expert, int) or expert < 0:
            raise AssertionError(f"route is missing expert identity: {row}")
        if not isinstance(rank, int) or rank < 0:
            raise AssertionError(f"route is missing rank identity: {row}")
        if not isinstance(weight, (int, float)):
            raise AssertionError(f"route is missing numeric weight: {row}")
        expected_phase = "prefill" if position < prompt_count else "decode"
        if row.get("phase") != expected_phase:
            raise AssertionError(
                f"route phase mismatch at position {position}: "
                f"expected {expected_phase}, got {row.get('phase')}: {row}"
            )
        if not isinstance(lookup_id, int) or lookup_id <= 0:
            raise AssertionError(f"route is missing physical lookup id: {row}")
        if not isinstance(lookup_ns, int) or lookup_ns < 0:
            raise AssertionError(f"route is missing lookup duration: {row}")
        if not isinstance(lookup_routes, int) or lookup_routes < 1:
            raise AssertionError(f"route is missing lookup fanout: {row}")
        if lookup_result != 0:
            raise AssertionError(f"route physical lookup failed: {row}")
        if not isinstance(generation, int) or generation < 0:
            raise AssertionError(f"route is missing lease generation: {row}")
        positions.add(position)
        route_groups.setdefault((position, layer), []).append(rank)
        lookup_groups.setdefault(lookup_id, []).append(row)

    if not set(range(prompt_count)).issubset(positions):
        raise AssertionError(
            f"route trace missed prompt positions: expected 0..{prompt_count - 1}, "
            f"got {sorted(positions)}"
        )
    if max_new > 1 and prompt_count not in positions:
        raise AssertionError("route trace missed the first decode input position")

    for key, ranks in route_groups.items():
        ordered = sorted(ranks)
        if ordered != list(range(len(ordered))):
            raise AssertionError(f"non-dense route ranks for {key}: {ranks}")

    for lookup_id, rows in lookup_groups.items():
        expected_fanout = len(rows)
        identities = {
            (row["layer"], row["expert"], row["lease_generation"])
            for row in rows
        }
        if len(identities) != 1:
            raise AssertionError(
                f"lookup {lookup_id} mixed physical identities: {identities}"
            )
        for row in rows:
            if row["lookup_routes"] != expected_fanout:
                raise AssertionError(
                    f"lookup {lookup_id} fanout mismatch: "
                    f"expected {expected_fanout}, got {row['lookup_routes']}"
                )
    if len(lookup_groups) != header["physical_lookups"]:
        raise AssertionError(
            f"physical lookup count mismatch: header={header['physical_lookups']} "
            f"groups={len(lookup_groups)}"
        )

    print(
        "PASS target trace: explicit logical v2 + physical lookup correlation "
        f"({len(selections)} selections, {len(positions)} token positions, "
        f"{len(lookup_groups)} lookups)"
    )


def check_serve(binary: Path, model: Path, case: dict[str, object]) -> None:
    engine = openai_server.Engine(
        binary,
        model,
        max_tokens=int(case["max_new_tokens"]),
        env=dict(os.environ, CTX="128"),
        kv_slots=1,
    )
    try:
        expected = token_prompt(case["greedy_new_ids"])
        for ordinal in range(2):
            pieces: list[str] = []
            stats = engine.generate(
                token_prompt(case["prompt_ids"]),
                int(case["max_new_tokens"]),
                0.0,
                1.0,
                pieces.append,
            )
            actual = "".join(pieces)
            if actual != expected:
                raise AssertionError(
                    f"serve round {ordinal}: expected {expected!r}, got {actual!r}"
                )
            if stats["prompt_tokens"] != len(case["prompt_ids"]):
                raise AssertionError(f"serve round {ordinal}: bad prompt stats {stats}")
    finally:
        engine.close()
    print("PASS target serve: persistent SUBMIT/DATA/DONE protocol is token-exact")


def check_cli_uses_engine_context(binary: Path, model: Path, temporary: Path) -> None:
    prompt_tokens = 520
    prompt_file = temporary / "context-limit-prompt.txt"
    prompt_file.write_text(token_prompt([1] * prompt_tokens), encoding="utf-8")
    result = run(
        "target CLI engine context",
        [
            binary.as_posix(),
            model.as_posix(),
            "--prompt-file",
            prompt_file.as_posix(),
            "--raw-prompt",
            "--max-tokens",
            "1",
        ],
        env=dict(os.environ, CTX="768"),
    )
    tune = re.search(r"TUNE decode: (\d+) tokens in ([0-9.]+)s", result.stdout)
    if not tune:
        raise AssertionError(
            f"target CLI: no TUNE decode line on stdout: {result.stdout!r}"
        )
    if int(tune.group(1)) != 1 or not float(tune.group(2)) > 0:
        raise AssertionError(f"target CLI: implausible TUNE decode line {tune.group(0)!r}")
    print("PASS target CLI: TUNE decode line is present and parseable")

    stats = re.search(r"v4_tokens prompt=(\d+) generated=(\d+)", result.stderr)
    if not stats or tuple(map(int, stats.groups())) != (prompt_tokens, 1):
        raise AssertionError(
            f"CLI did not use the 768-token engine context: {result.stderr}"
        )
    print("PASS target CLI: prompt beyond the old 512-token cap")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()

    binary = args.binary.resolve()
    fixture = args.fixture.resolve()
    reference = json.loads((fixture / "ref.json").read_text(encoding="utf-8"))
    if reference.get("source") != "transformers":
        raise AssertionError("tiny oracle must come from independent Transformers")
    cases = reference["cases"]

    with tempfile.TemporaryDirectory(prefix="colibri-v4-tiny-") as directory:
        temporary = Path(directory)
        check_target(binary, fixture, "short", cases["short"], temporary)
        check_prefix_is_rejected(binary, fixture, cases["short"], temporary)
        check_target(binary, fixture, "compressed", cases["compressed"], temporary)
        check_target(binary, fixture, "long", cases["long"], temporary)

        for ordinal in range(3):
            check_session(
                binary,
                fixture,
                "short",
                cases["short"],
                temporary,
                ordinal=ordinal,
                compatibility_flag=ordinal == 0,
            )
        check_session(binary, fixture, "long", cases["long"], temporary)
        check_route_trace(binary, fixture, cases["short"], temporary)
        check_cli_uses_engine_context(binary, fixture, temporary)
        check_serve(binary, fixture, cases["short"])

    print("PASS tiny DeepSeek V4 target oracle: all checks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
